from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


SOURCE_ROOT = Path(
    os.getenv("LAUNCHER_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
STATIC_DIR = SOURCE_ROOT / "launcher" / "static"
CATALOG_PATH = Path(
    os.getenv("WORKFLOW_CATALOG", SOURCE_ROOT / "catalog" / "workflows.json")
).resolve()
COMFYUI_DIR = Path(
    os.getenv("COMFYUI_DIR", "/workspace/runpod-slim/ComfyUI")
).resolve()
CUSTOM_NODES_DIR = COMFYUI_DIR / "custom_nodes"
COMFYUI_VENV = COMFYUI_DIR / ".venv-cu128"


class InstallCancelled(Exception):
    pass


@dataclass
class JobState:
    status: str = "idle"
    workflow_id: str | None = None
    title: str | None = None
    stage: str = "idle"
    message: str = "Choose a workflow to begin."
    current_file: str | None = None
    file_index: int = 0
    file_count: int = 0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    file_downloaded_bytes: int = 0
    file_total_bytes: int = 0
    bytes_per_second: float = 0
    percent: float = 0
    error: str | None = None
    comfy_url: str = ""
    restart_required: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def export(self) -> dict[str, Any]:
        result = asdict(self)
        result["percent"] = round(max(0, min(100, self.percent)), 1)
        result["bytes_per_second"] = round(max(0, self.bytes_per_second), 1)
        return result


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_catalog() -> dict[str, Any]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Workflow catalog not found: {CATALOG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Workflow catalog is invalid JSON: {exc}") from exc

    workflows = data.get("workflows")
    if not isinstance(workflows, list):
        raise RuntimeError("Workflow catalog must contain a 'workflows' list.")

    seen: set[str] = set()
    for workflow in workflows:
        workflow_id = workflow.get("id")
        if not isinstance(workflow_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", workflow_id):
            raise RuntimeError(f"Invalid workflow id: {workflow_id!r}")
        if workflow_id in seen:
            raise RuntimeError(f"Duplicate workflow id: {workflow_id}")
        seen.add(workflow_id)
    return data


def public_catalog() -> dict[str, Any]:
    catalog = load_catalog()
    allowed = {
        "id",
        "title",
        "description",
        "badge",
        "accent",
        "thumbnail",
        "estimated_size",
        "disabled",
    }
    return {
        "version": catalog.get("version", 1),
        "workflows": [
            {key: value for key, value in workflow.items() if key in allowed}
            for workflow in catalog["workflows"]
        ],
    }


def comfy_public_url() -> str:
    explicit = os.getenv("COMFYUI_PUBLIC_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    pod_id = os.getenv("RUNPOD_POD_ID", "").strip()
    if pod_id:
        return f"https://{pod_id}-8188.proxy.runpod.net"
    return ""


def safe_destination(relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise RuntimeError("A download destination must be relative to the ComfyUI directory.")
    destination = (COMFYUI_DIR / relative_path).resolve()
    if not destination.is_relative_to(COMFYUI_DIR):
        raise RuntimeError(f"Unsafe download destination: {relative_path}")
    return destination


def tokenized_request(file_spec: dict[str, Any]) -> tuple[str, dict[str, str]]:
    url = str(file_spec.get("url", "")).strip()
    if not url.startswith(("https://", "http://")):
        raise RuntimeError(f"Invalid URL for {file_spec.get('name', 'download')}")

    auth = file_spec.get("auth", "none")
    headers = {"User-Agent": "10sorLabs-Model-Grabber/1.0"}

    if auth == "huggingface":
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                f"{file_spec.get('name', 'This file')} requires HF_TOKEN."
            )
        headers["Authorization"] = f"Bearer {token}"
    elif auth == "civitai":
        token = os.getenv("CIVITAI_TOKEN") or os.getenv("CIVITAI_API_TOKEN")
        if not token:
            raise RuntimeError(
                f"{file_spec.get('name', 'This file')} requires CIVITAI_TOKEN."
            )
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["token"] = token
        url = urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    elif auth == "github":
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                f"{file_spec.get('name', 'This file')} requires GITHUB_TOKEN."
            )
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/octet-stream"
    elif auth not in {"none", None, ""}:
        raise RuntimeError(f"Unknown authentication type: {auth}")

    return url, headers


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JobController:
    def __init__(self) -> None:
        self.state = JobState(comfy_url=comfy_public_url())
        self.task: asyncio.Task[None] | None = None
        self.cancel_event = asyncio.Event()
        self.lock = asyncio.Lock()

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.state, key, value)
        self.state.updated_at = utc_now()

    async def start(self, workflow: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            if self.task and not self.task.done():
                raise HTTPException(status_code=409, detail="A workflow is already installing.")
            if workflow.get("disabled"):
                raise HTTPException(status_code=400, detail="This workflow is not available yet.")

            self.cancel_event = asyncio.Event()
            self.state = JobState(
                status="running",
                workflow_id=workflow["id"],
                title=workflow.get("title", workflow["id"]),
                stage="preparing",
                message="Preparing workflow…",
                comfy_url=comfy_public_url(),
                started_at=utc_now(),
            )
            self.task = asyncio.create_task(self._run(workflow))
            return self.state.export()

    async def cancel(self) -> dict[str, Any]:
        if self.task and not self.task.done():
            self.cancel_event.set()
            self.update(message="Cancelling after the current chunk…")
        return self.state.export()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise InstallCancelled()

    async def _run(self, workflow: dict[str, Any]) -> None:
        try:
            if workflow.get("demo"):
                await self._run_demo(workflow)
            else:
                await self._install_workflow(workflow)
            self.update(
                status="complete",
                stage="complete",
                message="Workflow ready.",
                current_file=None,
                file_downloaded_bytes=self.state.file_total_bytes,
                percent=100,
                bytes_per_second=0,
                completed_at=utc_now(),
            )
        except InstallCancelled:
            self.update(
                status="cancelled",
                stage="cancelled",
                message="Installation cancelled. Partial downloads can resume later.",
                bytes_per_second=0,
                completed_at=utc_now(),
            )
        except Exception as exc:
            self.update(
                status="error",
                stage="error",
                message="The workflow could not be installed.",
                error=str(exc),
                bytes_per_second=0,
                completed_at=utc_now(),
            )

    async def _run_demo(self, workflow: dict[str, Any]) -> None:
        duration = float(
            os.getenv(
                "DEMO_DURATION_OVERRIDE",
                workflow.get("demo_seconds", 6),
            )
        )
        duration = max(0.1, duration)
        total = int(workflow.get("demo_bytes", 64 * 1024 * 1024))
        steps = max(10, int(duration * 10))
        started = time.monotonic()
        self.update(
            stage="downloading",
            message="Testing the download engine…",
            current_file="placeholder-model.safetensors",
            file_index=1,
            file_count=1,
            total_bytes=total,
            file_total_bytes=total,
        )
        for step in range(steps + 1):
            self.check_cancelled()
            fraction = step / steps
            downloaded = int(total * fraction)
            elapsed = max(time.monotonic() - started, 0.01)
            self.update(
                downloaded_bytes=downloaded,
                file_downloaded_bytes=downloaded,
                bytes_per_second=downloaded / elapsed,
                percent=fraction * 100,
            )
            await asyncio.sleep(duration / steps)

    async def _wait_for_comfyui(self) -> None:
        timeout = int(os.getenv("COMFYUI_READY_TIMEOUT", "600"))
        started = time.monotonic()
        while not COMFYUI_DIR.exists():
            self.check_cancelled()
            if time.monotonic() - started > timeout:
                raise RuntimeError("ComfyUI did not become ready in time.")
            self.update(
                stage="preparing",
                message="Waiting for the stock ComfyUI setup…",
            )
            await asyncio.sleep(1)

    async def _install_workflow(self, workflow: dict[str, Any]) -> None:
        await self._wait_for_comfyui()
        files = workflow.get("files", [])
        nodes = workflow.get("custom_nodes", [])
        if not files and not nodes:
            raise RuntimeError("This workflow does not define any files or custom nodes.")

        known_total = sum(
            max(0, int(file_spec.get("size_bytes", 0))) for file_spec in files
        )
        completed_bytes = 0
        download_ceiling = 88 if nodes else 99
        self.update(
            stage="downloading",
            message="Downloading workflow files…",
            file_count=len(files),
            total_bytes=known_total,
        )

        timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            for index, file_spec in enumerate(files):
                self.check_cancelled()
                downloaded = await self._download_file(
                    client,
                    file_spec,
                    index,
                    len(files),
                    completed_bytes,
                    known_total,
                    download_ceiling,
                )
                completed_bytes += downloaded

        if nodes:
            await self._install_custom_nodes(nodes)

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        file_spec: dict[str, Any],
        index: int,
        file_count: int,
        completed_bytes: int,
        known_total: int,
        download_ceiling: float,
    ) -> int:
        name = str(file_spec.get("name") or Path(file_spec["destination"]).name)
        destination = safe_destination(str(file_spec["destination"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = max(0, int(file_spec.get("size_bytes", 0)))
        expected_sha = str(file_spec.get("sha256", "")).lower().strip()

        if destination.exists() and destination.stat().st_size > 0:
            size_matches = not expected_size or destination.stat().st_size == expected_size
            hash_matches = (
                not expected_sha
                or await asyncio.to_thread(file_sha256, destination) == expected_sha
            )
            if size_matches and hash_matches:
                completed = destination.stat().st_size
                fraction = (index + 1) / max(file_count, 1)
                self.update(
                    current_file=name,
                    file_index=index + 1,
                    file_downloaded_bytes=completed,
                    file_total_bytes=completed,
                    downloaded_bytes=completed_bytes + completed,
                    percent=fraction * download_ceiling,
                    message=f"{name} already exists — skipped.",
                )
                return completed

        url, headers = tokenized_request(file_spec)
        partial = destination.with_name(destination.name + ".part")
        partial_size = partial.stat().st_size if partial.exists() else 0
        if partial_size:
            headers["Range"] = f"bytes={partial_size}-"

        self.update(
            stage="downloading",
            message=f"Downloading {name}…",
            current_file=name,
            file_index=index + 1,
            file_downloaded_bytes=partial_size,
            file_total_bytes=expected_size,
        )

        started = time.monotonic()
        request_started_at = partial_size
        try:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"Access denied while downloading {name}. Check the required token."
                    )
                if response.is_error:
                    raise RuntimeError(
                        f"Download failed for {name} (HTTP {response.status_code})."
                    )

                resumed = response.status_code == 206 and partial_size > 0
                mode = "ab" if resumed else "wb"
                if not resumed:
                    partial_size = 0
                    request_started_at = 0

                response_length = int(response.headers.get("content-length", "0") or 0)
                file_total = expected_size or (partial_size + response_length)
                current = partial_size

                with partial.open(mode) as handle:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        self.check_cancelled()
                        handle.write(chunk)
                        current += len(chunk)
                        elapsed = max(time.monotonic() - started, 0.01)
                        speed = (current - request_started_at) / elapsed
                        file_fraction = current / file_total if file_total else 0
                        overall_fraction = (
                            (index + file_fraction) / max(file_count, 1)
                        )
                        aggregate = completed_bytes + current
                        self.update(
                            file_downloaded_bytes=current,
                            file_total_bytes=file_total,
                            downloaded_bytes=aggregate,
                            total_bytes=known_total or file_total,
                            bytes_per_second=speed,
                            percent=overall_fraction * download_ceiling,
                        )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Network error while downloading {name} ({type(exc).__name__})."
            ) from None

        if expected_size and partial.stat().st_size != expected_size:
            raise RuntimeError(
                f"{name} has the wrong size after download; it was left as a .part file."
            )
        if expected_sha:
            self.update(message=f"Verifying {name}…", bytes_per_second=0)
            actual_sha = await asyncio.to_thread(file_sha256, partial)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Checksum verification failed for {name}; the .part file was retained."
                )

        os.replace(partial, destination)
        return destination.stat().st_size

    async def _install_custom_nodes(self, nodes: list[dict[str, Any]]) -> None:
        CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
        for index, node in enumerate(nodes):
            self.check_cancelled()
            name = str(node.get("name", "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
                raise RuntimeError(f"Unsafe custom node name: {name!r}")
            repo = str(node.get("repo", "")).strip()
            if not repo.startswith("https://github.com/"):
                raise RuntimeError(f"Custom node {name} must use a GitHub HTTPS URL.")

            destination = (CUSTOM_NODES_DIR / name).resolve()
            if not destination.is_relative_to(CUSTOM_NODES_DIR):
                raise RuntimeError(f"Unsafe custom node destination: {name}")

            progress = 88 + (index / max(len(nodes), 1)) * 10
            self.update(
                stage="installing",
                message=f"Installing custom node {name}…",
                current_file=name,
                file_index=index + 1,
                file_count=len(nodes),
                percent=progress,
                bytes_per_second=0,
            )

            if not destination.exists():
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "clone",
                    "--filter=blob:none",
                    repo,
                    str(destination),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output, _ = await process.communicate()
                if process.returncode:
                    shutil.rmtree(destination, ignore_errors=True)
                    raise RuntimeError(
                        f"Could not install custom node {name}: "
                        f"{output.decode(errors='replace')[-500:]}"
                    )

            ref = str(node.get("ref", "")).strip()
            if ref:
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    ref,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output, _ = await process.communicate()
                if process.returncode:
                    raise RuntimeError(
                        f"Could not select the pinned version for {name}: "
                        f"{output.decode(errors='replace')[-500:]}"
                    )

            requirements = destination / "requirements.txt"
            if node.get("install_requirements", True) and requirements.exists():
                pip = COMFYUI_VENV / "bin" / "python"
                if not pip.exists():
                    pip = Path("python3.12")
                process = await asyncio.create_subprocess_exec(
                    str(pip),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(requirements),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output, _ = await process.communicate()
                if process.returncode:
                    raise RuntimeError(
                        f"Dependencies failed for {name}: "
                        f"{output.decode(errors='replace')[-500:]}"
                    )
            self.state.restart_required = True

        self.update(percent=99, message="Finishing workflow setup…")


controller = JobController()
app = FastAPI(
    title="10sorLabs Model Grabber",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "catalog": CATALOG_PATH.exists(),
        "comfyui": COMFYUI_DIR.exists(),
    }


@app.get("/api/catalog")
async def catalog() -> dict[str, Any]:
    try:
        return public_catalog()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return controller.state.export()


@app.post("/api/install/{workflow_id}")
async def install(workflow_id: str) -> dict[str, Any]:
    catalog_data = load_catalog()
    workflow = next(
        (item for item in catalog_data["workflows"] if item["id"] == workflow_id),
        None,
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return await controller.start(workflow)


@app.post("/api/cancel")
async def cancel() -> dict[str, Any]:
    return await controller.cancel()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "logo.png", media_type="image/png")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
