from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


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
COMFYUI_LOCAL_URL = os.getenv("COMFYUI_LOCAL_URL", "http://127.0.0.1:8188").rstrip("/")
DEFAULT_HF_TOKEN_FILE = Path("/opt/10sorlabs/secrets/hf_token")

# Current built-in model locations from ComfyUI's folder_paths.py, plus the two
# legacy physical directories that ComfyUI still searches for compatible files.
DEFAULT_MODEL_FOLDERS = (
    "checkpoints",
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "clip_vision",
    "loras",
    "vae",
    "vae_approx",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "embeddings",
    "style_models",
    "model_patches",
    "audio_encoders",
    "background_removal",
    "frame_interpolation",
    "geometry_estimation",
    "optical_flow",
    "detection",
    "classifiers",
    "photomaker",
    "gligen",
    "hypernetworks",
    "diffusers",
    "configs",
    "datasets",
)


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
    warnings: list[str] = field(default_factory=list)
    comfy_url: str = ""
    restart_required: bool = False
    comfy_restarted: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def export(self) -> dict[str, Any]:
        result = asdict(self)
        result["percent"] = round(max(0, min(100, self.percent)), 1)
        result["bytes_per_second"] = round(max(0, self.bytes_per_second), 1)
        return result


class CustomModelRequest(BaseModel):
    url: str
    location: str


class CustomNodeRequest(BaseModel):
    url: str


@dataclass
class CustomModelState:
    id: str
    url: str = field(repr=False)
    source_host: str = ""
    location: str = ""
    filename: str = "Resolving filename..."
    status: str = "queued"
    message: str = "Waiting in download queue."
    downloaded_bytes: int = 0
    total_bytes: int = 0
    bytes_per_second: float = 0
    percent: float = 0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def export(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_host": self.source_host,
            "location": self.location,
            "filename": self.filename,
            "status": self.status,
            "message": self.message,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "bytes_per_second": round(max(0, self.bytes_per_second), 1),
            "percent": round(max(0, min(100, self.percent)), 1),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
        }


@dataclass
class CustomNodeState:
    id: str
    url: str = field(repr=False)
    source_host: str = "github.com"
    name: str = "Resolving repository..."
    status: str = "queued"
    message: str = "Waiting in install queue."
    percent: float = 0
    error: str | None = None
    restart_required: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def export(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_host": self.source_host,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "percent": round(max(0, min(100, self.percent)), 1),
            "error": self.error,
            "restart_required": self.restart_required,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
        }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_custom_model_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url or len(url) > 8192:
        raise RuntimeError("Enter a valid model download URL.")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise RuntimeError("Model links must use http:// or https://.")
    return url


def validate_model_location(raw_location: str) -> tuple[str, Path]:
    location = raw_location.strip().replace("\\", "/").strip("/")
    if not location or len(location) > 180:
        raise RuntimeError("Choose a valid model location.")

    relative = PurePosixPath(location)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("The custom model location is not safe.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", part) for part in relative.parts):
        raise RuntimeError(
            "Folder names may contain letters, numbers, spaces, dots, dashes and underscores."
        )

    models_dir = (COMFYUI_DIR / "models").resolve()
    destination = (models_dir / Path(*relative.parts)).resolve()
    if not destination.is_relative_to(models_dir):
        raise RuntimeError("The model location must stay inside ComfyUI/models.")
    return relative.as_posix(), destination


def available_model_locations() -> list[str]:
    locations = list(DEFAULT_MODEL_FOLDERS)
    models_dir = COMFYUI_DIR / "models"
    if models_dir.is_dir():
        for root, directories, _files in os.walk(models_dir, followlinks=False):
            directories[:] = [name for name in directories if not name.startswith(".")]
            root_path = Path(root)
            for name in directories:
                path = root_path / name
                try:
                    relative = path.relative_to(models_dir).as_posix()
                    validate_model_location(relative)
                except (ValueError, RuntimeError):
                    continue
                if relative not in locations:
                    locations.append(relative)
    return locations


def safe_download_filename(raw_name: str) -> str:
    name = unquote(raw_name).replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = name.strip('"\'')
    if not name or name in {".", ".."} or "\x00" in name:
        return "model-download"
    if len(name) > 240:
        suffix = Path(name).suffix[:20]
        name = f"{Path(name).stem[: 240 - len(suffix)]}{suffix}"
    return name


def filename_from_url(url: str) -> str:
    parts = urlsplit(url)
    candidate = Path(parts.path).name
    if not candidate:
        candidate = f"model-{uuid4().hex[:8]}"
    return safe_download_filename(candidate)


def filename_from_response(response: httpx.Response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    extended = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", disposition, re.IGNORECASE)
    regular = re.search(r'filename\s*=\s*"?([^";]+)', disposition, re.IGNORECASE)
    if extended:
        return safe_download_filename(extended.group(1))
    if regular:
        return safe_download_filename(regular.group(1))

    redirected = filename_from_url(str(response.url))
    generic = {"download", "models", "resolve", "main", "model-download"}
    if redirected.lower() not in generic:
        return redirected
    return safe_download_filename(fallback)


def custom_download_request(url: str) -> tuple[str, dict[str, str]]:
    parts = urlsplit(url)
    hostname = (parts.hostname or "").lower()
    headers = {"User-Agent": "10sorLabs-Model-Grabber/1.1"}

    if hostname == "huggingface.co" or hostname.endswith(".huggingface.co"):
        token = huggingface_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif hostname == "civitai.com" or hostname.endswith(".civitai.com"):
        token = (os.getenv("CIVITAI_TOKEN") or os.getenv("CIVITAI_API_TOKEN") or "").strip()
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if token and "token" not in query:
            query["token"] = token
            url = urlunsplit(
                (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
            )

    return url, headers


def response_sha256(response: httpx.Response) -> str:
    for header in ("x-linked-etag", "x-checksum-sha256", "x-amz-checksum-sha256"):
        value = response.headers.get(header, "").strip().strip('"')
        if re.fullmatch(r"[a-fA-F0-9]{64}", value):
            return value.lower()
    return ""


def validate_custom_node_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if not url or len(url) > 2048:
        raise RuntimeError("Enter a valid GitHub repository link.")
    parts = urlsplit(url)
    path_parts = [part for part in parts.path.split("/") if part]
    if (
        parts.scheme != "https"
        or (parts.hostname or "").lower() != "github.com"
        or len(path_parts) != 2
    ):
        raise RuntimeError("Custom nodes must use a GitHub repository link.")
    owner, repository = path_parts
    repository = repository.removesuffix(".git")
    safe_part = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    if not re.fullmatch(safe_part, owner) or not re.fullmatch(safe_part, repository):
        raise RuntimeError("The GitHub repository link is not valid.")
    return f"https://github.com/{owner}/{repository}.git"


def custom_node_name(url: str) -> str:
    name = Path(urlsplit(url).path.rstrip("/")).name.removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise RuntimeError("The custom node repository name is not safe.")
    return name


def normalized_git_remote(url: str) -> str:
    normalized = url.strip().rstrip("/").removesuffix(".git")
    if normalized.startswith("https://github.com/"):
        return normalized.lower()
    return normalized


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


def huggingface_token() -> str:
    token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if token:
        return token

    token_file = Path(
        os.getenv("HF_TOKEN_FILE", str(DEFAULT_HF_TOKEN_FILE))
    ).expanduser()
    try:
        return token_file.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return ""


def tokenized_request(file_spec: dict[str, Any]) -> tuple[str, dict[str, str]]:
    url = str(file_spec.get("url", "")).strip()
    if not url.startswith(("https://", "http://")):
        raise RuntimeError(f"Invalid URL for {file_spec.get('name', 'download')}")

    auth = file_spec.get("auth", "none")
    headers = {"User-Agent": "10sorLabs-Model-Grabber/1.0"}

    if auth == "huggingface":
        token = huggingface_token()
        if not token:
            raise RuntimeError(
                f"{file_spec.get('name', 'This file')} requires Hugging Face access."
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


class CustomModelController:
    def __init__(self) -> None:
        self.items: dict[str, CustomModelState] = {}
        self.pending: deque[str] = deque()
        self.worker_task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        active_statuses = {"queued", "downloading", "error"}
        queue = [
            item.export()
            for item in self.items.values()
            if item.status in active_statuses
        ]
        downloaded = [
            item.export()
            for item in reversed(self.items.values())
            if item.status in {"complete", "skipped"}
        ]
        return {
            "locations": available_model_locations(),
            "queue": queue,
            "downloaded": downloaded,
        }

    async def enqueue(self, raw_url: str, raw_location: str) -> dict[str, Any]:
        url = validate_custom_model_url(raw_url)
        location, _destination = validate_model_location(raw_location)
        item = CustomModelState(
            id=uuid4().hex,
            url=url,
            source_host=(urlsplit(url).hostname or "download").lower(),
            location=location,
            filename=filename_from_url(url),
        )

        async with self.lock:
            self.items[item.id] = item
            self.pending.append(item.id)
            if not self.worker_task or self.worker_task.done():
                self.worker_task = asyncio.create_task(self._drain_queue())
        return item.export()

    async def _drain_queue(self) -> None:
        while True:
            async with self.lock:
                if not self.pending:
                    self.worker_task = None
                    return
                item_id = self.pending.popleft()
            item = self.items[item_id]
            await self._run_item(item)

    def update(self, item: CustomModelState, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(item, key, value)
        item.updated_at = utc_now()

    async def _run_item(self, item: CustomModelState) -> None:
        partial: Path | None = None
        try:
            if not COMFYUI_DIR.exists():
                raise RuntimeError("ComfyUI is not ready yet.")

            location, folder = validate_model_location(item.location)
            folder.mkdir(parents=True, exist_ok=True)
            self.update(
                item,
                status="downloading",
                message="Connecting to the model host...",
                started_at=utc_now(),
                error=None,
            )

            url, headers = custom_download_request(item.url)
            timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in {401, 403}:
                        raise RuntimeError("Access denied by the model host.")
                    if response.is_error:
                        raise RuntimeError(
                            f"Download failed (HTTP {response.status_code})."
                        )

                    content_type = response.headers.get("content-type", "").lower()
                    if "text/html" in content_type:
                        raise RuntimeError(
                            "The link returned a web page instead of a model file."
                        )

                    filename = filename_from_response(response, item.filename)
                    destination = (folder / filename).resolve()
                    if not destination.is_relative_to(folder.resolve()):
                        raise RuntimeError("The download filename is not safe.")

                    partial = destination.with_name(f"{destination.name}.part")
                    partial.unlink(missing_ok=True)

                    total = int(response.headers.get("content-length", "0") or 0)
                    linked_size = int(response.headers.get("x-linked-size", "0") or 0)
                    if linked_size > 0:
                        total = linked_size
                    remote_sha = response_sha256(response)

                    self.update(
                        item,
                        location=location,
                        filename=filename,
                        total_bytes=total,
                        message=f"Checking {filename}...",
                    )

                    if destination.is_file() and destination.stat().st_size > 0:
                        same_size = total > 0 and destination.stat().st_size == total
                        same_hash = False
                        if same_size and remote_sha:
                            same_hash = (
                                await asyncio.to_thread(file_sha256, destination)
                                == remote_sha
                            )
                        if same_size and (same_hash or not remote_sha):
                            self.update(
                                item,
                                status="skipped",
                                message="Model already found — download skipped.",
                                downloaded_bytes=destination.stat().st_size,
                                total_bytes=destination.stat().st_size,
                                bytes_per_second=0,
                                percent=100,
                                completed_at=utc_now(),
                            )
                            return

                    started = time.monotonic()
                    downloaded = 0
                    self.update(item, message=f"Downloading {filename}...")
                    with partial.open("wb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            handle.write(chunk)
                            downloaded += len(chunk)
                            elapsed = max(time.monotonic() - started, 0.01)
                            percent = (downloaded / total * 100) if total else 0
                            self.update(
                                item,
                                downloaded_bytes=downloaded,
                                bytes_per_second=downloaded / elapsed,
                                percent=percent,
                            )

            if not partial or not partial.exists():
                raise RuntimeError("The model host returned no file data.")
            if item.total_bytes and partial.stat().st_size != item.total_bytes:
                raise RuntimeError("The downloaded file has an unexpected size.")

            if remote_sha:
                self.update(item, message=f"Verifying {item.filename}...", bytes_per_second=0)
                actual_sha = await asyncio.to_thread(file_sha256, partial)
                if actual_sha != remote_sha:
                    raise RuntimeError("The downloaded file failed checksum verification.")

            os.replace(partial, destination)
            self.update(
                item,
                status="complete",
                message="Download complete.",
                downloaded_bytes=destination.stat().st_size,
                total_bytes=destination.stat().st_size,
                bytes_per_second=0,
                percent=100,
                completed_at=utc_now(),
            )
        except httpx.RequestError as exc:
            if partial:
                partial.unlink(missing_ok=True)
            self.update(
                item,
                status="error",
                message="Download failed.",
                error=f"Network error ({type(exc).__name__}).",
                bytes_per_second=0,
                completed_at=utc_now(),
            )
        except Exception as exc:
            if partial:
                partial.unlink(missing_ok=True)
            self.update(
                item,
                status="error",
                message="Download failed.",
                error=str(exc),
                bytes_per_second=0,
                completed_at=utc_now(),
            )


class CustomNodeController:
    def __init__(self) -> None:
        self.items: dict[str, CustomNodeState] = {}
        self.pending: deque[str] = deque()
        self.worker_task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        active_statuses = {"queued", "cloning", "installing", "error"}
        return {
            "queue": [
                item.export()
                for item in self.items.values()
                if item.status in active_statuses
            ],
            "downloaded": [
                item.export()
                for item in reversed(self.items.values())
                if item.status in {"complete", "skipped"}
            ],
        }

    async def enqueue(self, raw_url: str) -> dict[str, Any]:
        url = validate_custom_node_url(raw_url)
        item = CustomNodeState(
            id=uuid4().hex,
            url=url,
            source_host=(urlsplit(url).hostname or "github.com").lower(),
            name=custom_node_name(url),
        )
        async with self.lock:
            self.items[item.id] = item
            self.pending.append(item.id)
            if not self.worker_task or self.worker_task.done():
                self.worker_task = asyncio.create_task(self._drain_queue())
        return item.export()

    async def _drain_queue(self) -> None:
        while True:
            async with self.lock:
                if not self.pending:
                    self.worker_task = None
                    return
                item_id = self.pending.popleft()
            await self._run_item(self.items[item_id])

    def update(self, item: CustomNodeState, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(item, key, value)
        item.updated_at = utc_now()

    async def _origin_url(self, destination: Path) -> str:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(destination),
            "remote",
            "get-url",
            "origin",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        if process.returncode:
            return ""
        return output.decode(errors="replace").strip()

    async def _run_item(self, item: CustomNodeState) -> None:
        staging: Path | None = None
        try:
            if not COMFYUI_DIR.exists():
                raise RuntimeError("ComfyUI is not ready yet.")

            CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
            destination = (CUSTOM_NODES_DIR / item.name).resolve()
            if not destination.is_relative_to(CUSTOM_NODES_DIR.resolve()):
                raise RuntimeError("The custom node destination is not safe.")

            self.update(
                item,
                status="cloning",
                message=f"Checking {item.name}...",
                percent=5,
                started_at=utc_now(),
                error=None,
            )

            if destination.exists():
                existing_origin = await self._origin_url(destination)
                if existing_origin and normalized_git_remote(existing_origin) == normalized_git_remote(item.url):
                    self.update(
                        item,
                        status="skipped",
                        message="Custom node already found — install skipped.",
                        percent=100,
                        completed_at=utc_now(),
                    )
                    return
                raise RuntimeError(
                    f"A folder named {item.name} already exists but does not match this repository."
                )

            staging = (CUSTOM_NODES_DIR / f".10sorlabs-{item.id}.part").resolve()
            if not staging.is_relative_to(CUSTOM_NODES_DIR.resolve()):
                raise RuntimeError("The temporary custom node path is not safe.")
            shutil.rmtree(staging, ignore_errors=True)

            self.update(
                item,
                message=f"Cloning {item.name}...",
                percent=12,
            )
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--filter=blob:none",
                "--single-branch",
                item.url,
                str(staging),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await process.communicate()
            if process.returncode:
                raise RuntimeError(
                    "Git could not clone this custom node: "
                    f"{output.decode(errors='replace')[-500:]}"
                )

            self.update(item, percent=74, message="Repository cloned.")
            requirements = staging / "requirements.txt"
            if requirements.is_file():
                self.update(
                    item,
                    status="installing",
                    message="Installing Python requirements...",
                    percent=82,
                )
                python = COMFYUI_VENV / "bin" / "python"
                if not python.exists():
                    python = Path(sys.executable)
                process = await asyncio.create_subprocess_exec(
                    str(python),
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
                        "Custom node requirements failed: "
                        f"{output.decode(errors='replace')[-500:]}"
                    )
                self.update(item, percent=96, message="Requirements installed.")

            os.replace(staging, destination)
            staging = None
            self.update(
                item,
                status="complete",
                message="Custom node installed. Restart ComfyUI to load it.",
                percent=100,
                restart_required=True,
                completed_at=utc_now(),
            )
        except Exception as exc:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            self.update(
                item,
                status="error",
                message="Custom node installation failed.",
                error=str(exc),
                percent=0,
                completed_at=utc_now(),
            )


@dataclass
class ComfyServiceState:
    status: str = "idle"
    message: str = "ComfyUI is running."
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def export(self) -> dict[str, Any]:
        return asdict(self)


class ComfyServiceController:
    def __init__(self) -> None:
        self.state = ComfyServiceState()
        self.task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.state, key, value)
        self.state.updated_at = utc_now()

    async def start(self) -> dict[str, Any]:
        async with self.lock:
            if self.task and not self.task.done():
                return self.state.export()
            self.state = ComfyServiceState(
                status="restarting",
                message="Restarting ComfyUI…",
                started_at=utc_now(),
            )
            self.task = asyncio.create_task(self._restart())
            return self.state.export()

    async def wait(self) -> dict[str, Any]:
        task = self.task
        if task:
            await task
        if self.state.status == "error":
            raise RuntimeError(self.state.error or "ComfyUI restart failed.")
        return self.state.export()

    async def _is_ready(self, client: httpx.AsyncClient) -> bool:
        try:
            response = await client.get(f"{COMFYUI_LOCAL_URL}/system_stats")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def _restart(self) -> None:
        timeout = httpx.Timeout(connect=3, read=5, write=5, pool=3)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                manager = await client.get(f"{COMFYUI_LOCAL_URL}/manager/version")
                if manager.status_code != 200:
                    raise RuntimeError(
                        "ComfyUI Manager is unavailable, so ComfyUI could not be restarted."
                    )

                try:
                    response = await client.post(
                        f"{COMFYUI_LOCAL_URL}/manager/reboot",
                        json={},
                    )
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"ComfyUI Manager rejected the restart (HTTP {response.status_code})."
                        )
                except httpx.RequestError:
                    # A successful reboot normally closes the current HTTP connection.
                    pass

                started = time.monotonic()
                saw_offline = False
                while time.monotonic() - started < 120:
                    await asyncio.sleep(1)
                    ready = await self._is_ready(client)
                    saw_offline = saw_offline or not ready
                    if ready and (saw_offline or time.monotonic() - started >= 4):
                        mark_comfy_restart_complete()
                        self.update(
                            status="ready",
                            message="ComfyUI restarted and is ready.",
                            error=None,
                            completed_at=utc_now(),
                        )
                        return
            raise RuntimeError("ComfyUI did not become ready again within two minutes.")
        except Exception as exc:
            self.update(
                status="error",
                message="ComfyUI restart failed.",
                error=str(exc),
                completed_at=utc_now(),
            )


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

    def add_warning(self, warning: str) -> None:
        self.state.warnings.append(warning)
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
            is_demo = bool(workflow.get("demo"))
            if is_demo:
                await self._run_demo(workflow)
            else:
                await self._install_workflow(workflow)
            if not is_demo:
                self.state.restart_required = True
                self.update(
                    stage="restarting",
                    message="Restarting ComfyUI to load the installed workflow…",
                    current_file=None,
                    percent=99,
                    bytes_per_second=0,
                )
                try:
                    await comfy_service_controller.start()
                    await comfy_service_controller.wait()
                    self.state.restart_required = False
                    self.state.comfy_restarted = True
                except Exception as exc:
                    self.add_warning(f"Automatic ComfyUI restart: {exc}")
            warning_count = len(self.state.warnings)
            self.update(
                status="complete",
                stage="complete",
                message=(
                    f"Setup finished with {warning_count} skipped "
                    f"{'item' if warning_count == 1 else 'items'}. Review the warning"
                    f"{'' if warning_count == 1 else 's'} below."
                    if warning_count
                    else (
                        "Workflow ready. ComfyUI restarted automatically."
                        if self.state.comfy_restarted
                        else "Workflow ready."
                    )
                ),
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

    async def _update_comfyui(self) -> None:
        git_directory = COMFYUI_DIR / ".git"
        requirements = COMFYUI_DIR / "requirements.txt"
        if not git_directory.is_dir():
            raise RuntimeError(
                "ComfyUI cannot be updated because its Git repository was not found."
            )

        commands: list[tuple[str, tuple[str | Path, ...]]] = [
            (
                "Configuring the official ComfyUI repository…",
                (
                    "git",
                    "-C",
                    COMFYUI_DIR,
                    "remote",
                    "set-url",
                    "origin",
                    "https://github.com/Comfy-Org/ComfyUI.git",
                ),
            ),
            (
                "Downloading the latest ComfyUI version…",
                ("git", "-C", COMFYUI_DIR, "fetch", "--prune", "origin", "master"),
            ),
            (
                "Installing the latest ComfyUI version…",
                ("git", "-C", COMFYUI_DIR, "reset", "--hard", "origin/master"),
            ),
        ]
        for message, command in commands:
            self.check_cancelled()
            self.update(stage="updating", message=message, bytes_per_second=0)
            returncode, output = await self._run_process(*command)
            if returncode:
                raise RuntimeError(f"ComfyUI update failed: {output[-500:]}")

        if not requirements.is_file():
            raise RuntimeError("ComfyUI requirements.txt was not found after the update.")

        python = COMFYUI_VENV / "bin" / "python"
        if not python.exists():
            python = Path(sys.executable)
        self.update(
            stage="updating",
            message="Installing the latest ComfyUI requirements…",
            bytes_per_second=0,
        )
        returncode, output = await self._run_process(
            python,
            "-m",
            "pip",
            "install",
            "-r",
            requirements,
        )
        if returncode:
            raise RuntimeError(f"ComfyUI requirements failed: {output[-500:]}")

    async def _install_workflow(self, workflow: dict[str, Any]) -> None:
        await self._wait_for_comfyui()
        files = workflow.get("files", [])
        nodes = workflow.get("custom_nodes", [])
        should_update_comfyui = bool(workflow.get("update_comfyui"))
        if not files and not nodes and not should_update_comfyui:
            raise RuntimeError(
                "This workflow does not define any files, custom nodes or updates."
            )

        if should_update_comfyui:
            await self._update_comfyui()

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
                name = str(
                    file_spec.get("name")
                    or Path(str(file_spec.get("destination", "file"))).name
                )
                try:
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
                except InstallCancelled:
                    raise
                except Exception as exc:
                    self.add_warning(f"{name}: {exc}")
                    self.update(
                        message=f"{name} failed — skipped; continuing setup…",
                        percent=((index + 1) / max(len(files), 1)) * download_ceiling,
                        bytes_per_second=0,
                    )

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

    async def _run_process(self, *command: str | Path) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *(str(part) for part in command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        return process.returncode or 0, output.decode(errors="replace")

    async def _install_custom_node(self, node: dict[str, Any]) -> None:
        name = str(node.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise RuntimeError(f"Unsafe custom node name: {name!r}")
        repo = str(node.get("repo", "")).strip()
        if not repo.startswith("https://github.com/"):
            raise RuntimeError(f"Custom node {name} must use a GitHub HTTPS URL.")

        destination = (CUSTOM_NODES_DIR / name).resolve()
        if not destination.is_relative_to(CUSTOM_NODES_DIR.resolve()):
            raise RuntimeError(f"Unsafe custom node destination: {name}")

        if not destination.exists():
            returncode, output = await self._run_process(
                "git",
                "clone",
                "--filter=blob:none",
                repo,
                destination,
            )
            if returncode:
                shutil.rmtree(destination, ignore_errors=True)
                raise RuntimeError(
                    f"Could not install custom node {name}: {output[-500:]}"
                )
        else:
            returncode, origin = await self._run_process(
                "git",
                "-C",
                destination,
                "remote",
                "get-url",
                "origin",
            )
            if returncode or normalized_git_remote(origin) != normalized_git_remote(repo):
                raise RuntimeError(
                    f"The existing {name} folder is not the expected Git repository."
                )

        ref = str(node.get("ref", "")).strip()
        if ref:
            returncode, _ = await self._run_process(
                "git",
                "-C",
                destination,
                "cat-file",
                "-e",
                f"{ref}^{{commit}}",
            )
            if returncode:
                returncode, output = await self._run_process(
                    "git",
                    "-C",
                    destination,
                    "fetch",
                    "--no-tags",
                    "--filter=blob:none",
                    "origin",
                    ref,
                )
                if returncode:
                    raise RuntimeError(
                        f"Could not fetch the pinned version for {name}: {output[-500:]}"
                    )

            returncode, output = await self._run_process(
                "git",
                "-C",
                destination,
                "checkout",
                "--detach",
                ref,
            )
            if returncode:
                raise RuntimeError(
                    f"Could not select the pinned version for {name}: {output[-500:]}"
                )

        self.state.restart_required = True
        requirements = destination / "requirements.txt"
        if node.get("install_requirements", True) and requirements.exists():
            pip = COMFYUI_VENV / "bin" / "python"
            if not pip.exists():
                pip = Path("python3.12")
            returncode, output = await self._run_process(
                pip,
                "-m",
                "pip",
                "install",
                "-r",
                requirements,
            )
            if returncode:
                raise RuntimeError(
                    f"Dependencies failed for {name}: {output[-500:]}"
                )

    async def _install_custom_nodes(self, nodes: list[dict[str, Any]]) -> None:
        CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
        for index, node in enumerate(nodes):
            self.check_cancelled()
            name = str(node.get("name", "")).strip() or f"Custom node {index + 1}"
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
            try:
                await self._install_custom_node(node)
            except InstallCancelled:
                raise
            except Exception as exc:
                self.add_warning(f"{name}: {exc}")
                self.update(
                    message=f"{name} failed — skipped; continuing setup…",
                    percent=88 + ((index + 1) / max(len(nodes), 1)) * 10,
                )

        self.update(percent=99, message="Finishing workflow setup…")


def mark_comfy_restart_complete() -> None:
    controller.state.restart_required = False
    for item in custom_node_controller.items.values():
        if item.restart_required:
            item.restart_required = False
            item.updated_at = utc_now()


comfy_service_controller = ComfyServiceController()
controller = JobController()
custom_model_controller = CustomModelController()
custom_node_controller = CustomNodeController()
app = FastAPI(
    title="10sorLabs Model Grabber",
    version="1.1.0",
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


@app.get("/api/comfy-restart")
async def comfy_restart_status() -> dict[str, Any]:
    return comfy_service_controller.state.export()


@app.post("/api/comfy-restart")
async def restart_comfy() -> dict[str, Any]:
    if comfy_service_controller.task and not comfy_service_controller.task.done():
        return comfy_service_controller.state.export()
    busy = any(
        task and not task.done()
        for task in (
            controller.task,
            custom_model_controller.worker_task,
            custom_node_controller.worker_task,
        )
    )
    if busy:
        raise HTTPException(
            status_code=409,
            detail="Wait for the current installation queue to finish before restarting ComfyUI.",
        )
    return await comfy_service_controller.start()


@app.get("/api/custom-models")
async def custom_models() -> dict[str, Any]:
    return custom_model_controller.snapshot()


@app.post("/api/custom-models")
async def add_custom_model(request: CustomModelRequest) -> dict[str, Any]:
    try:
        return await custom_model_controller.enqueue(request.url, request.location)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/custom-nodes")
async def custom_nodes() -> dict[str, Any]:
    return custom_node_controller.snapshot()


@app.post("/api/custom-nodes")
async def add_custom_node(request: CustomNodeRequest) -> dict[str, Any]:
    try:
        return await custom_node_controller.enqueue(request.url)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "logo.png", media_type="image/png")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
