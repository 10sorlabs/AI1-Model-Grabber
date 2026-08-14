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
from pydantic import BaseModel, SecretStr, field_validator

from launcher import remote


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

# Resolved once; a file may only use the parallel downloader when this is present.
ARIA2C_PATH = shutil.which("aria2c")
# Cleared for the rest of the process the first time an aria2c build refuses
# --checksum, so an unexpected option can never break more than one download.
ARIA2C_SUPPORTS_CHECKSUM = True
if ARIA2C_PATH is None:
    print(
        "10sorLabs launcher: aria2c is not installed; "
        "every file will download on a single connection.",
        flush=True,
    )

# Hosts each credential may be sent to. The catalog can come from a remote API, so a
# token is never applied on the strength of the file spec's `auth` field alone.
AUTH_HOSTS = {
    "huggingface": ("huggingface.co",),
    "civitai": ("civitai.com",),
    "github": ("github.com", "objects.githubusercontent.com"),
}

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


class AccountLoginRequest(BaseModel):
    """Both fields default and coerce so no validation error can echo the password.

    Without the defaults, pydantic reports a missing field with the whole parent dict
    as `input`, so POSTing {"password": "..."} alone returns the password verbatim in
    the 422 body. Without the before-validator, a wrong-typed password is echoed the
    same way. With both, no field-level 422 is reachable.

    A body that is not an object at all (e.g. POST '"hunter2"') still produces a 422
    echoing the raw body. Left as-is deliberately: only a caller sending the password
    as the whole body can trigger it, the response goes only to that caller, and there
    is no CORS policy that would let a browser read it cross-origin. Closing it would
    mean hand-parsing JSON on the auth route or degrading 422s everywhere else.
    """

    email: str = ""
    password: SecretStr = SecretStr("")

    @field_validator("email", "password", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> str:
        # From HTTP this is always raw JSON, but a SecretStr built in Python would
        # otherwise stringify to '**********' - the same trap the route unwrap avoids.
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return "" if value is None else str(value)


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


def _validate_catalog(data: dict[str, Any]) -> None:
    workflows = data.get("workflows")
    if not isinstance(workflows, list):
        raise RuntimeError("Workflow catalog must contain a 'workflows' list.")

    seen: set[str] = set()
    for workflow in workflows:
        if not isinstance(workflow, dict):
            raise RuntimeError(f"Workflow entry is not an object: {workflow!r}")
        workflow_id = workflow.get("id")
        if not isinstance(workflow_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", workflow_id):
            raise RuntimeError(f"Invalid workflow id: {workflow_id!r}")
        if workflow_id in seen:
            raise RuntimeError(f"Duplicate workflow id: {workflow_id}")
        seen.add(workflow_id)


def load_catalog(fresh: bool = False) -> dict[str, Any]:
    # The API decides which URLs this pod receives; the bundled file is the fallback.
    data = remote.fetch_catalog(fresh=fresh)
    if data is not None:
        try:
            _validate_catalog(data)
        except RuntimeError as exc:
            # A malformed server response means standard speed, not a broken pod.
            print(
                f"10sorLabs launcher: remote catalog rejected ({exc}); "
                f"using the bundled catalog.",
                flush=True,
            )
            data = None

    if data is None:
        try:
            data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Workflow catalog not found: {CATALOG_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Workflow catalog is invalid JSON: {exc}") from exc
        # A broken image should fail loudly, so the bundled file still raises.
        _validate_catalog(data)

    return data


def public_catalog() -> dict[str, Any]:
    catalog = load_catalog()
    # Strictly an allowlist: url, destination, sha256, size_bytes, auth and parallel
    # are install-time details and must never reach the browser.
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


def host_matches(hostname: str, allowed: tuple[str, ...]) -> bool:
    return any(hostname == host or hostname.endswith(f".{host}") for host in allowed)


def tokenized_request(file_spec: dict[str, Any]) -> tuple[str, dict[str, str]]:
    url = str(file_spec.get("url", "")).strip()
    if not url.startswith(("https://", "http://")):
        raise RuntimeError(f"Invalid URL for {file_spec.get('name', 'download')}")

    auth = file_spec.get("auth", "none")
    headers = {"User-Agent": "10sorLabs-Model-Grabber/1.1"}

    # Bind every credential to its own hosts. Without this a catalog served by the API
    # could name auth "huggingface" on an attacker's URL and be handed the pod's token.
    if auth in AUTH_HOSTS:
        hostname = (urlsplit(url).hostname or "").lower()
        if not host_matches(hostname, AUTH_HOSTS[auth]):
            raise RuntimeError(
                f"{file_spec.get('name', 'This file')}: {auth} credential refused "
                f"for host {hostname or 'unknown'}."
            )

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


def shared_destinations(catalog: dict[str, Any]) -> dict[str, list[str]]:
    """sha256 -> every destination in the catalog that claims it.

    One file can legitimately be listed under two paths: ComfyUI looks for the Qwen
    text encoder in both models/text_encoders and models/clip, so both entries are
    correct and neither can be removed. Installing both workflows would otherwise pull
    the same 8.66 GB twice. The duplicates are across workflows, so this has to be
    built from the whole catalog rather than from one workflow's file list.
    """
    grouped: dict[str, list[str]] = {}
    for workflow in catalog.get("workflows", []):
        if not isinstance(workflow, dict):
            continue
        for file_spec in workflow.get("files", []) or []:
            if not isinstance(file_spec, dict):
                continue
            # Normalised on insert and on lookup: one uppercase entry would disable
            # this silently, and the file would just download twice with no error.
            sha256 = str(file_spec.get("sha256", "")).lower().strip()
            destination = str(file_spec.get("destination", "")).strip()
            if not sha256 or not destination:
                continue
            paths = grouped.setdefault(sha256, [])
            if destination not in paths:
                paths.append(destination)
    return {sha: paths for sha, paths in grouped.items() if len(paths) > 1}


def link_or_copy(source: Path, target: Path) -> bool:
    """Hard link source to target, falling back to a copy. False if neither worked.

    A hard link costs no disk and both paths live under the same ComfyUI models tree,
    so they are on one filesystem. copy2 covers a filesystem that does not support
    links, and returning False rather than raising keeps this a pure optimisation.
    """
    try:
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
            return True
        except OSError:
            shutil.copy2(source, target)
            return True
    except Exception:
        # Anything at all - out of disk part way through an 8 GB copy included.
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def written_bytes(path: Path) -> int:
    """Bytes actually on disk, not the file's extent.

    aria2c -s16 writes sixteen ranges at their own offsets, so the file is sparse and
    st_size reports the extent. st_blocks counts allocated blocks (512-byte units,
    POSIX). This is a stat call - we still never parse aria2c's output. Windows has no
    st_blocks; the extent is correct there because nothing on Windows runs a real
    segmented download.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0
    blocks = getattr(stat, "st_blocks", None)
    return blocks * 512 if blocks is not None else stat.st_size


def file_sha256(path: Path, on_progress: Any = None) -> str:
    """Hash a file, optionally reporting bytes read so far.

    The callback exists so a multi-gigabyte hash does not freeze the panel. It runs on
    whichever thread calls this, which is an asyncio.to_thread worker.
    """
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            if on_progress is not None:
                read += len(chunk)
                on_progress(read)
    return digest.hexdigest()


def seed_hash_from_partial(digest: Any, path: Path, byte_count: int) -> None:
    """Fold the bytes already on disk into a running hash before a resume.

    Reads exactly byte_count bytes: the Range request continues from that offset, so
    anything past it is not part of what the server is about to append.
    """
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)


def human_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(count) < 1024 or unit == "GB":
            return f"{count:.2f} {unit}" if unit == "GB" else f"{count:.0f} {unit}"
        count /= 1024
    return f"{count:.2f} GB"


def transfer_phrase(
    verb: str,
    byte_count: int,
    seconds: float,
    note: str = "",
) -> str:
    """'downloaded 13.14 GB in 35.4s (371 MB/s, hashed inline)'"""
    rate = byte_count / seconds if seconds > 0.001 else 0
    suffix = f", {note}" if note else ""
    return (
        f"{verb} {human_bytes(byte_count)} in {seconds:.1f}s "
        f"({human_bytes(rate)}/s{suffix})"
    )


def rejects_checksum_option(output: str) -> bool:
    """True when aria2c refused the --checksum option itself.

    Narrow on purpose: the message must name an option-parsing failure *and* mention
    checksum, so a 5xx, a timeout or a genuine mismatch is never mistaken for one.
    """
    lowered = output.lower()
    if "checksum" not in lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "unrecognized option",
            "unrecognised option",
            "unknown option",
            "invalid option",
        )
    )


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
        # sha256 -> other destinations claiming it, from shared_destinations().
        self.shared_destinations: dict[str, list[str]] = {}

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.state, key, value)
        self.state.updated_at = utc_now()

    def add_warning(self, warning: str) -> None:
        self.state.warnings.append(warning)
        self.state.updated_at = utc_now()

    async def start(
        self,
        workflow: dict[str, Any],
        shared: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        async with self.lock:
            if self.task and not self.task.done():
                raise HTTPException(status_code=409, detail="A workflow is already installing.")
            if workflow.get("disabled"):
                raise HTTPException(status_code=400, detail="This workflow is not available yet.")

            self.shared_destinations = shared or {}
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
        file_started = time.monotonic()
        source_url = str(file_spec.get("url", ""))

        if destination.exists() and destination.stat().st_size > 0:
            size_matches = not expected_size or destination.stat().st_size == expected_size
            verify_started = time.monotonic()
            hash_matches = (
                not expected_sha
                or await asyncio.to_thread(
                    file_sha256,
                    destination,
                    self._hash_progress(name, expected_size),
                )
                == expected_sha
            )
            verify_seconds = time.monotonic() - verify_started
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
                self._log_file_timing(
                    name,
                    source_url,
                    [
                        "already present",
                        transfer_phrase("verified", completed, verify_seconds),
                    ],
                    time.monotonic() - file_started,
                )
                return completed

        partial = destination.with_name(destination.name + ".part")

        # The same file can be listed under two destinations, so a second workflow
        # would otherwise re-download gigabytes that are already on disk. Ahead of
        # tokenized_request, so a linkable file needs no token at all.
        twin = self._existing_twin(expected_sha, expected_size, destination)
        if twin is not None:
            self.update(
                stage="downloading",
                message=f"Linking {name} from {twin.name}…",
                current_file=name,
                file_index=index + 1,
                file_downloaded_bytes=0,
                file_total_bytes=expected_size,
                bytes_per_second=0,
            )
            link_started = time.monotonic()
            if await asyncio.to_thread(link_or_copy, twin, partial):
                link_seconds = time.monotonic() - link_started
                verify_started = time.monotonic()
                try:
                    completed = await self._verify_and_place(
                        partial, destination, expected_size, expected_sha, name
                    )
                except RuntimeError as exc:
                    # A twin that does not verify is worth no more than no twin. Clear
                    # it so the httpx branch cannot resume from a wrong-length .part.
                    print(
                        f"10sorLabs launcher: {name} could not be linked from "
                        f"{twin} ({exc}); downloading it instead.",
                        flush=True,
                    )
                    partial.unlink(missing_ok=True)
                else:
                    print(
                        f"10sorLabs launcher: {name} linked from {twin} "
                        f"instead of downloading it again.",
                        flush=True,
                    )
                    fraction = (index + 1) / max(file_count, 1)
                    self.update(
                        current_file=name,
                        file_index=index + 1,
                        file_downloaded_bytes=completed,
                        file_total_bytes=completed,
                        downloaded_bytes=completed_bytes + completed,
                        percent=fraction * download_ceiling,
                        bytes_per_second=0,
                        message=f"{name} linked from {twin.parent.name} — not downloaded again.",
                    )
                    self._log_file_timing(
                        name,
                        source_url,
                        [
                            f"linked from {twin.parent.name} in {link_seconds:.1f}s",
                            transfer_phrase(
                                "verified",
                                completed,
                                time.monotonic() - verify_started,
                            ),
                        ],
                        time.monotonic() - file_started,
                    )
                    return completed
            else:
                print(
                    f"10sorLabs launcher: could not link {name} from {twin}; "
                    f"downloading it instead.",
                    flush=True,
                )

        url, headers = tokenized_request(file_spec)

        # Never open more than one connection to a file that did not opt in:
        # HuggingFace answers parallel range requests with 403 and collapses to
        # ~394 KiB/s, which is worse than a single stream. `is True` rather than
        # bool(): a catalog carrying "parallel": "false" would otherwise be truthy.
        # Authenticated files stay on httpx as well — credentials passed to aria2c
        # would be visible in the process argv.
        auth = file_spec.get("auth", "none")
        use_aria2 = (
            file_spec.get("parallel") is True
            and auth in (None, "", "none")
            and ARIA2C_PATH is not None
        )

        # Bound in one branch each, read by the shared epilogue below.
        verified_externally = False
        inline_digest: str | None = None
        fetch_phrase = ""

        if use_aria2:
            self.check_cancelled()
            control = partial.with_name(partial.name + ".aria2")
            if partial.exists() and not control.exists():
                # aria2c only resumes a .part it wrote and can verify against its own
                # control file. Without one this came from the httpx path or a crash.
                partial.unlink(missing_ok=True)
            # Same units as the poller: a resumed .part is sparse, so measuring the
            # baseline as an extent here would make the first speed reading negative.
            start_size = written_bytes(partial)

            self.update(
                stage="downloading",
                message=f"Downloading {name}…",
                current_file=name,
                file_index=index + 1,
                file_downloaded_bytes=start_size,
                file_total_bytes=expected_size,
            )
            fetch_started = time.monotonic()
            verified_externally = await self._download_with_aria2c(
                url,
                partial,
                name,
                index,
                file_count,
                completed_bytes,
                known_total,
                download_ceiling,
                expected_size,
                start_size,
                expected_sha,
            )
            fetch_seconds = time.monotonic() - fetch_started
            # Same units as start_size, so a resumed file reports only the new bytes.
            fetch_phrase = transfer_phrase(
                "aria2c", max(0, written_bytes(partial) - start_size), fetch_seconds
            )
        else:
            control = partial.with_name(partial.name + ".aria2")
            if control.exists():
                # This .part belongs to aria2c, and with --file-allocation=none its
                # st_size is the full file length while most of it is holes. Resuming
                # from it would send a Range past the real data and hash a file that is
                # mostly zeroes, so it goes and this path starts clean.
                partial.unlink(missing_ok=True)
                control.unlink(missing_ok=True)
            # Deliberately st_size, not written_bytes(): this is a byte offset for a
            # Range header, and blocks are not an offset. The guard above is what makes
            # extent and bytes written the same number here.
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

                    # Hash as the bytes stream past rather than reading the finished
                    # file back. Seeded only here, never before the request: a server
                    # that ignores Range answers 200 and the write below truncates, so
                    # seeding earlier would digest bytes that never reach the file.
                    hasher = hashlib.sha256() if expected_sha else None
                    if hasher is not None and resumed:
                        await asyncio.to_thread(
                            seed_hash_from_partial, hasher, partial, partial_size
                        )

                    response_length = int(response.headers.get("content-length", "0") or 0)
                    file_total = expected_size or (partial_size + response_length)
                    current = partial_size

                    with partial.open(mode) as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            self.check_cancelled()
                            handle.write(chunk)
                            if hasher is not None:
                                hasher.update(chunk)
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

            inline_digest = hasher.hexdigest() if hasher is not None else None
            fetch_seconds = time.monotonic() - started
            fetch_phrase = transfer_phrase(
                "downloaded",
                max(0, current - request_started_at),
                fetch_seconds,
                note="hashed inline" if inline_digest is not None else "",
            )

        verify_started = time.monotonic()
        completed = await self._verify_and_place(
            partial,
            destination,
            expected_size,
            expected_sha,
            name,
            digest=None if use_aria2 else inline_digest,
            verified_externally=verified_externally,
        )
        verify_seconds = time.monotonic() - verify_started

        phases = [fetch_phrase]
        if verified_externally:
            phases.append("verified inline by aria2c")
        elif expected_sha and inline_digest is None:
            # aria2c refused --checksum, so this fell back to a second pass. That is
            # the case the timing split is here to make visible.
            phases.append(transfer_phrase("verified", completed, verify_seconds))
        self._log_file_timing(
            name, source_url, phases, time.monotonic() - file_started
        )
        return completed

    def _existing_twin(
        self,
        expected_sha: str,
        expected_size: int,
        destination: Path,
    ) -> Path | None:
        """Another destination for the same sha256 that is already on disk, or None.

        Requires a known size: without one there is nothing cheap to check before
        committing to a copy, so it is safer to download.
        """
        if not expected_sha or not expected_size:
            return None
        for relative in self.shared_destinations.get(expected_sha.lower().strip(), ()):
            try:
                candidate = safe_destination(relative)
            except RuntimeError:
                continue
            if candidate == destination:
                continue
            try:
                if candidate.is_file() and candidate.stat().st_size == expected_size:
                    return candidate
            except OSError:
                continue
        return None

    def _log_file_timing(
        self,
        name: str,
        url: str,
        phases: list[str],
        total_seconds: float,
    ) -> None:
        """One permanent line per file: where it came from, and where the time went.

        Every performance question about this launcher so far has been answered by
        guessing from file mtimes, twice wrongly. The host is the raw hostname rather
        than a hand-written "r2"/"huggingface" label, so a silent fallback shows up as
        the hostname changing.
        """
        host = (urlsplit(url).hostname or "unknown").lower()
        print(
            f"10sorLabs launcher: {name} [{host}]: "
            + ", ".join(phase for phase in phases if phase)
            + f", total {total_seconds:.1f}s",
            flush=True,
        )

    def _hash_progress(self, name: str, total: int) -> Any:
        """A file_sha256 callback that keeps the panel moving during a long hash.

        Called from an asyncio.to_thread worker. update() is plain setattr plus a
        timestamp with no lock, so the worst a concurrent status poll can see is a
        snapshot mixing two ticks - fine for a progress display.
        """
        last = 0.0

        def report(done: int) -> None:
            nonlocal last
            now = time.monotonic()
            if now - last < 0.25:
                return
            last = now
            percent = (done / total * 100) if total else 0
            self.update(
                stage="verifying",
                message=f"Verifying {name}… {percent:.0f}%",
                bytes_per_second=0,
            )

        return report

    async def _verify_and_place(
        self,
        partial: Path,
        destination: Path,
        expected_size: int,
        expected_sha: str,
        name: str,
        digest: str | None = None,
        verified_externally: bool = False,
    ) -> int:
        """Size, checksum, move. Shared by the download and link paths alike.

        The checksum can arrive three ways: already confirmed by aria2c as it wrote,
        supplied as a digest computed from the bytes as they streamed past, or - for a
        file that was written earlier and has settled - read back and hashed here.
        """
        # Deliberately st_size, not written_bytes(): a finished file's extent is exactly
        # its size, while blocks are rounded up and would fail this on every file.
        if expected_size and partial.stat().st_size != expected_size:
            raise RuntimeError(
                f"{name} has the wrong size after download; it was left as a .part file."
            )
        if expected_sha and not verified_externally:
            if digest is None:
                # The panel styles this stage distinctly; it is not a download.
                self.update(
                    stage="verifying",
                    message=f"Verifying {name}…",
                    bytes_per_second=0,
                )
                digest = await asyncio.to_thread(
                    file_sha256, partial, self._hash_progress(name, expected_size)
                )
            if digest != expected_sha:
                raise RuntimeError(
                    f"Checksum verification failed for {name}; the .part file was retained."
                )

        os.replace(partial, destination)
        return destination.stat().st_size

    async def _download_with_aria2c(
        self,
        url: str,
        partial: Path,
        name: str,
        index: int,
        file_count: int,
        completed_bytes: int,
        known_total: int,
        download_ceiling: float,
        expected_size: int,
        start_size: int,
        expected_sha: str = "",
    ) -> bool:
        """Fetch one file on sixteen connections. True when aria2c verified it itself.

        aria2c's own output is still never parsed for progress - that comes from the
        .part file's size. Its checksum support, however, is now used deliberately.
        Reading a just-written file back to hash it measured at 60-80 MB/s on a pod
        against 1.7 GB/s for a settled one, which made verification about 80% of a
        19.7 GB install. Handing the digest to aria2c trades Python's SHA-256 for a
        mature, widely deployed downloader's and removes the second pass over the data.
        """
        global ARIA2C_SUPPORTS_CHECKSUM

        started = time.monotonic()
        use_checksum = bool(expected_sha) and ARIA2C_SUPPORTS_CHECKSUM

        async def poll_progress() -> None:
            highest = start_size
            while True:
                await asyncio.sleep(0.5)
                # Returns 0 until aria2c creates the file; the monotonic guard below
                # holds the reading at start_size rather than dropping it to zero.
                current = written_bytes(partial)
                if expected_size:
                    # Whole-block rounding can overshoot the byte count near the end.
                    # Cap before the max, or one over-rounded tick would pin `highest`
                    # above expected_size and the bar would read past 100% for good.
                    current = min(current, expected_size)
                # ext4 delays allocation, so st_blocks can read lower than the previous
                # tick. A bar that goes backwards looks broken.
                highest = max(highest, current)
                current = highest
                elapsed = max(time.monotonic() - started, 0.01)
                speed = (current - start_size) / elapsed
                file_total = expected_size or current
                # Without the guard a catalog that omits size_bytes would make
                # file_total equal current on every tick and the bar would read 100%.
                file_fraction = (current / file_total) if expected_size else 0.0
                overall_fraction = (index + file_fraction) / max(file_count, 1)
                aggregate = completed_bytes + current
                self.update(
                    file_downloaded_bytes=current,
                    file_total_bytes=file_total,
                    downloaded_bytes=aggregate,
                    total_bytes=known_total or file_total,
                    bytes_per_second=speed,
                    percent=overall_fraction * download_ceiling,
                )

        process = await asyncio.create_subprocess_exec(
            "aria2c",
            "-x16",
            "-s16",
            # aria2c uses at most min(-s, size / -k) pieces, and re-splits an idle
            # connection's work only when the remainder is at least -k. At 100M a 50 MB
            # file got one connection and a 230 MB LoRA got two, so they spent the
            # transfer in TCP slow start; and a straggler with 80 MB left could not be
            # re-split, so fifteen connections idled while one crawled - measured on a
            # pod as 7.70 -> 7.72 -> 7.73 GB with the rate falling 157 -> 78 MB/s.
            # Files at or above 1.6 GB are already capped at 16 by -s16, so this only
            # adds parallelism where there was too little.
            "-k",
            "4M",
            "--continue=true",
            # Without this aria2c creates the file at full size before any bytes
            # arrive, so the progress poll reads it as complete on the first tick -
            # and pre-allocated blocks would make written_bytes() lie too. Write-once
            # model files on a container disk; fragmentation does not matter here.
            "--file-allocation=none",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--summary-interval=0",
            "--console-log-level=warn",
            # Verified as the bytes are written, so the file is never read back.
            *(["--checksum=sha-256=" + expected_sha] if use_checksum else []),
            "-d",
            str(partial.parent),
            "-o",
            partial.name,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        poller = asyncio.create_task(poll_progress())
        waiter = asyncio.create_task(process.communicate())
        canceller = asyncio.create_task(self.cancel_event.wait())
        try:
            await asyncio.wait(
                {waiter, canceller},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not waiter.done():
                process.terminate()
                try:
                    # Shielded so the waiter survives the timeout and can still be
                    # awaited after SIGKILL; otherwise the transport is never closed.
                    await asyncio.wait_for(asyncio.shield(waiter), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await waiter
                raise InstallCancelled()

            output, _ = waiter.result()
            if process.returncode:
                text = output.decode(errors="replace") if output else ""
                tail = text[-500:]

                if use_checksum and rejects_checksum_option(text):
                    # This build will not take --checksum. Failing here would break
                    # every file on every pod, so drop the flag for the rest of the
                    # process and fall back to hashing after the download.
                    ARIA2C_SUPPORTS_CHECKSUM = False
                    print(
                        "10sorLabs launcher: this aria2c does not support --checksum; "
                        "falling back to verifying with a second pass.",
                        flush=True,
                    )
                    return await self._download_with_aria2c(
                        url,
                        partial,
                        name,
                        index,
                        file_count,
                        completed_bytes,
                        known_total,
                        download_ceiling,
                        expected_size,
                        start_size,
                        # The flag is already false, so this cannot recurse again.
                        expected_sha="",
                    )

                if process.returncode == 32:
                    # 32 is aria2c's "checksum validation failed": these bytes are known
                    # bad, so neither aria2c nor the httpx branch may resume from them.
                    # Every other non-zero exit - a dropped connection, a timeout, a 5xx,
                    # a retry limit - leaves a legitimately partial file that
                    # --continue=true exists to resume, and deleting that would make a
                    # blip cost a full re-download.
                    partial.unlink(missing_ok=True)
                    partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Checksum verification failed for {name}; "
                        f"the partial download was discarded."
                    )

                raise RuntimeError(
                    f"aria2c failed for {name} (exit {process.returncode}). {tail}".strip()
                )
        finally:
            # The poller must be dead before the caller writes "Verifying…", or its
            # next tick overwrites that message and the stale speed with it.
            for task in (poller, canceller, waiter):
                if not task.done():
                    task.cancel()
            await asyncio.gather(poller, canceller, waiter, return_exceptions=True)

        return use_checksum

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
        # A ref reaches git's argv, and anything starting with "-" is read as an
        # option. This validation is what makes that unreachable from a remote catalog:
        # a [0-9a-f]{40} string cannot begin with "-". cat-file and fetch below also
        # carry --end-of-options, but checkout must not - it reads the flag as the
        # argument to --detach and fails outright on the git the pods run.
        ref = str(node.get("ref", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", ref, re.IGNORECASE):
            raise RuntimeError(
                f"Custom node {name} must pin a 40-character commit sha."
            )

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

        returncode, _ = await self._run_process(
            "git",
            "-C",
            destination,
            "cat-file",
            "-e",
            "--end-of-options",
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
                "--end-of-options",
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
            # No --end-of-options here: git checkout reads it as the argument to
            # --detach ("does not take a path argument"). The 40-hex validation above
            # is what keeps this ref from ever being parsed as an option.
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
        # Off the event loop: the catalog API call blocks for up to ten seconds.
        return await asyncio.to_thread(public_catalog)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return controller.state.export()


@app.post("/api/install/{workflow_id}")
async def install(workflow_id: str) -> dict[str, Any]:
    # fresh=True: the URLs the API hands back are time limited.
    catalog_data = await asyncio.to_thread(load_catalog, True)
    workflow = next(
        (item for item in catalog_data["workflows"] if item["id"] == workflow_id),
        None,
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    # Built from the whole catalog: the duplicates worth linking are cross-workflow.
    return await controller.start(workflow, shared_destinations(catalog_data))


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


def account_snapshot() -> dict[str, Any]:
    # The credential itself is never part of this, masked or otherwise.
    source = remote.credential_source()
    status = remote.fetch_status()
    return {
        "configured": source != "none",
        "source": source,
        "status": status["data"],
        # Why there is no data, so the panel can tell a revoked credential from an
        # outage and offer sign-in rather than telling the user to wait.
        "service": status["reason"],
    }


@app.get("/api/account")
async def account() -> dict[str, Any]:
    # fetch_status blocks for up to ten seconds, so keep it off the event loop.
    return await asyncio.to_thread(account_snapshot)


@app.post("/api/account/login")
async def account_login(request: AccountLoginRequest) -> dict[str, Any]:
    if remote.credential_source() == "env":
        raise HTTPException(
            status_code=409,
            detail=(
                "This pod's licence key comes from the LCT_LICENSE_KEY template "
                "variable. Remove it to sign in here instead."
            ),
        )

    email = request.email.strip()
    # str(SecretStr(...)) is '**********', so unwrap here rather than in remote.login.
    password = request.password.get_secret_value()
    if not email or not password:
        # Deliberately the same message as a wrong password: telling the two apart
        # would make this an account-enumeration oracle.
        raise HTTPException(status_code=401, detail="Email or password not recognised.")

    result = await asyncio.to_thread(remote.login, email, password)
    if not result.get("ok"):
        error = str(result.get("error", "Account service unavailable."))
        # A missing service is misconfiguration, not a rejected credential.
        status_code = 503 if error == "No account service configured." else 401
        raise HTTPException(status_code=status_code, detail=error)

    # Built from the login result rather than a second fetch_status call: one round
    # trip, and the two answers cannot disagree.
    return {
        "configured": True,
        "source": remote.credential_source(),
        "status": {
            key: result[key]
            for key in ("tier", "email", "expires_at")
            if key in result
        },
        "service": "ok",
    }


@app.post("/api/account/logout")
async def account_logout() -> dict[str, Any]:
    if remote.credential_source() == "env":
        raise HTTPException(
            status_code=409,
            detail=(
                "This pod's licence key comes from the LCT_LICENSE_KEY template "
                "variable. There is nothing to sign out of."
            ),
        )
    remote.write_credential("")
    return await asyncio.to_thread(account_snapshot)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "logo.png", media_type="image/png")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
