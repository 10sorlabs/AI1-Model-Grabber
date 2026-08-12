"""Optional subscriber catalog and account API.

The launcher asks this module for a catalog and downloads whatever list it is handed.
It never inspects subscription state: the API decides which URLs a pod receives, and a
pod with no API configured simply falls back to the catalog baked into the image.

Self-contained on purpose - ``launcher.app`` imports this module, never the reverse.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx


CATALOG_TTL = 60
STATUS_TTL = 30
FAILURE_TTL = 15
REQUEST_TIMEOUT = 10
USER_AGENT = "10sorLabs-Model-Grabber/1.1"

_MISS = object()

_state_lock = threading.Lock()
_cache: tuple[float, dict[str, Any] | None] | None = None
_status_cache: tuple[float, dict[str, Any] | None] | None = None
_logged_no_api = False
_logged_credential_source = False
_last_failure_reason: str | None = None


def _api_base() -> str:
    """Read at call time; a module constant would freeze before tests can patch it."""
    return os.getenv("LCT_API_BASE", "").strip().rstrip("/")


def _token_file() -> Path:
    # /workspace, not the runtime root: bootstrap wipes /tmp/10sorlabs-runtime on
    # every boot (see launcher/bootstrap.py RUNTIME_ROOT).
    return Path(os.getenv("LCT_TOKEN_FILE", "/workspace/.lct"))


def _log(message: str) -> None:
    print(f"10sorLabs launcher: {message}", flush=True)


def _reset_state() -> None:
    """Clear every cached value and one-shot log flag (used by the tests)."""
    global _cache, _status_cache
    global _logged_no_api, _logged_credential_source, _last_failure_reason
    with _state_lock:
        _cache = None
        _status_cache = None
        _logged_no_api = False
        _logged_credential_source = False
        _last_failure_reason = None


def _invalidate_caches() -> None:
    """Drop cached answers only.

    Deliberately narrow: the one-shot log flags and the failure-reason suppressor are
    left alone. Clearing those would make every sign-in and sign-out reprint the
    startup lines, and would let a flapping API log on every call again.
    """
    global _cache, _status_cache
    with _state_lock:
        _cache = None
        _status_cache = None


def _credential_and_source() -> tuple[str, str]:
    value = os.getenv("LCT_LICENSE_KEY", "").strip()
    if value:
        return value, "env"
    try:
        value = _token_file().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        value = ""
    if value:
        return value, "file"
    return "", "none"


def read_credential() -> str:
    value, source = _credential_and_source()
    global _logged_credential_source
    with _state_lock:
        first_use = not _logged_credential_source
        _logged_credential_source = True
    if first_use:
        _log(f"catalog credential source: {source}.")
    return value


def credential_source() -> str:
    """Where the credential comes from - never the credential itself."""
    return _credential_and_source()[1]


def write_credential(value: str) -> None:
    try:
        cleaned = (value or "").strip()
        path = _token_file()
        if not cleaned:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cleaned, encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        return
    finally:
        # Anything fetched under the previous credential is no longer the right answer.
        _invalidate_caches()


def _auth_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    credential = read_credential()
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    pod_id = os.getenv("RUNPOD_POD_ID", "").strip()
    if pod_id:
        headers["X-Pod-Id"] = pod_id
    return headers


def _read_cache(status: bool = False) -> Any:
    with _state_lock:
        entry = _status_cache if status else _cache
        if entry is None:
            return _MISS
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            return _MISS
        return value


def _remember_success(data: dict[str, Any]) -> dict[str, Any]:
    global _cache, _last_failure_reason
    with _state_lock:
        _cache = (time.monotonic() + CATALOG_TTL, data)
        # Reset the suppression so a flapping API logs again next time it breaks.
        _last_failure_reason = None
    return data


def _remember_failure(reason: str) -> None:
    global _cache, _last_failure_reason
    with _state_lock:
        _cache = (time.monotonic() + FAILURE_TTL, None)
        repeated = _last_failure_reason == reason
        _last_failure_reason = reason
    if not repeated:
        _log(f"Catalog API unavailable ({reason}); using the bundled catalog.")
    return None


def _remember_status(reason: str, data: dict[str, Any] | None) -> dict[str, Any]:
    global _status_cache
    result = {"reason": reason, "data": data}
    with _state_lock:
        ttl = STATUS_TTL if reason == "ok" else FAILURE_TTL
        _status_cache = (time.monotonic() + ttl, result)
    return result


def _log_no_api_configured() -> None:
    global _logged_no_api
    with _state_lock:
        first_time = not _logged_no_api
        _logged_no_api = True
    if first_time:
        _log("No catalog API configured; using the bundled catalog.")


def _has_integrity_metadata(data: dict[str, Any]) -> bool:
    """Every remote file must carry a sha256 and a real size.

    The download path only verifies when a checksum is present, so a catalog that
    omits one would let the API write arbitrary bytes into ComfyUI/models. The
    bundled catalog keeps its existing behaviour; this gate is remote-only.
    """
    for workflow in data["workflows"]:
        if not isinstance(workflow, dict):
            return False
        files = workflow.get("files", [])
        if not isinstance(files, list):
            return False
        for file_spec in files:
            if not isinstance(file_spec, dict):
                return False
            sha256 = file_spec.get("sha256")
            if not isinstance(sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", sha256, re.IGNORECASE
            ):
                return False
            size = file_spec.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                return False
    return True


def fetch_catalog(fresh: bool = False) -> dict[str, Any] | None:
    """Return the remote catalog, or None so the caller falls back to disk.

    ``fresh=True`` bypasses the cache; installs use it because the URLs the API
    hands back are time limited.
    """
    if not fresh:
        cached = _read_cache()
        if cached is not _MISS:
            return cached

    base = _api_base()
    if not base:
        # Not a failure: this is the normal state of a pod with no API configured.
        _log_no_api_configured()
        return None

    try:
        response = httpx.get(
            f"{base}/v1/catalog",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return _remember_failure(f"HTTP {response.status_code}")
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("workflows"), list):
            return _remember_failure("no workflows list")
        if not _has_integrity_metadata(data):
            return _remember_failure("missing sha256")
        return _remember_success(data)
    except Exception as exc:
        return _remember_failure(type(exc).__name__)


def fetch_status() -> dict[str, Any]:
    """Always a dict: {"reason": ..., "data": ...}. Never None.

    The reason matters as much as the data. Collapsing every failure into "no answer"
    would make a revoked credential look identical to an outage, so the panel would
    tell a lapsed subscriber to wait rather than to sign in again.

      unconfigured   no API base - the normal state of a pod, nothing is wrong
      unauthenticated  401 or 403 - the credential is no longer accepted
      ok             200 with a dict body
      unavailable    anything else, a network error, or a body we cannot read

    Cached like the catalog, so a dead API costs one timeout rather than one a call:
    GET /api/account calls this on every page load.
    """
    cached = _read_cache(status=True)
    if cached is not _MISS:
        return cached

    base = _api_base()
    if not base:
        return _remember_status("unconfigured", None)

    try:
        response = httpx.get(
            f"{base}/v1/status",
            headers=_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in {401, 403}:
            return _remember_status("unauthenticated", None)
        if response.status_code != 200:
            return _remember_status("unavailable", None)
        data = response.json()
    except Exception:
        return _remember_status("unavailable", None)

    if not isinstance(data, dict):
        return _remember_status("unavailable", None)
    return _remember_status("ok", data)


def login(email: str, password: str) -> dict[str, Any]:
    """Exchange an email and password for a stored token. Never raises.

    Both parameters are plain strings: unwrapping pydantic's SecretStr is the caller's
    job, because ``str(SecretStr("x"))`` is '**********' and would be sent verbatim.
    The password is never written to disk, never logged and never returned - only the
    token is persisted.
    """
    base = _api_base()
    if not base:
        return {"ok": False, "error": "No account service configured."}

    try:
        response = httpx.post(
            f"{base}/v1/auth/login",
            json={"email": email, "password": password},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in {401, 403}:
            return {"ok": False, "error": "Email or password not recognised."}
        if response.status_code == 429:
            return {"ok": False, "error": "Too many attempts. Wait a minute."}
        if response.status_code != 200:
            return {"ok": False, "error": "Account service unavailable."}

        data = response.json()
        token = data.get("token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token.strip():
            return {"ok": False, "error": "Account service unavailable."}

        write_credential(token)
        result: dict[str, Any] = {"ok": True}
        for key in ("tier", "email", "expires_at"):
            if key in data:
                result[key] = data[key]
        return result
    except Exception:
        return {"ok": False, "error": "Account service unavailable."}
