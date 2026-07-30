from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


FALLBACK_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/tmp/10sorlabs-runtime")


def enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def github_archive() -> Path:
    repository = os.getenv(
        "LAUNCHER_GITHUB_REPO", "10sorlabs/AI1-Model-Grabber"
    ).strip()
    ref = os.getenv("LAUNCHER_GITHUB_REF", "main").strip()
    if not repository or "/" not in repository or not ref:
        raise RuntimeError("Invalid LAUNCHER_GITHUB_REPO or LAUNCHER_GITHUB_REF.")

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/zipball/{ref}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "10sorLabs-Launcher-Bootstrap/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    temp_file = tempfile.NamedTemporaryFile(
        prefix="10sorlabs-", suffix=".zip", delete=False
    )
    temp_file.close()
    archive = Path(temp_file.name)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return archive
    except Exception:
        archive.unlink(missing_ok=True)
        raise


def extract_runtime(archive: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="10sorlabs-runtime-"))
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging):
                    raise RuntimeError("Unsafe path in GitHub archive.")
            bundle.extractall(staging)

        candidates = [
            path.parent.parent
            for path in staging.rglob("launcher/app.py")
            if (path.parent.parent / "catalog" / "workflows.json").exists()
        ]
        if len(candidates) != 1:
            raise RuntimeError("GitHub archive does not contain a valid launcher.")

        source = candidates[0]
        metadata = {
            "repository": os.getenv("LAUNCHER_GITHUB_REPO"),
            "ref": os.getenv("LAUNCHER_GITHUB_REF"),
        }
        (source / ".launcher-source.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        if RUNTIME_ROOT.exists():
            shutil.rmtree(RUNTIME_ROOT)
        shutil.copytree(source, RUNTIME_ROOT)
        return RUNTIME_ROOT
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def select_source() -> Path:
    if not enabled("LAUNCHER_AUTO_UPDATE", True):
        print("10sorLabs launcher: using the version baked into the image.", flush=True)
        return FALLBACK_ROOT

    archive: Path | None = None
    try:
        print("10sorLabs launcher: checking GitHub for the current UI.", flush=True)
        archive = github_archive()
        source = extract_runtime(archive)
        print("10sorLabs launcher: current GitHub version loaded.", flush=True)
        return source
    except (urllib.error.URLError, zipfile.BadZipFile, RuntimeError, OSError) as exc:
        print(
            "10sorLabs launcher: GitHub update unavailable; "
            f"using the baked fallback ({type(exc).__name__}).",
            flush=True,
        )
        return FALLBACK_ROOT
    finally:
        if archive:
            archive.unlink(missing_ok=True)


def main() -> None:
    source = select_source()
    os.environ["LAUNCHER_SOURCE_ROOT"] = str(source)
    os.chdir(source)
    port = os.getenv("LAUNCHER_PORT", "3000")
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "launcher.app:app",
        "--app-dir",
        str(source),
        "--host",
        "0.0.0.0",
        "--port",
        port,
        "--proxy-headers",
        "--forwarded-allow-ips",
        "*",
    ]
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()

