import asyncio
import contextlib
import errno
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ["RUNPOD_POD_ID"] = "test-pod"

launcher_app = importlib.import_module("launcher.app")
launcher_remote = importlib.import_module("launcher.remote")

# Captured before any fixture can replace it: pin_the_hosts_filesystem swaps this out for
# every test, and the probe's own test is the one place that has to run the real syscalls.
probe_block_accounting = launcher_app._probe_block_accounting


@pytest.fixture(autouse=True)
def reset_remote_catalog_state():
    """The catalog cache and its one-shot log flags outlive a single test."""
    launcher_remote._reset_state()
    yield
    launcher_remote._reset_state()


@pytest.fixture(autouse=True)
def pin_the_hosts_filesystem(monkeypatch):
    """No test may depend on this machine's device layout, free space or block accounting.

    _download_file calls scratch_partial_for on every download, and _scratch_dir is a
    module global that outlives a test. So on any machine where _resolve_scratch_dir()
    happens to find a second device - a container with a tmpfs /tmp over an overlayfs /,
    which is the CI shape - the whole suite would silently start staging, and
    test_parallel_file_downloads_through_aria2c asserts on a -o filename that staging
    rewrites. None here means "resolved: no scratch device", not "not yet resolved".

    The same for the block-accounting probe, which would otherwise ask whatever
    filesystem pytest's tmp_path landed on and answer differently on Windows, on ext4
    and in a container. Pinned honest, so the default path behaves as it does on a pod
    that stages.

    Both are opt-out: a test that wants staging or a lying volume sets its own value and
    monkeypatch restores these afterwards. .github/workflows/docker-publish.yml runs the
    suite before it builds the image, so a host-dependent test blocks publishing.

    The probe is pinned at the syscall layer rather than at blocks_are_real, so every
    test still runs the real caching and the real one-line downgrade log. Fresh cache
    objects per test, so one test's verdict cannot leak into another's directory.
    """
    monkeypatch.setattr(launcher_app, "_scratch_dir", None)
    monkeypatch.setattr(launcher_app, "_probe_block_accounting", lambda _directory: True)
    monkeypatch.setattr(launcher_app, "_block_accounting", {})
    monkeypatch.setattr(launcher_app, "_block_accounting_logged", set())


@contextlib.contextmanager
def catalog_api(body: bytes, status: int = 200, captured: list | None = None):
    """Serve one canned response on 127.0.0.1 and yield its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if captured is not None:
                captured.append(self.headers)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class AccountApiStub:
    """Records every request so tests can assert what actually went over the wire."""

    def __init__(self) -> None:
        self.login_bodies: list[dict] = []
        self.paths: list[str] = []


@contextlib.contextmanager
def account_api(
    login_status: int = 200,
    login_body: dict | None = None,
    status_status: int = 200,
):
    stub = AccountApiStub()
    payload = json.dumps(login_body if login_body is not None else {}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            stub.paths.append(self.path)
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                stub.login_bodies.append(json.loads(raw))
            except ValueError:
                stub.login_bodies.append({})
            self._respond(login_status, payload if login_status == 200 else b"{}")

        def do_GET(self) -> None:
            stub.paths.append(self.path)
            if status_status != 200:
                self._respond(status_status, b"{}")
                return
            self._respond(200, json.dumps({"tier": "fast"}).encode("utf-8"))

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def closed_port() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    return port


def remote_catalog_bytes(files: list | None = None) -> bytes:
    if files is None:
        files = [
            {
                "name": "Remote model",
                "url": "https://cdn.example/remote.safetensors",
                "destination": "models/checkpoints/remote.safetensors",
                "size_bytes": 1024,
                "sha256": "a" * 64,
                "auth": "none",
                "parallel": True,
            }
        ]
    return json.dumps(
        {
            "version": 3,
            "workflows": [
                {
                    "id": "remote-workflow",
                    "title": "Remote Workflow",
                    "description": "Served by the catalog API.",
                    "estimated_size": "Approx. 1 KB",
                    "files": files,
                    "custom_nodes": [],
                }
            ],
        }
    ).encode("utf-8")


def download_one_file(controller, file_spec: dict, known_total: int = 0) -> int:
    async def runner() -> int:
        timeout = launcher_app.httpx.Timeout(connect=30, read=None, write=30, pool=30)
        async with launcher_app.httpx.AsyncClient(
            follow_redirects=True, timeout=timeout
        ) as client:
            return await controller._download_file(
                client, file_spec, 0, 1, 0, known_total, 99
            )

    return asyncio.run(runner())


def test_health_and_public_catalog() -> None:
    with TestClient(launcher_app.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.get("/api/catalog")
        assert response.status_code == 200
        workflows = response.json()["workflows"]
        assert len(workflows) == 6
        assert sum(not item.get("disabled", False) for item in workflows) == 6
        assert "files" not in workflows[0]
        assert "custom_nodes" not in workflows[0]
        assert "url" not in workflows[0]


def test_catalog_contains_installers_but_no_product_workflows() -> None:
    catalog = launcher_app.load_catalog()
    enabled = [item for item in catalog["workflows"] if not item.get("disabled")]

    assert [item["id"] for item in enabled] == [
        "image-generation",
        "krea-2",
        "dataset-generator",
        "image-edit",
        "motion-control",
        "minimax-h3",
    ]
    assert all(item["files"] for item in enabled)
    assert all(item["custom_nodes"] for item in enabled)

    for installer in enabled:
        for file_spec in installer["files"]:
            destination = file_spec["destination"].lower()
            assert not destination.endswith(".json")
            assert "workflow" not in destination
            assert file_spec["size_bytes"] > 0
            assert len(file_spec["sha256"]) == 64
            assert file_spec["auth"] in {"none", "huggingface"}


def test_krea_2_installer_matches_the_runpod_manifest() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(item for item in catalog["workflows"] if item["id"] == "krea-2")

    assert installer["estimated_size"] == "Approx. 18.4 GB"
    assert [item["destination"] for item in installer["files"]] == [
        "models/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "models/vae/qwen_image_vae.safetensors",
        "models/loras/MysticXXX_KREA2_v3.safetensors",
        "models/loras/pawg_krea2.safetensors",
        "models/loras/RealisticSnapshotKrea2.safetensors",
        "models/upscale_models/4xNMKDSuperscale_4xNMKDSuperscale.pt",
        "models/ultralytics/bbox/face_yolov8m.pt",
        "models/sams/sam_vit_b_01ec64.pth",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "rgthree-comfy",
        "ComfyUI-Impact-Pack",
        "ComfyUI-Impact-Subpack",
        "ComfyUI-KJNodes",
        "RES4LYF",
    ]

    res4lyf_refs = {
        node["ref"]
        for workflow in catalog["workflows"]
        for node in workflow.get("custom_nodes", [])
        if node["name"] == "RES4LYF"
    }
    assert res4lyf_refs == {
        "e716cd1cb2c5cff90131bf4914b75b75a0489d48",
    }


def test_minimax_h3_installer_matches_the_runpod_manifest() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(
        item for item in catalog["workflows"] if item["id"] == "minimax-h3"
    )

    assert installer["estimated_size"] == "Approx. 63.4 GB"
    assert installer["update_comfyui"] is True
    assert [item["destination"] for item in installer["files"]] == [
        "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "models/vae/minimax_h3_video_vae_fp16.safetensors",
        "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "ComfyUI-KJNodes",
        "rgthree-comfy",
        "ComfyUI-VideoHelperSuite",
    ]
    assert all("/resolve/main/" in item["url"] for item in installer["files"])


def test_local_windows_installers_match_the_catalog() -> None:
    catalog = launcher_app.load_catalog()
    workflows = {item["id"]: item for item in catalog["workflows"]}
    installers = {
        "dataset_generator_model_installer.bat": "dataset-generator",
        "krea2_model_installer.bat": "krea-2",
        "minimax_h3_model_installer.bat": "minimax-h3",
    }

    for filename, workflow_id in installers.items():
        script = (
            launcher_app.SOURCE_ROOT / "local-installers" / filename
        ).read_text(encoding="utf-8")
        workflow = workflows[workflow_id]
        downloads = [
            (url, destination.replace("\\", "/"), sha256)
            for url, destination, sha256 in re.findall(
                r'^call :download "([^"]+)" "([^"]+)" "([0-9a-f]{64})"',
                script,
                flags=re.MULTILINE,
            )
        ]
        nodes = re.findall(
            r'^call :install_node "([^"]+)" "([^"]+)" "([0-9a-f]{40})"',
            script,
            flags=re.MULTILINE,
        )

        assert downloads == [
            (item["url"], item["destination"], item["sha256"])
            for item in workflow["files"]
        ]
        assert nodes == [
            (item["name"], item["repo"], item["ref"])
            for item in workflow["custom_nodes"]
        ]
        assert "Get-FileHash -Algorithm SHA256" in script
        assert "checkout --detach" in script
        assert "pip install --disable-pip-version-check" in script


def test_unknown_workflow_cannot_start() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.post("/api/install/does-not-exist")
        assert response.status_code == 404


def test_huggingface_auth_can_use_baked_token_file(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_test_only", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_FILE", str(token_file))

    url, headers = launcher_app.tokenized_request(
        {
            "name": "Gated test model",
            "url": "https://huggingface.co/example/model/resolve/main/model.safetensors",
            "auth": "huggingface",
        }
    )

    assert url.endswith("model.safetensors")
    assert headers["Authorization"] == "Bearer hf_test_only"


def test_frontend_is_served() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "10sorLabs Model Grabber" in response.text
        assert "Custom models" in response.text
        assert "Download queue" in response.text
        assert "Custom nodes" in response.text
        assert "Install queue" in response.text
        assert 'id="job-warnings"' in response.text
        assert 'id="restart-button"' in response.text
        assert 'id="custom-node-restart-button"' in response.text

        logo = client.get("/logo.png")
        assert logo.status_code == 200
        assert logo.headers["content-type"] == "image/png"


def test_real_download_writes_and_verifies_file(tmp_path, monkeypatch) -> None:
    payload = b"10sorLabs-download-test-" * 32768
    expected_hash = hashlib.sha256(payload).hexdigest()

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(
        launcher_app,
        "CUSTOM_NODES_DIR",
        comfy_dir / "custom_nodes",
    )

    workflow = {
        "id": "real-download",
        "title": "Real Download",
        "files": [
            {
                "name": "test-model.safetensors",
                "url": f"http://127.0.0.1:{server.server_port}/model",
                "destination": "models/checkpoints/test-model.safetensors",
                "size_bytes": len(payload),
                "sha256": expected_hash,
                "auth": "none",
            }
        ],
        "custom_nodes": [],
    }
    controller = launcher_app.JobController()
    try:
        asyncio.run(controller._run(workflow))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    destination = comfy_dir / "models" / "checkpoints" / "test-model.safetensors"
    assert destination.read_bytes() == payload
    assert controller.state.status == "complete"
    assert controller.state.percent == 100


def test_custom_model_locations_include_defaults_and_existing_folders(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "models" / "sams").mkdir(parents=True)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    locations = launcher_app.available_model_locations()

    assert "checkpoints" in locations
    assert "diffusion_models" in locations
    assert "text_encoders" in locations
    assert "controlnet" in locations
    assert "sams" in locations


def test_custom_model_location_cannot_escape_models(tmp_path, monkeypatch) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    try:
        launcher_app.validate_model_location("sams/../../outside")
    except RuntimeError as exc:
        assert "safe" in str(exc).lower()
    else:
        raise AssertionError("Path traversal should be rejected.")


def test_custom_download_deletes_partial_and_starts_from_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"fresh-custom-model" * 65536
    received_range_headers: list[str | None] = []

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_range_headers.append(self.headers.get("Range"))
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    destination_dir = comfy_dir / "models" / "sams"
    destination_dir.mkdir(parents=True)
    partial = destination_dir / "model.safetensors.part"
    partial.write_bytes(b"corrupt partial data")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    controller = launcher_app.CustomModelController()

    async def run_download() -> launcher_app.CustomModelState:
        item = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/model.safetensors",
            "sams",
        )
        if controller.worker_task:
            await controller.worker_task
        return controller.items[item["id"]]

    try:
        state = asyncio.run(run_download())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    destination = destination_dir / "model.safetensors"
    assert state.status == "complete"
    assert destination.read_bytes() == payload
    assert not partial.exists()
    assert received_range_headers == [None]


def test_custom_download_queue_is_strictly_sequential(tmp_path, monkeypatch) -> None:
    payload = b"queued-model" * 32768
    counter_lock = threading.Lock()
    active_requests = 0
    maximum_active_requests = 0

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal active_requests, maximum_active_requests
            with counter_lock:
                active_requests += 1
                maximum_active_requests = max(maximum_active_requests, active_requests)
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                midpoint = len(payload) // 2
                self.wfile.write(payload[:midpoint])
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(payload[midpoint:])
            finally:
                with counter_lock:
                    active_requests -= 1

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.CustomModelController()

    async def run_downloads() -> list[launcher_app.CustomModelState]:
        first = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/first.safetensors",
            "checkpoints",
        )
        second = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/second.safetensors",
            "loras",
        )
        if controller.worker_task:
            await controller.worker_task
        return [controller.items[first["id"]], controller.items[second["id"]]]

    try:
        states = asyncio.run(run_downloads())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert maximum_active_requests == 1
    assert [state.status for state in states] == ["complete", "complete"]
    assert (comfy_dir / "models" / "checkpoints" / "first.safetensors").exists()
    assert (comfy_dir / "models" / "loras" / "second.safetensors").exists()


def test_existing_custom_model_is_moved_to_downloaded_as_found(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"already-installed-model"

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    destination = comfy_dir / "models" / "vae" / "existing.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.CustomModelController()

    async def run_download() -> launcher_app.CustomModelState:
        item = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/existing.safetensors",
            "vae",
        )
        if controller.worker_task:
            await controller.worker_task
        return controller.items[item["id"]]

    try:
        state = asyncio.run(run_download())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    snapshot = controller.snapshot()
    assert state.status == "skipped"
    assert snapshot["queue"] == []
    assert snapshot["downloaded"][0]["status"] == "skipped"
    assert destination.read_bytes() == payload


def test_custom_node_url_must_be_a_github_repository() -> None:
    assert (
        launcher_app.validate_custom_node_url("https://github.com/example/ComfyUI-Test")
        == "https://github.com/example/ComfyUI-Test.git"
    )

    for invalid in (
        "https://example.com/example/ComfyUI-Test",
        "https://github.com/example/ComfyUI-Test/issues",
        "http://github.com/example/ComfyUI-Test",
    ):
        try:
            launcher_app.validate_custom_node_url(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Invalid custom node URL was accepted: {invalid}")


def test_custom_node_queue_is_strictly_sequential(monkeypatch) -> None:
    controller = launcher_app.CustomNodeController()
    active = 0
    maximum_active = 0
    order: list[str] = []

    async def fake_install(item) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(f"start:{item.name}")
        await asyncio.sleep(0.03)
        controller.update(item, status="complete", percent=100)
        order.append(f"end:{item.name}")
        active -= 1

    monkeypatch.setattr(controller, "_run_item", fake_install)

    async def run_installs() -> None:
        await controller.enqueue("https://github.com/example/Node-One")
        await controller.enqueue("https://github.com/example/Node-Two")
        await controller.enqueue("https://github.com/example/Node-Three")
        if controller.worker_task:
            await controller.worker_task

    asyncio.run(run_installs())

    assert maximum_active == 1
    assert order == [
        "start:Node-One",
        "end:Node-One",
        "start:Node-Two",
        "end:Node-Two",
        "start:Node-Three",
        "end:Node-Three",
    ]


def test_custom_node_clone_requirements_and_existing_detection(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "Example-ComfyUI-Node"
    source.mkdir()
    (source / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (source / "requirements.txt").write_text("# no extra packages\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=10sorLabs Test",
            "-c",
            "user.email=test@10sorlabs.invalid",
            "commit",
            "-m",
            "Initial node",
        ],
        check=True,
        capture_output=True,
    )

    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes_dir = comfy_dir / "custom_nodes"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(launcher_app, "validate_custom_node_url", lambda url: url)

    controller = launcher_app.CustomNodeController()
    source_url = source.resolve().as_uri()

    async def install_twice() -> tuple[launcher_app.CustomNodeState, launcher_app.CustomNodeState]:
        first = await controller.enqueue(source_url)
        if controller.worker_task:
            await controller.worker_task
        second = await controller.enqueue(source_url)
        if controller.worker_task:
            await controller.worker_task
        return controller.items[first["id"]], controller.items[second["id"]]

    first_state, second_state = asyncio.run(install_twice())
    destination = custom_nodes_dir / "Example-ComfyUI-Node"

    assert first_state.status == "complete"
    assert first_state.restart_required is True
    assert second_state.status == "skipped"
    assert (destination / "__init__.py").exists()
    assert not list(custom_nodes_dir.glob(".10sorlabs-*.part"))


def test_workflow_fetches_a_missing_pinned_custom_node_commit(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes_dir = comfy_dir / "custom_nodes"
    destination = custom_nodes_dir / "ComfyUI-KJNodes"
    destination.mkdir(parents=True)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)

    controller = launcher_app.JobController()
    commands: list[tuple[str, ...]] = []
    repo = "https://github.com/kijai/ComfyUI-KJNodes.git"
    ref = "1289b52fbb6d64a339a4047b9ea74cf7758ccf1e"

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        commands.append(normalized)
        if "remote" in normalized:
            return 0, repo + "\n"
        if "cat-file" in normalized:
            return 1, "missing"
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI-KJNodes",
                "repo": repo,
                "ref": ref,
                "install_requirements": True,
            }
        )
    )

    assert any("fetch" in command and ref in command for command in commands)
    assert any("checkout" in command and ref in command for command in commands)
    assert controller.state.restart_required is True

    # git checkout reads --end-of-options as the argument to --detach and fails
    # outright; the 40-hex validation is what keeps the ref from parsing as an
    # option. cat-file and fetch do accept it, so they keep it.
    checkout = next(command for command in commands if "checkout" in command)
    assert "--detach" in checkout
    assert ref in checkout
    assert "--end-of-options" not in checkout
    assert any(
        "cat-file" in command and "--end-of-options" in command for command in commands
    )
    assert any(
        "fetch" in command and "--end-of-options" in command for command in commands
    )


def test_workflow_skips_failed_custom_node_and_continues(
    tmp_path,
    monkeypatch,
) -> None:
    custom_nodes_dir = tmp_path / "ComfyUI" / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)
    controller = launcher_app.JobController()
    attempted: list[str] = []

    async def fake_install(node, *, on_step=None) -> None:
        attempted.append(node["name"])
        if node["name"] == "Broken-Node":
            raise RuntimeError("simulated node failure")

    monkeypatch.setattr(controller, "_install_custom_node", fake_install)

    asyncio.run(
        controller._install_custom_nodes(
            [
                {"name": "Broken-Node"},
                {"name": "Working-Node"},
            ]
        )
    )

    assert attempted == ["Broken-Node", "Working-Node"]
    assert controller.state.percent == 99
    assert controller.state.warnings == [
        "Broken-Node: simulated node failure",
    ]


def test_workflow_skips_failed_model_and_finishes_with_warning(monkeypatch) -> None:
    controller = launcher_app.JobController()
    attempted: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            return {"status": "restarting"}

        async def wait(self) -> dict:
            return {"status": "ready"}

    async def ready() -> None:
        return None

    async def fake_download(
        _client,
        file_spec,
        _index,
        _file_count,
        _completed_bytes,
        _known_total,
        _download_ceiling,
    ) -> int:
        attempted.append(file_spec["name"])
        if file_spec["name"] == "Broken model":
            raise RuntimeError("simulated download failure")
        return 10

    monkeypatch.setattr(controller, "_wait_for_comfyui", ready)
    monkeypatch.setattr(controller, "_download_file", fake_download)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "continue-test",
                "title": "Continue Test",
                "files": [
                    {
                        "name": "Broken model",
                        "destination": "models/checkpoints/broken.safetensors",
                        "size_bytes": 10,
                    },
                    {
                        "name": "Working model",
                        "destination": "models/checkpoints/working.safetensors",
                        "size_bytes": 10,
                    },
                ],
                "custom_nodes": [],
            }
        )
    )

    assert attempted == ["Broken model", "Working model"]
    assert controller.state.status == "complete"
    assert controller.state.percent == 100
    assert controller.state.warnings == [
        "Broken model: simulated download failure",
    ]
    assert "1 skipped item" in controller.state.message


def test_comfyui_manager_restart_waits_until_comfyui_is_ready(monkeypatch) -> None:
    states = iter([503, 200])
    marked_ready: list[bool] = []

    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str):
            if url.endswith("/manager/version"):
                return Response(200)
            return Response(next(states))

        async def post(self, _url: str, **_kwargs):
            raise launcher_app.httpx.RemoteProtocolError("expected reboot disconnect")

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(launcher_app.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(launcher_app.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        launcher_app,
        "mark_comfy_restart_complete",
        lambda: marked_ready.append(True),
    )
    service = launcher_app.ComfyServiceController()

    async def restart() -> None:
        await service.start()
        await service.wait()

    asyncio.run(restart())

    assert service.state.status == "ready"
    assert service.state.error is None
    assert marked_ready == [True]


def test_every_real_workflow_automatically_restarts_comfyui(
    monkeypatch,
) -> None:
    controller = launcher_app.JobController()
    calls: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            calls.append("start")
            return {"status": "restarting"}

        async def wait(self) -> dict:
            calls.append("wait")
            return {"status": "ready"}

    async def fake_install(_workflow) -> None:
        return None

    monkeypatch.setattr(controller, "_install_workflow", fake_install)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "restart-test",
                "title": "Restart Test",
                "files": [],
                "custom_nodes": [],
            }
        )
    )

    assert calls == ["start", "wait"]
    assert controller.state.status == "complete"
    assert controller.state.restart_required is False
    assert controller.state.comfy_restarted is True


def test_demo_workflow_does_not_restart_comfyui(monkeypatch) -> None:
    controller = launcher_app.JobController()
    calls: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            calls.append("start")
            return {"status": "restarting"}

        async def wait(self) -> dict:
            calls.append("wait")
            return {"status": "ready"}

    async def fake_demo(_workflow) -> None:
        return None

    monkeypatch.setattr(controller, "_run_demo", fake_demo)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "foundation-test",
                "title": "Foundation Test",
                "demo": True,
            }
        )
    )

    assert calls == []
    assert controller.state.status == "complete"
    assert controller.state.comfy_restarted is False


def test_comfyui_update_uses_official_master_and_runtime_python(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / ".git").mkdir(parents=True)
    (comfy_dir / "requirements.txt").write_text("", encoding="utf-8")
    comfy_python = comfy_dir / ".venv-cu128" / "bin" / "python"
    comfy_python.parent.mkdir(parents=True)
    comfy_python.touch()

    controller = launcher_app.JobController()
    commands: list[tuple[str, ...]] = []

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        commands.append(tuple(str(part) for part in command))
        return 0, "ok"

    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(controller._update_comfyui())

    assert commands == [
        # The two probes. This fake answers "ok" to everything, which is not a sha, so the
        # 40-hex guard refuses the shortcut and the full path below still runs.
        ("git", "-C", str(comfy_dir), "rev-parse", "HEAD"),
        ("git", "ls-remote", "https://github.com/Comfy-Org/ComfyUI.git", "master"),
        (
            "git",
            "-C",
            str(comfy_dir),
            "remote",
            "set-url",
            "origin",
            "https://github.com/Comfy-Org/ComfyUI.git",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "fetch",
            "--prune",
            "origin",
            "master",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "reset",
            "--hard",
            "origin/master",
        ),
        # HEAD after the reset. "ok" is not a sha either, so the pip install below is not
        # skipped - only a pair of real, equal shas may skip it.
        ("git", "-C", str(comfy_dir), "rev-parse", "HEAD"),
        (
            str(comfy_python),
            "-m",
            "pip",
            "install",
            # No --no-build-isolation on this path: ComfyUI's own requirements are all
            # wheels and nothing here was ever measured as slow. See the custom-node
            # install for the path where it matters.
            "--timeout",
            "15",
            "--retries",
            "3",
            "-r",
            str(comfy_dir / "requirements.txt"),
        ),
    ]


def test_catalog_api_is_skipped_when_no_base_is_configured(monkeypatch) -> None:
    monkeypatch.delenv("LCT_API_BASE", raising=False)

    assert launcher_remote.fetch_catalog() is None
    assert len(launcher_app.load_catalog()["workflows"]) == 6


def test_catalog_api_failures_fall_back_to_the_bundled_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / "absent.lct")

    monkeypatch.setenv("LCT_API_BASE", f"http://127.0.0.1:{closed_port()}")
    launcher_remote._reset_state()
    assert launcher_remote.fetch_catalog() is None
    assert len(launcher_app.load_catalog()["workflows"]) == 6

    with catalog_api(b"upstream exploded", status=500) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        launcher_remote._reset_state()
        assert launcher_remote.fetch_catalog() is None
        assert len(launcher_app.load_catalog()["workflows"]) == 6

    with catalog_api(b'{"workflows": [') as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        launcher_remote._reset_state()
        assert launcher_remote.fetch_catalog() is None
        assert len(launcher_app.load_catalog()["workflows"]) == 6


def test_catalog_request_omits_authorization_without_a_credential(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / "absent.lct")
    captured: list = []

    with catalog_api(remote_catalog_bytes(), captured=captured) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        assert launcher_remote.fetch_catalog() is not None

    assert captured[0].get("Authorization") is None
    assert captured[0].get("X-Pod-Id") == "test-pod"


def test_catalog_credential_prefers_the_env_var_over_the_token_file(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    token_file.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)
    captured: list = []

    with catalog_api(remote_catalog_bytes(), captured=captured) as base:
        monkeypatch.setenv("LCT_API_BASE", base)

        monkeypatch.setenv("LCT_LICENSE_KEY", "env-key")
        launcher_remote._reset_state()
        assert launcher_remote.fetch_catalog(fresh=True) is not None

        monkeypatch.delenv("LCT_LICENSE_KEY")
        launcher_remote._reset_state()
        assert launcher_remote.fetch_catalog(fresh=True) is not None

    assert captured[0].get("Authorization") == "Bearer env-key"
    assert captured[1].get("Authorization") == "Bearer file-key"


def test_malformed_remote_catalog_falls_back_instead_of_breaking_the_pod(
    monkeypatch,
) -> None:
    # One bad row from the API must mean standard speed, not a 500 on every pod.
    for body in (
        json.dumps(
            {"version": 3, "workflows": [{"id": "Not A Valid Id", "files": []}]}
        ).encode("utf-8"),
        json.dumps({"version": 3, "workflows": ["oops"]}).encode("utf-8"),
        json.dumps(
            {
                "version": 3,
                "workflows": [
                    {"id": "twice", "files": []},
                    {"id": "twice", "files": []},
                ],
            }
        ).encode("utf-8"),
    ):
        with catalog_api(body) as base:
            monkeypatch.setenv("LCT_API_BASE", base)
            launcher_remote._reset_state()
            catalog = launcher_app.load_catalog(fresh=True)
        assert len(catalog["workflows"]) == 6


def test_a_malformed_bundled_catalog_still_raises(tmp_path, monkeypatch) -> None:
    # A broken image should fail loudly; only the remote path falls back.
    broken = tmp_path / "workflows.json"
    broken.write_text(
        json.dumps({"version": 3, "workflows": [{"id": "Not A Valid Id"}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("LCT_API_BASE", raising=False)
    monkeypatch.setattr(launcher_app, "CATALOG_PATH", broken)

    with pytest.raises(RuntimeError, match="Invalid workflow id"):
        launcher_app.load_catalog()


def test_remote_catalog_without_a_checksum_falls_back_to_the_bundled_catalog(
    monkeypatch,
) -> None:
    body = remote_catalog_bytes(
        files=[
            {
                "name": "Unverifiable model",
                "url": "https://cdn.example/unverifiable.safetensors",
                "destination": "models/checkpoints/unverifiable.safetensors",
                "size_bytes": 1024,
                "auth": "none",
            }
        ]
    )

    with catalog_api(body) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        assert launcher_remote.fetch_catalog(fresh=True) is None
        assert len(launcher_app.load_catalog()["workflows"]) == 6


def test_public_catalog_never_leaks_install_details(monkeypatch) -> None:
    private = {"url", "destination", "sha256", "size_bytes", "auth", "parallel"}

    with catalog_api(remote_catalog_bytes()) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        remote_public = launcher_app.public_catalog()

    monkeypatch.delenv("LCT_API_BASE", raising=False)
    launcher_remote._reset_state()
    bundled_public = launcher_app.public_catalog()

    for catalog in (remote_public, bundled_public):
        assert catalog["workflows"]
        for workflow in catalog["workflows"]:
            assert not private & set(workflow)
            assert "files" not in workflow
            assert "custom_nodes" not in workflow


def fake_aria2c(monkeypatch, payload: bytes, recorded: list, returncode: int = 0):
    class FakeProcess:
        def __init__(self, target) -> None:
            self.target = target
            self.returncode = returncode

        async def communicate(self):
            self.target.write_bytes(payload)
            return b"", b""

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        recorded.append(argv)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1])

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)


def test_parallel_file_downloads_through_aria2c(tmp_path, monkeypatch) -> None:
    payload = b"aria2c-payload-" * 4096
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    fake_aria2c(monkeypatch, payload, recorded)
    controller = launcher_app.JobController()

    written = download_one_file(
        controller,
        {
            "name": "Parallel model",
            "url": "https://cdn.example/parallel.safetensors",
            "destination": "models/checkpoints/parallel.safetensors",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "auth": "none",
            "parallel": True,
        },
    )

    destination = comfy_dir / "models" / "checkpoints" / "parallel.safetensors"
    assert len(recorded) == 1
    # The whole argv, so no flag can change silently. -k 4M keeps small files
    # parallel and lets idle connections take over a straggler's tail;
    # --file-allocation=none is what makes the progress poll measurable.
    assert recorded[0][:11] == (
        "aria2c",
        "-x16",
        "-s16",
        "-k",
        "4M",
        "--continue=true",
        "--file-allocation=none",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        # notice + a 30s summary, so a download that succeeds slowly still says what it
        # did. Until this changed, aria2c's output was read only on a non-zero exit.
        "--summary-interval=30",
        "--console-log-level=notice",
    )
    # Verified as it writes, so the file is never read back to hash it.
    assert recorded[0][11] == f"--checksum=sha-256={hashlib.sha256(payload).hexdigest()}"
    assert recorded[0][12] == "-d"
    assert recorded[0][-3:] == (
        "-o",
        "parallel.safetensors.part",
        "https://cdn.example/parallel.safetensors",
    )
    assert not any(part.startswith("--header") for part in recorded[0])
    assert destination.read_bytes() == payload
    assert written == len(payload)
    assert not destination.with_name(destination.name + ".part").exists()


def test_files_not_flagged_parallel_stay_on_a_single_stream(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"single-stream-payload" * 2048
    expected_hash = hashlib.sha256(payload).hexdigest()

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    fake_aria2c(monkeypatch, b"", recorded)
    controller = launcher_app.JobController()

    # "false" is a non-empty string and 1 is truthy, so only an identity check on
    # True keeps a server-side typo off the parallel path.
    variants = [("missing", {}), ("string", {"parallel": "false"}), ("int", {"parallel": 1})]
    try:
        for label, extra in variants:
            download_one_file(
                controller,
                {
                    "name": f"Single {label}",
                    "url": f"http://127.0.0.1:{server.server_port}/{label}",
                    "destination": f"models/checkpoints/{label}.safetensors",
                    "size_bytes": len(payload),
                    "sha256": expected_hash,
                    "auth": "none",
                    **extra,
                },
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert recorded == []
    for label, _extra in variants:
        assert (
            comfy_dir / "models" / "checkpoints" / f"{label}.safetensors"
        ).read_bytes() == payload


def test_todays_catalog_shape_installs_entirely_over_httpx(tmp_path, monkeypatch) -> None:
    payload = b"no-parallel-key-anywhere" * 2048
    expected_hash = hashlib.sha256(payload).hexdigest()

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    fake_aria2c(monkeypatch, b"", recorded)

    body = remote_catalog_bytes(
        files=[
            {
                "name": "Legacy model",
                "url": f"http://127.0.0.1:{server.server_port}/legacy",
                "destination": "models/checkpoints/legacy.safetensors",
                "size_bytes": len(payload),
                "sha256": expected_hash,
                "auth": "none",
            }
        ]
    )

    controller = launcher_app.JobController()
    try:
        with catalog_api(body) as base:
            monkeypatch.setenv("LCT_API_BASE", base)
            catalog = launcher_app.load_catalog(fresh=True)
        workflow = catalog["workflows"][0]
        asyncio.run(controller._install_workflow(workflow))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert recorded == []
    assert controller.state.warnings == []
    assert (
        comfy_dir / "models" / "checkpoints" / "legacy.safetensors"
    ).read_bytes() == payload


def test_parallel_file_falls_back_to_httpx_when_aria2c_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"no-aria2c-installed" * 2048
    expected_hash = hashlib.sha256(payload).hexdigest()

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    recorded: list = []
    fake_aria2c(monkeypatch, b"", recorded)
    controller = launcher_app.JobController()

    try:
        download_one_file(
            controller,
            {
                "name": "Parallel but unsupported",
                "url": f"http://127.0.0.1:{server.server_port}/model",
                "destination": "models/checkpoints/fallback.safetensors",
                "size_bytes": len(payload),
                "sha256": expected_hash,
                "auth": "none",
                "parallel": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert recorded == []
    assert (
        comfy_dir / "models" / "checkpoints" / "fallback.safetensors"
    ).read_bytes() == payload


def test_aria2c_checksum_mismatch_aborts_and_keeps_the_part_file(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"corrupted-by-the-mirror" * 2048
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    # Exit 32 is aria2c's own "checksum validation failed".
    fake_aria2c(monkeypatch, payload, recorded, returncode=32)
    controller = launcher_app.JobController()

    destination = comfy_dir / "models" / "checkpoints" / "tampered.safetensors"
    partial = destination.with_name(destination.name + ".part")
    control = partial.with_name(partial.name + ".aria2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    control.write_bytes(b"aria2 control")

    with pytest.raises(RuntimeError, match="Checksum verification failed"):
        download_one_file(
            controller,
            {
                "name": "Tampered model",
                "url": "https://cdn.example/tampered.safetensors",
                "destination": "models/checkpoints/tampered.safetensors",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(b"what we actually asked for").hexdigest(),
                "auth": "none",
                "parallel": True,
            },
        )

    assert not destination.exists()
    # Bytes aria2c has declared corrupt must not survive to be resumed from, by
    # aria2c or by the httpx branch, so the .part and its control file both go.
    assert not partial.exists()
    assert not control.exists()


def test_a_non_checksum_aria2c_failure_keeps_the_partial_for_resume(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"interrupted-transfer" * 2048
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    # Exit 1: a dropped connection, a timeout, a 5xx - not corruption.
    fake_aria2c(monkeypatch, payload, recorded, returncode=1)
    controller = launcher_app.JobController()

    with pytest.raises(RuntimeError, match="aria2c failed"):
        download_one_file(
            controller,
            {
                "name": "Interrupted model",
                "url": "https://cdn.example/interrupted.safetensors",
                "destination": "models/checkpoints/interrupted.safetensors",
                "size_bytes": len(payload) * 4,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "auth": "none",
                "parallel": True,
            },
        )

    partial = (
        comfy_dir / "models" / "checkpoints" / "interrupted.safetensors.part"
    )
    # --continue=true exists to resume this. Deleting it would make a network blip
    # cost a full re-download - up to 63 GB for the largest workflow.
    assert partial.exists()


def test_aria2c_progress_is_polled_from_the_part_file(tmp_path, monkeypatch) -> None:
    payload = b"progress-payload" * 32768
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    class FakeProcess:
        def __init__(self, target) -> None:
            self.target = target
            self.returncode = 0

        async def communicate(self):
            self.target.write_bytes(payload[: len(payload) // 2])
            await asyncio.sleep(0.7)
            self.target.write_bytes(payload)
            return b"", b""

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1])

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)

    controller = launcher_app.JobController()
    observed: list[int] = []
    speeds: list[float] = []
    original_update = controller.update

    def recording_update(**changes) -> None:
        if "file_downloaded_bytes" in changes:
            observed.append(changes["file_downloaded_bytes"])
        if "bytes_per_second" in changes:
            speeds.append(changes["bytes_per_second"])
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)

    settled: list[int] = []

    async def runner() -> None:
        timeout = launcher_app.httpx.Timeout(connect=30, read=None, write=30, pool=30)
        async with launcher_app.httpx.AsyncClient(
            follow_redirects=True, timeout=timeout
        ) as client:
            await controller._download_file(
                client,
                {
                    "name": "Polled model",
                    "url": "https://cdn.example/polled.safetensors",
                    "destination": "models/checkpoints/polled.safetensors",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "auth": "none",
                    "parallel": True,
                },
                0,
                1,
                0,
                len(payload),
                99,
            )
        settled.append(len(observed))
        # Longer than the 0.5s poll interval: a poller still alive would tick here.
        await asyncio.sleep(0.7)

    asyncio.run(runner())

    # The poller reported real mid-flight progress rather than 0 then done.
    assert any(0 < value < len(payload) for value in observed)
    assert any(speed > 0 for speed in speeds)
    assert 0 < controller.state.percent < 100
    # …and it was dead before the caller moved on. The old form of this assertion
    # watched for the "Verifying…" message, which no longer exists on this path now
    # that aria2c checksums as it writes.
    assert len(observed) == settled[0]


def drive_aria2_download(
    tmp_path,
    monkeypatch,
    file_spec,
    payload,
    linger=0.0,
    supports_checksum=True,
    output=b"",
):
    """Run _download_file down the aria2c branch and report what it did.

    aria2c is never executed. create_subprocess_exec is replaced with a stub that writes
    the bytes a real download would have written, so these tests assert on the argument
    list the launcher constructed rather than on any downloader's behaviour.

    file_sha256 is counted rather than stubbed out, because on this path "did we read the
    file back" is the whole question and the argv only answers half of it.

    Returns a dict: argv, stages, error, hash_calls.
    """
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")
    monkeypatch.setattr(launcher_app, "ARIA2C_SUPPORTS_CHECKSUM", supports_checksum)

    hashed: list[Path] = []
    real_sha256 = launcher_app.file_sha256

    def counting_sha256(path, on_progress=None):
        hashed.append(path)
        return real_sha256(path, on_progress)

    monkeypatch.setattr(launcher_app, "file_sha256", counting_sha256)

    captured: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, target) -> None:
            self.target = target
            self.returncode = 0

        async def communicate(self):
            self.target.write_bytes(payload)
            if linger:
                # Held open so the poller sees a complete file next to a live process,
                # which is exactly the state aria2c is in during its checksum pass.
                await asyncio.sleep(linger)
            # What aria2c printed. Only the failure path used to read this.
            return output, b""

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        captured.append(argv)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1])

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)

    controller = launcher_app.JobController()
    stages: list[str] = []
    original_update = controller.update

    def recording_update(**changes) -> None:
        if "stage" in changes:
            stages.append(str(changes["stage"]))
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)

    failure: list[Exception] = []

    async def runner() -> None:
        timeout = launcher_app.httpx.Timeout(connect=30, read=None, write=30, pool=30)
        async with launcher_app.httpx.AsyncClient(
            follow_redirects=True, timeout=timeout
        ) as client:
            try:
                await controller._download_file(
                    client, file_spec, 0, 1, 0, len(payload), 99
                )
            except Exception as exc:
                failure.append(exc)

    asyncio.run(runner())
    return {
        "argv": captured[0] if captured else (),
        "stages": stages,
        "error": failure[0] if failure else None,
        "hash_calls": len(hashed),
    }


def mirrored_spec(payload, **overrides) -> dict:
    spec = {
        "name": "Mirrored model",
        "url": "https://cdn.example/mirrored.safetensors",
        "destination": "models/checkpoints/mirrored.safetensors",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "auth": "none",
        "parallel": True,
    }
    spec.update(overrides)
    return spec


def pretend_free_space(monkeypatch, free_bytes: int) -> None:
    """Answer every disk_usage with this much free, so no test asks the real machine.

    scratch_partial_for wants expected_size + a 2 GB margin, and copy_into_place refuses
    to start when the destination has less room than the source. Both are correct and
    both would otherwise make a passing test a property of the host's spare disk - on a
    CI runner that publishes the image only if the suite passes first.
    """
    monkeypatch.setattr(
        launcher_app.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(free_bytes, 0, free_bytes),
    )


def test_only_a_literal_false_turns_the_digest_off() -> None:
    """The field is an optimisation the server opts into, never one we infer.

    Everything that is not the boolean False means verify, so a server that has not
    learned the field yet, a null, and a hand-edited catalog carrying the string "false"
    all land on the safe side. Same reasoning as `parallel is True` in _download_file.
    """
    size = 1024
    assert launcher_app.should_verify_digest({}, size) is True
    assert launcher_app.should_verify_digest({"verify": True}, size) is True
    assert launcher_app.should_verify_digest({"verify": None}, size) is True
    assert launcher_app.should_verify_digest({"verify": "false"}, size) is True
    assert launcher_app.should_verify_digest({"verify": 0}, size) is True
    assert launcher_app.should_verify_digest({"verify": False}, size) is False

    # The composition that must never leave a file with no gate at all: without a
    # size_bytes the length check cannot run, so the digest stays on regardless.
    assert launcher_app.should_verify_digest({"verify": False}, 0) is True


def test_the_checksum_flag_follows_the_catalogs_verify_field(
    tmp_path,
    monkeypatch,
) -> None:
    """--checksum is what costs the four minutes, so this asserts on the argv itself.

    aria2c's --checksum is not incremental for an HTTP download: it re-reads the finished
    file, which on MooseFS measured 4m41s against 3.75s for the transfer. A file the
    RapidCache server mirrored itself carries verify: false and must not get the flag.
    """
    payload = b"mirrored-payload" * 4096

    # Distinct destinations: the same one twice would make the second call take the
    # "already exists - skipped" branch and never reach aria2c at all.
    trusted = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(
            payload, verify=False, destination="models/checkpoints/trusted.safetensors"
        ),
        payload,
    )
    assert trusted["error"] is None
    assert not [arg for arg in trusted["argv"] if arg.startswith("--checksum")]

    untrusted = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload, destination="models/checkpoints/untrusted.safetensors"),
        payload,
    )
    assert untrusted["error"] is None
    assert (
        "--checksum=sha-256=" + hashlib.sha256(payload).hexdigest()
    ) in untrusted["argv"]


def test_a_skipped_digest_is_not_quietly_rehashed_in_python(
    tmp_path,
    monkeypatch,
) -> None:
    """Dropping --checksum without also clearing the sha makes this slower, not faster.

    _verify_and_place takes verified_externally from _download_with_aria2c's return, which
    is just use_checksum. Remove the flag alone and that becomes False, so a still-populated
    expected_sha sends the file through file_sha256 instead - the same read-back this
    change exists to remove, now in Python rather than aria2c's C, and with a green suite
    because the argv assertion above still passes.

    So this asserts the behaviour rather than the argument: the file is never read back.
    The two other spellings of "correct" also hash zero times, and only the bug hashes
    once, so the count is what separates them - and the ARIA2C_SUPPORTS_CHECKSUM=False leg
    proves the counter is wired to something that can fire.
    """
    payload = b"unhashed-payload" * 4096

    mirrored = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(
            payload, verify=False, destination="models/checkpoints/mirrored.safetensors"
        ),
        payload,
    )
    assert mirrored["error"] is None
    assert mirrored["hash_calls"] == 0

    # Positive control. With --checksum unavailable there is nothing to trust, so the
    # fallback second pass must happen - if this were also 0 the assertion above would be
    # proving only that the probe is dead.
    fallback = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload, destination="models/checkpoints/fallback.safetensors"),
        payload,
        supports_checksum=False,
    )
    assert fallback["error"] is None
    assert fallback["hash_calls"] == 1


def test_resolving_scratch_creates_nothing(tmp_path, monkeypatch) -> None:
    """Importing this module must not touch the filesystem.

    An earlier draft resolved at module scope with a mkdir inside the resolver, so
    `import launcher.app` created /root/.10sorlabs-scratch or /tmp/10sorlabs-scratch on a
    developer machine, in CI, and at pytest collection.
    """
    wanted = tmp_path / "scratch-that-should-not-appear"
    monkeypatch.setenv("LCT_SCRATCH_DIR", str(wanted))
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")

    launcher_app._resolve_scratch_dir()

    assert not wanted.exists()


def test_scratch_is_only_used_when_it_is_a_different_device(
    tmp_path,
    monkeypatch,
) -> None:
    """Different device is the whole point.

    On a pod with no network volume the models tree is already on container disk, and
    staging would buy a second full copy of every byte for nothing.
    """
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")
    candidate = tmp_path / "same-device"
    # Created so the resolver judges the candidate itself rather than walking up to its
    # nearest existing ancestor, which here would be tmp_path - the models tree.
    candidate.mkdir()
    monkeypatch.setenv("LCT_SCRATCH_DIR", str(candidate))

    # tmp_path and the override are the same filesystem, so every candidate is rejected
    # and the caller keeps writing beside the destination.
    assert launcher_app._resolve_scratch_dir() is None

    real_stat = Path.stat

    def pretend_other_device(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if "same-device" in str(self):
            return os.stat_result(
                (result.st_mode, result.st_ino, result.st_dev + 1)
                + tuple(result)[3:]
            )
        return result

    monkeypatch.setattr(Path, "stat", pretend_other_device)
    assert launcher_app._resolve_scratch_dir() == candidate


def test_staging_never_costs_a_download_that_would_otherwise_work(
    tmp_path,
    monkeypatch,
) -> None:
    """Container disk on a stock RunPod template is small.

    A 20 GB model must still install on a pod that cannot stage it, so every one of these
    falls back to writing beside the destination rather than failing.
    """
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(launcher_app, "_scratch_dir", scratch)
    destination = tmp_path / "ComfyUI" / "models" / "checkpoints" / "big.safetensors"
    gigabyte = 1024**3

    def with_free(free_bytes):
        pretend_free_space(monkeypatch, free_bytes)

    # Ample room: 10 GB file, 300 GB free.
    with_free(300 * gigabyte)
    staged = launcher_app.scratch_partial_for(destination, 10 * gigabyte)
    assert staged is not None and staged.parent == scratch

    # Same file, only 11 GB free - inside the 2 GB margin, so no.
    with_free(11 * gigabyte)
    assert launcher_app.scratch_partial_for(destination, 10 * gigabyte) is None

    # A catalog with no size_bytes cannot be judged, so it is not staged.
    with_free(300 * gigabyte)
    assert launcher_app.scratch_partial_for(destination, 0) is None

    # No scratch device at all.
    monkeypatch.setattr(launcher_app, "_scratch_dir", None)
    assert launcher_app.scratch_partial_for(destination, 10 * gigabyte) is None


def test_placement_renames_on_one_device_and_copies_across_two(
    tmp_path,
    monkeypatch,
) -> None:
    """os.replace is free within a filesystem and impossible across one."""
    payload = b"placement-payload" * 4096
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    # Same device: the copy helper must never run.
    def explode(*_args, **_kwargs):
        raise AssertionError("copy_into_place ran for a same-device placement")

    monkeypatch.setattr(launcher_app, "copy_into_place", explode)
    controller = launcher_app.JobController()
    completed, place_seconds = asyncio.run(
        controller._verify_and_place(source, destination, len(payload), "", "Placed")
    )
    assert completed == len(payload)
    assert place_seconds == 0.0
    assert destination.read_bytes() == payload
    assert not source.exists()


def test_a_cross_device_placement_copies_and_lands_byte_identical(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"cross-device-payload" * 8192
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    real_replace = os.replace
    refused: list[int] = []

    def refuse_the_first_rename(src, dst, *args, **kwargs):
        # EXDEV is what a rename from container disk to the models volume actually
        # raises; the sidecar rename inside copy_into_place must still go through.
        if not refused and str(src).endswith("staged.part"):
            refused.append(1)
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(launcher_app.os, "replace", refuse_the_first_rename)

    controller = launcher_app.JobController()
    completed, place_seconds = asyncio.run(
        controller._verify_and_place(source, destination, len(payload), "", "Placed")
    )

    assert refused, "the test never exercised the cross-device path"
    assert completed == len(payload)
    assert place_seconds >= 0
    assert destination.read_bytes() == payload
    assert not source.exists(), "the staged copy was left behind"
    assert not (destination.parent / (destination.name + ".placing")).exists()


def test_a_crash_mid_placement_leaves_nothing_at_the_destination(
    tmp_path,
    monkeypatch,
) -> None:
    """A truncated file at the real path would be trusted forever.

    _download_file opens by checking whether the destination already exists at the right
    length and skipping the download if so. A half-copied file there is not a slow
    install, it is a corrupt model that never gets repaired.
    """
    # Three 8 MiB reads, so raising on the second genuinely lands mid-copy with a partly
    # written sidecar on disk - not after the last chunk, where there is nothing to lose.
    payload = b"x" * (20 * 1024 * 1024)
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    calls = {"n": 0}

    def die_part_way(done: int) -> None:
        calls["n"] += 1
        assert done < len(payload), "the crash must land before the last chunk"
        if calls["n"] >= 2:
            raise RuntimeError("simulated crash mid-copy")

    with pytest.raises(RuntimeError, match="simulated crash"):
        launcher_app.copy_into_place(source, destination, die_part_way)

    assert not destination.exists()
    assert not (destination.parent / (destination.name + ".placing")).exists()
    assert source.exists(), "the staged copy is the only surviving original"


def test_cancel_is_answered_during_a_placement(tmp_path, monkeypatch) -> None:
    """A 20 GB copy is a minute of a Cancel button that does nothing, without this."""
    payload = b"cancelled-payload" * 8192
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    controller = launcher_app.JobController()
    controller.cancel_event.set()

    with pytest.raises(launcher_app.InstallCancelled):
        launcher_app.copy_into_place(
            source, destination, None, controller.check_cancelled
        )

    assert not destination.exists()
    assert not (destination.parent / (destination.name + ".placing")).exists()


def test_placement_refuses_before_copying_when_the_volume_is_full(
    tmp_path,
    monkeypatch,
) -> None:
    """Otherwise ENOSPC surfaces after a complete download, having spent every byte twice."""
    payload = b"too-big-payload" * 4096
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)

    monkeypatch.setattr(
        launcher_app.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(len(payload) // 2, 0, len(payload) // 2),
    )

    with pytest.raises(RuntimeError, match="Not enough room"):
        launcher_app.copy_into_place(source, destination)

    assert not destination.exists()
    assert not (destination.parent / (destination.name + ".placing")).exists()


def test_a_staged_download_writes_to_scratch_and_lands_on_the_volume(
    tmp_path,
    monkeypatch,
) -> None:
    """The staged path, end to end, through the code the panel actually runs.

    The unit tests above cover scratch_partial_for and copy_into_place separately. This
    is the one that fails if _download_file stops routing the .part through scratch, or
    stops placing it afterwards - the seam between them, which no unit test can see.
    """
    payload = b"staged-payload" * 4096
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(launcher_app, "_scratch_dir", scratch)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    result = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload, destination="models/checkpoints/staged.safetensors"),
        payload,
    )

    assert result["error"] is None
    argv = result["argv"]
    # aria2c was pointed at container disk, not at the models tree.
    assert Path(argv[argv.index("-d") + 1]) == scratch
    # Derived from the destination rather than random, so --continue still means
    # something after a restart.
    written = argv[argv.index("-o") + 1]
    assert written.endswith("-staged.safetensors.part") and written != "staged.part"

    destination = tmp_path / "ComfyUI" / "models" / "checkpoints" / "staged.safetensors"
    assert destination.read_bytes() == payload
    # Nothing left behind on either side of the placement.
    assert list(scratch.iterdir()) == []
    assert not (destination.parent / (destination.name + ".placing")).exists()


def test_the_sweep_clears_stale_staging_files_but_not_fresh_ones(
    tmp_path,
    monkeypatch,
) -> None:
    """Cancel keeps partials on purpose; the sweep is only for what a kill leaves.

    The age bound is what keeps "Partial downloads can resume later" true for a launcher
    that crashed and came back a minute ago.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(launcher_app, "_scratch_dir", scratch)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy)

    stale = scratch / "aaaa-old.part"
    stale_control = scratch / "aaaa-old.part.aria2"
    fresh = scratch / "bbbb-new.part"
    sidecar = comfy / "models" / "checkpoints" / "model.safetensors.placing"
    for path in (stale, stale_control, fresh, sidecar):
        path.write_bytes(b"x")

    long_ago = time.time() - 48 * 60 * 60
    for path in (stale, stale_control, sidecar):
        os.utime(path, (long_ago, long_ago))

    removed = launcher_app.sweep_scratch()

    assert removed == 3
    assert not stale.exists() and not stale_control.exists()
    assert not sidecar.exists(), "a .placing left by a kill would sit where ComfyUI scans"
    assert fresh.exists(), "a recent partial is still resumable"


def test_the_sweep_does_nothing_without_a_scratch_device(tmp_path, monkeypatch) -> None:
    """Most of the suite boots the app through TestClient, which runs the lifespan."""
    monkeypatch.setattr(launcher_app, "_scratch_dir", None)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "nothing-here")

    assert launcher_app.sweep_scratch() == 0


def test_the_probe_spots_a_filesystem_that_derives_blocks_from_length(
    tmp_path,
    monkeypatch,
) -> None:
    """The one test that runs the probe's real syscalls, with only the answer faked.

    MooseFS does no allocation accounting: mfs_fuse.c:1127,1135,1143 compute st_blocks as
    (attrlength+511)/512, straight from the file's length. So a 1 MiB sparse file holding
    512 bytes reports every one of those bytes as allocated, and anything downstream that
    reads st_blocks as "bytes written" is reading the extent instead. ext4 answers one
    4 KiB block for the same file - a 256x margin between the two verdicts.
    """

    class FakeStat:
        def __init__(self, size: int, blocks: int) -> None:
            self.st_size = size
            self.st_blocks = blocks

    opened: dict = {}
    real_mkstemp = tempfile.mkstemp
    real_fstat = os.fstat

    def recording_mkstemp(*args, **kwargs):
        handle, name = real_mkstemp(*args, **kwargs)
        opened["fd"] = handle
        opened["name"] = name
        return handle, name

    monkeypatch.setattr(launcher_app.tempfile, "mkstemp", recording_mkstemp)

    def answer_with(size: int, blocks: int) -> None:
        def fake_fstat(fd):
            # Only ever ours: pytest's own capture machinery stats descriptors too, and
            # handing it a FakeStat would break the run rather than the assertion.
            if fd == opened.get("fd"):
                return FakeStat(size, blocks)
            return real_fstat(fd)

        monkeypatch.setattr(launcher_app.os, "fstat", fake_fstat)

    length = launcher_app._PROBE_LENGTH

    # ext4, tmpfs, overlayfs: one block, because one block is what was written.
    answer_with(length, 4096 // 512)
    assert probe_block_accounting(tmp_path) is True

    # MooseFS, using its own arithmetic rather than an approximation of it.
    answer_with(length, (length + 511) // 512)
    assert probe_block_accounting(tmp_path) is False

    # A platform with no allocation accounting at all is not a filesystem that lies:
    # written_bytes falls back to the extent on Windows and that is correct there,
    # because nothing on Windows runs a segmented download.
    class NoBlocks:
        st_size = length

    monkeypatch.setattr(
        launcher_app.os,
        "fstat",
        lambda fd: NoBlocks() if fd == opened.get("fd") else real_fstat(fd),
    )
    assert probe_block_accounting(tmp_path) is True

    # Whatever the verdict, the probe file itself is not left behind.
    assert not Path(opened["name"]).exists()
    assert list(tmp_path.iterdir()) == []


def test_an_unprobeable_directory_is_not_cached_and_says_so_once(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """A directory that cannot be probed loses the progress display. Say so, once.

    This project has been caught more than once by a fallback that logged nothing, which
    is how a fabricated 94% survived long enough to cost a day. A definite verdict is
    cached; "could not tell" is not, so a full or read-only directory is asked again on
    the next file rather than written off for the life of the process.
    """
    verdicts: list = []
    asked: list[Path] = []

    def probe(directory):
        asked.append(Path(directory))
        return verdicts.pop(0)

    monkeypatch.setattr(launcher_app, "_probe_block_accounting", probe)

    # Definite: probed once, cached, and the downgrade is announced exactly once.
    verdicts.extend([False])
    assert launcher_app.blocks_are_real(tmp_path) is False
    assert launcher_app.blocks_are_real(tmp_path) is False
    assert len(asked) == 1
    announced = capsys.readouterr().out
    assert announced.count("10sorLabs launcher:") == 1
    assert "derived from the file's length" in announced

    # Indefinite: asked again every time, still refused, still only one line about it.
    other = tmp_path / "unprobeable"
    other.mkdir()
    verdicts.extend([None, None])
    assert launcher_app.blocks_are_real(other) is False
    assert launcher_app.blocks_are_real(other) is False
    assert asked.count(other) == 2
    announced = capsys.readouterr().out
    assert announced.count("10sorLabs launcher:") == 1
    assert "could not establish block accounting" in announced


def test_written_bytes_refuses_to_guess_where_blocks_are_derived(
    tmp_path,
    monkeypatch,
) -> None:
    """None is not zero: it is "this filesystem cannot answer the question"."""
    partial = tmp_path / "model.safetensors.part"
    partial.write_bytes(b"x" * 8192)

    measured = launcher_app.written_bytes(partial)
    assert measured is not None and measured > 0
    # The poller leans on this: 0 until aria2c creates the file, never None.
    assert launcher_app.written_bytes(tmp_path / "not-created-yet.part") == 0

    monkeypatch.setattr(launcher_app, "_block_accounting", {})
    monkeypatch.setattr(launcher_app, "_probe_block_accounting", lambda _directory: False)
    assert launcher_app.written_bytes(partial) is None


def test_a_verify_false_spec_with_no_size_still_gets_its_checksum(
    tmp_path,
    monkeypatch,
) -> None:
    """The length check and the digest are alternatives; one of them always runs.

    _verify_and_place only compares lengths when expected_size is truthy, so honouring
    verify: false on a spec with no size_bytes would leave the file with no integrity
    check of any kind. That combination keeps the digest instead.
    """
    payload = b"sizeless-payload" * 4096
    spec = mirrored_spec(payload, verify=False)
    spec.pop("size_bytes")

    result = drive_aria2_download(tmp_path, monkeypatch, spec, payload)

    assert result["error"] is None
    assert (
        "--checksum=sha-256=" + hashlib.sha256(payload).hexdigest()
    ) in result["argv"]


def test_a_wrong_length_still_fails_a_file_whose_digest_was_skipped(
    tmp_path,
    monkeypatch,
) -> None:
    """With the digest gone the length check is the only gate, so it has to be real.

    Guards launcher/app.py's `if expected_size and partial.stat().st_size != expected_size`
    against being removed by someone who assumes the checksum covers it.
    """
    payload = b"truncated-payload" * 4096
    # The catalog claims more bytes than aria2c will produce.
    spec = mirrored_spec(payload, verify=False, size_bytes=len(payload) + 4096)

    result = drive_aria2_download(tmp_path, monkeypatch, spec, payload)

    assert isinstance(result["error"], RuntimeError)
    assert "wrong size" in str(result["error"])


def test_the_poller_names_the_checksum_pass_instead_of_freezing(
    tmp_path,
    monkeypatch,
) -> None:
    """A file that has landed but is still being hashed must not look like a stall.

    Every byte is on disk while aria2c is still running, which on this path can only mean
    its checksum pass. This also pins the placement of that block: it reads file_total,
    and poll_progress is a bare create_task nobody awaits, so a NameError there would be
    swallowed and the panel would simply freeze with a clean console.
    """
    payload = b"lingering-payload" * 4096

    result = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload),
        payload,
        # Longer than the 0.5s poll interval, so at least one tick sees the finished
        # file beside a process that has not exited.
        linger=1.2,
    )

    assert result["error"] is None
    assert "verifying" in result["stages"]


def test_the_displayed_rate_is_a_window_not_a_lifetime_average() -> None:
    """The panel read 3.26 GB/s at 9% and 42.9 MB/s at 10% of one install.

    The network did nothing differently; the numerator froze when the bytes landed while
    the denominator kept climbing. A trailing window reports what is happening now.
    """
    window = launcher_app.RateWindow(window=4.0, min_interval=0.1)

    # One sample is not a rate.
    assert window.add(0.0, 0) == 0.0

    # A steady 100 MB/s reads back as 100 MB/s.
    for tick in range(1, 21):
        rate = window.add(tick * 0.5, int(tick * 0.5 * 100e6))
    assert 95e6 < rate < 105e6

    # The bytes stop but the clock does not - a lifetime average would decay slowly
    # while this settles at zero once the window has passed over the stall.
    done = int(10.0 * 100e6)
    for tick in range(1, 11):
        rate = window.add(10.0 + tick * 0.5, done)
    assert rate == 0.0


def test_a_fast_stream_is_measured_over_the_window_not_the_last_few_samples() -> None:
    """The httpx loop samples per 1 MiB, which at 1 GB/s is a thousand calls a second.

    Bounding the deque by sample count would quietly redefine the window as "the last N
    MiB" - a quarter of a second at that rate - so a brief stall at the end would read as
    a total stop. Throttling by time keeps the window four real seconds wide.
    """
    window = launcher_app.RateWindow(window=4.0, min_interval=0.1)

    now, done = 0.0, 0
    window.add(now, done)
    # Three seconds at 1 GB/s, sampled every millisecond: 3000 calls, far more than any
    # plausible sample cap.
    for _ in range(3000):
        now += 0.001
        done += 1_000_000
        window.add(now, done)

    # Then a brief stall, well inside the four-second window.
    for _ in range(300):
        now += 0.001
        rate = window.add(now, done)

    # Most of the window is still the fast stretch, so the reading stays high. A
    # count-bounded window would have forgotten it and reported ~0.
    assert rate > 5e8


def test_aria2c_progress_tracks_bytes_written_not_the_file_extent(
    tmp_path,
    monkeypatch,
) -> None:
    """Progress must follow allocated blocks, never the file's extent.

    Observed on a real 12.2 GiB pod download on 2026-08-12: the bar sat at one value
    for the whole transfer while bytes_per_second decayed 1302 -> 180 MiB/s, because
    aria2c pre-allocates and -s16 writes sixteen ranges at their own offsets. The
    extent inflation was reproduced locally - 16 x 64 KiB written reports st_size at
    94.1%. The stat sequence below is stubbed so the arithmetic is deterministic on
    every platform; the stub is not the only evidence.
    """
    segments = 16
    block = 64 * 1024
    total = 16 * 1024 * 1024
    landed = segments * block

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    partial = comfy_dir / "segmented.safetensors.part"
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    # st_size, st_blocks * 512 - one entry per poll tick.
    sequence = [
        # Sparse mid-download: the extent is near-full, the blocks are not.
        (total * 15 // 16 + block, landed),
        # Whole-block rounding overshoots the byte count at the end.
        (total, total + 4096),
        # ext4 delayed allocation: blocks read lower than the previous tick.
        (total, total // 2),
    ]

    class FakeStat:
        def __init__(self, size: int, written: int) -> None:
            self.st_size = size
            self.st_blocks = written // 512

    gate: dict = {}
    stat_calls: list[int] = []
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self != partial:
            return real_stat(self, *args, **kwargs)
        index = min(len(stat_calls), len(sequence) - 1)
        stat_calls.append(index)
        if len(stat_calls) >= len(sequence) and "done" in gate:
            gate["done"].set()
        return FakeStat(*sequence[index])

    monkeypatch.setattr(Path, "stat", fake_stat)

    class FakeProcess:
        def __init__(self, target) -> None:
            self.target = target
            self.returncode = 0

        async def communicate(self):
            # What aria2c -s16 actually does: sixteen ranges, each written at its own
            # offset, which extends the file well beyond the bytes delivered.
            with open(self.target, "wb") as handle:
                for segment in range(segments):
                    handle.seek(segment * (total // segments))
                    handle.write(b"x" * block)
            # Outlive the stat sequence, so the finally cannot cancel the poller
            # mid-run and leave the assertions racing the machine's speed.
            await gate["done"].wait()
            return b"", b""

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1])

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)

    controller = launcher_app.JobController()
    observed: list[int] = []
    original_update = controller.update

    def recording_update(**changes) -> None:
        if "file_downloaded_bytes" in changes:
            observed.append(changes["file_downloaded_bytes"])
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)

    async def runner() -> None:
        gate["done"] = asyncio.Event()
        await controller._download_with_aria2c(
            "https://cdn.example/segmented.safetensors",
            partial,
            "Segmented model",
            0,
            1,
            0,
            total,
            99,
            total,
            0,
        )

    asyncio.run(runner())

    assert len(stat_calls) >= len(sequence)
    assert len(observed) >= 3

    # The premise, measured rather than assumed: writing at offsets really does
    # inflate the extent. os.stat is untouched by the Path.stat stub.
    assert os.stat(partial).st_size > total * 0.9

    # 1. Tracks the blocks, not the extent - the bug this test exists for.
    assert observed[0] == landed
    assert observed[0] < total * 0.5
    # 2. Capped, so whole-block rounding cannot push percent past 100.
    assert observed[1] == total
    # 3. Never decreases, even when the block count reads lower than last tick.
    assert observed[2] == total
    assert controller.state.percent <= 99


def test_the_panel_reports_nothing_rather_than_94_percent_on_a_lying_volume(
    tmp_path,
    monkeypatch,
) -> None:
    """The test above asserts against the one filesystem where this bug cannot happen.

    MooseFS derives st_blocks from the file's length (mfs_fuse.c:1127,1135,1143), and
    aria2c -s16 opens its sixteenth connection at 15/16 of the file inside the first
    second - so the extent pins at 93.75% immediately and the panel then climbs at one
    connection's rate instead of the transfer's. Every observed stall sat just above
    93.75% and never below: 93.87, 94.42, 94.92, 97.77.

    So: 15% of the bytes have actually arrived, the extent already reads 93.75%, and the
    panel must publish neither that number nor anything derived from it.
    """
    total = 16 * 1024 * 1024
    landed = total * 15 // 100
    extent = total * 15 // 16
    index, file_count, ceiling = 2, 5, 99
    boundary = (index / file_count) * ceiling

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    partial = comfy_dir / "moosefs.safetensors.part"
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")
    monkeypatch.setattr(launcher_app, "_probe_block_accounting", lambda _directory: False)

    class FakeStat:
        def __init__(self, size: int) -> None:
            self.st_size = size
            # The MooseFS client's own arithmetic.
            self.st_blocks = (size + 511) // 512

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self != partial:
            return real_stat(self, *args, **kwargs)
        return FakeStat(extent)

    monkeypatch.setattr(Path, "stat", fake_stat)

    gate: dict = {}
    observed: list[dict] = []

    class FakeProcess:
        def __init__(self, target) -> None:
            self.target = target
            self.returncode = 0

        async def communicate(self):
            # What has genuinely landed - 15%, against an extent already at 93.75%.
            self.target.write_bytes(b"x" * landed)
            await gate["done"].wait()
            return b"", b""

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1])

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)

    controller = launcher_app.JobController()
    original_update = controller.update

    def recording_update(**changes) -> None:
        if "file_downloaded_bytes" in changes:
            observed.append(changes)
            if len(observed) >= 2:
                gate["done"].set()
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)

    async def runner() -> None:
        gate["done"] = asyncio.Event()
        await controller._download_with_aria2c(
            "https://cdn.example/moosefs.safetensors",
            partial,
            "MooseFS model",
            index,
            file_count,
            index * total,
            file_count * total,
            ceiling,
            total,
            0,
        )

    asyncio.run(runner())

    # The premise, not an assumption: 15% of the file is really there, and the number the
    # poller declined to publish is really sitting in the stat.
    assert os.stat(partial).st_size == landed
    monkeypatch.setattr(launcher_app, "_block_accounting", {})
    monkeypatch.setattr(launcher_app, "_probe_block_accounting", lambda _directory: True)
    fabricated = launcher_app.written_bytes(partial)
    assert fabricated / total > 0.93, "the 94% this test exists for is not reproduced"

    # Not one tick of it reached the panel: no byte count, no total, no rate.
    assert len(observed) >= 2
    assert all(change["file_downloaded_bytes"] == 0 for change in observed)
    assert all(change["file_total_bytes"] == 0 for change in observed)
    assert all(change["bytes_per_second"] == 0 for change in observed)
    # The bar stays on the file boundary. Trusting the extent would have put it at
    # (2 + 0.9375) / 5 * 99 = 58.2%, climbing on one connection's progress.
    assert all(change["percent"] == pytest.approx(boundary) for change in observed)
    assert controller.state.percent == pytest.approx(boundary)
    # And the aggregate byte counter never took the extent either.
    assert controller.state.downloaded_bytes == 0

    # Indeterminate, but not frozen: elapsed time is the one number here that cannot
    # lie, and without something moving this panel reads as hung - which is the failure
    # this whole investigation began with.
    assert re.search(r"\d+s \(progress not measurable", controller.state.message)
    assert launcher_app.human_duration(0) == "0s"
    assert launcher_app.human_duration(47.9) == "47s"
    assert launcher_app.human_duration(252) == "4m12s"


def test_cancelling_an_aria2c_download_terminates_the_process(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    controller = launcher_app.JobController()
    signals: list[str] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.stopped = asyncio.Event()

        async def communicate(self):
            await self.stopped.wait()
            return b"", b""

        def terminate(self) -> None:
            signals.append("terminate")
            self.returncode = -15
            self.stopped.set()

        def kill(self) -> None:
            signals.append("kill")
            self.returncode = -9
            self.stopped.set()

    async def fake_exec(*_command, **_kwargs):
        # Cancel only once the process is live, so the branch's own pre-flight
        # check_cancelled() is not what ends the download.
        asyncio.get_running_loop().call_later(0.05, controller.cancel_event.set)
        return FakeProcess()

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(launcher_app.InstallCancelled):
        download_one_file(
            controller,
            {
                "name": "Cancelled model",
                "url": "https://cdn.example/cancelled.safetensors",
                "destination": "models/checkpoints/cancelled.safetensors",
                "size_bytes": 4096,
                "sha256": "b" * 64,
                "auth": "none",
                "parallel": True,
            },
        )

    assert signals == ["terminate"]
    assert not (comfy_dir / "models" / "checkpoints" / "cancelled.safetensors").exists()


def shared_payload_server(payload: bytes):
    """A one-file HTTP server, used to prove a download did or did not happen."""

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_no_checksum_flag_when_the_catalog_entry_has_no_sha256(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"unverifiable-payload" * 2048
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    fake_aria2c(monkeypatch, payload, recorded)
    controller = launcher_app.JobController()

    download_one_file(
        controller,
        {
            "name": "Unverifiable model",
            "url": "https://cdn.example/unverifiable.safetensors",
            "destination": "models/checkpoints/unverifiable.safetensors",
            "size_bytes": len(payload),
            "auth": "none",
            "parallel": True,
        },
    )

    # Nothing to verify against, so the argv is exactly what it was before.
    assert not any(part.startswith("--checksum") for part in recorded[0])


def test_an_aria2c_verified_file_is_never_hashed_again(tmp_path, monkeypatch) -> None:
    payload = b"already-verified-by-aria2c" * 2048
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")

    recorded: list = []
    fake_aria2c(monkeypatch, payload, recorded)

    def refuse(*_args, **_kwargs):
        raise AssertionError("aria2c verified this file; it must not be hashed again")

    monkeypatch.setattr(launcher_app, "file_sha256", refuse)
    controller = launcher_app.JobController()

    written = download_one_file(
        controller,
        {
            "name": "Verified model",
            "url": "https://cdn.example/verified.safetensors",
            "destination": "models/checkpoints/verified.safetensors",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "auth": "none",
            "parallel": True,
        },
    )

    assert written == len(payload)
    assert (
        comfy_dir / "models" / "checkpoints" / "verified.safetensors"
    ).read_bytes() == payload


def test_an_aria2c_that_rejects_the_checksum_option_degrades_once(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"older-aria2c-build" * 2048
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", "/usr/bin/aria2c")
    monkeypatch.setattr(launcher_app, "ARIA2C_SUPPORTS_CHECKSUM", True)

    recorded: list = []

    class FakeProcess:
        def __init__(self, target, argv) -> None:
            self.target = target
            self.argv = argv
            self.returncode = 0

        async def communicate(self):
            if any(part.startswith("--checksum") for part in self.argv):
                self.returncode = 1
                return (
                    b"aria2c: unrecognized option '--checksum=sha-256=abc'\n",
                    b"",
                )
            self.target.write_bytes(payload)
            return b"", b""

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    async def fake_exec(*command, **_kwargs):
        argv = tuple(str(part) for part in command)
        recorded.append(argv)
        directory = Path(argv[argv.index("-d") + 1])
        return FakeProcess(directory / argv[argv.index("-o") + 1], argv)

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", fake_exec)
    controller = launcher_app.JobController()

    written = download_one_file(
        controller,
        {
            "name": "Legacy aria2c model",
            "url": "https://cdn.example/legacy.safetensors",
            "destination": "models/checkpoints/legacy.safetensors",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "auth": "none",
            "parallel": True,
        },
    )

    # An option this build does not know must cost one retry, not the whole install.
    assert len(recorded) == 2
    assert any(part.startswith("--checksum") for part in recorded[0])
    assert not any(part.startswith("--checksum") for part in recorded[1])
    assert launcher_app.ARIA2C_SUPPORTS_CHECKSUM is False
    # And the file is still verified - by the Python hash, on the second pass.
    assert written == len(payload)
    assert (
        comfy_dir / "models" / "checkpoints" / "legacy.safetensors"
    ).read_bytes() == payload


def test_the_httpx_path_hashes_inline_and_catches_a_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"streamed-and-hashed" * 4096
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the bytes were hashed inline; no re-read is allowed")

    monkeypatch.setattr(launcher_app, "file_sha256", refuse)

    server, thread = shared_payload_server(payload)
    controller = launcher_app.JobController()
    try:
        written = download_one_file(
            controller,
            {
                "name": "Streamed model",
                "url": f"http://127.0.0.1:{server.server_port}/streamed",
                "destination": "models/checkpoints/streamed.safetensors",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "auth": "none",
            },
        )
        assert written == len(payload)
        assert (
            comfy_dir / "models" / "checkpoints" / "streamed.safetensors"
        ).read_bytes() == payload

        with pytest.raises(RuntimeError, match="Checksum verification failed"):
            download_one_file(
                controller,
                {
                    "name": "Wrong checksum model",
                    "url": f"http://127.0.0.1:{server.server_port}/streamed",
                    "destination": "models/checkpoints/wrong.safetensors",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(b"a different file").hexdigest(),
                    "auth": "none",
                },
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def resumable_server(payload: bytes, honour_range: bool):
    """Serves payload, optionally honouring Range with a 206."""
    seen_ranges: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requested = self.headers.get("Range")
            seen_ranges.append(requested)
            if honour_range and requested:
                start = int(requested.split("=")[1].split("-")[0])
                body = payload[start:]
                self.send_response(206)
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                )
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, seen_ranges


def test_a_resumed_download_hashes_the_whole_file_not_just_the_tail(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"resume-me-completely" * 4096
    comfy_dir = tmp_path / "ComfyUI"
    destination_dir = comfy_dir / "models" / "checkpoints"
    destination_dir.mkdir(parents=True)
    partial = destination_dir / "resumed.safetensors.part"
    partial.write_bytes(payload[: len(payload) // 2])
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the digest must come from the inline hash, not a re-read")

    monkeypatch.setattr(launcher_app, "file_sha256", refuse)

    server, thread, seen = resumable_server(payload, honour_range=True)
    controller = launcher_app.JobController()
    try:
        download_one_file(
            controller,
            {
                "name": "Resumed model",
                "url": f"http://127.0.0.1:{server.server_port}/resumed",
                "destination": "models/checkpoints/resumed.safetensors",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "auth": "none",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    # It really did resume, and the digest still covered the bytes already on disk.
    assert seen == [f"bytes={len(payload) // 2}-"]
    assert (destination_dir / "resumed.safetensors").read_bytes() == payload


def test_a_server_that_ignores_range_is_not_hashed_against_the_stale_partial(
    tmp_path,
    monkeypatch,
) -> None:
    # The trap: the seed can only be decided after the response arrives. A 200 means
    # the write truncates, so folding the old .part into the hash would digest bytes
    # that never reach the finished file.
    payload = b"start-over-please" * 4096
    comfy_dir = tmp_path / "ComfyUI"
    destination_dir = comfy_dir / "models" / "checkpoints"
    destination_dir.mkdir(parents=True)
    partial = destination_dir / "restarted.safetensors.part"
    partial.write_bytes(b"stale bytes from an earlier attempt")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the digest must come from the inline hash, not a re-read")

    monkeypatch.setattr(launcher_app, "file_sha256", refuse)

    server, thread, seen = resumable_server(payload, honour_range=False)
    controller = launcher_app.JobController()
    try:
        download_one_file(
            controller,
            {
                "name": "Restarted model",
                "url": f"http://127.0.0.1:{server.server_port}/restarted",
                "destination": "models/checkpoints/restarted.safetensors",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "auth": "none",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert seen == ["bytes=35-"]
    assert (destination_dir / "restarted.safetensors").read_bytes() == payload


def test_the_httpx_path_never_resumes_an_aria2c_partial(tmp_path, monkeypatch) -> None:
    payload = b"sparse-partial-trap" * 4096
    comfy_dir = tmp_path / "ComfyUI"
    destination_dir = comfy_dir / "models" / "checkpoints"
    destination_dir.mkdir(parents=True)
    partial = destination_dir / "sparse.safetensors.part"
    control = destination_dir / "sparse.safetensors.part.aria2"
    # An aria2c partial: st_size is already the full length, the data is not there.
    with partial.open("wb") as handle:
        handle.seek(len(payload) - 1)
        handle.write(b"\0")
    control.write_bytes(b"aria2 control")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    server, thread, seen = resumable_server(payload, honour_range=True)
    controller = launcher_app.JobController()
    try:
        download_one_file(
            controller,
            {
                "name": "Sparse partial model",
                "url": f"http://127.0.0.1:{server.server_port}/sparse",
                "destination": "models/checkpoints/sparse.safetensors",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "auth": "none",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    # No Range at all: the aria2c partial was discarded rather than resumed from,
    # which would have sent bytes=<full length>- and hashed a file of zeroes.
    assert seen == [None]
    assert not control.exists()
    assert (destination_dir / "sparse.safetensors").read_bytes() == payload


def test_a_file_already_on_disk_elsewhere_is_linked_not_downloaded(
    tmp_path,
    monkeypatch,
) -> None:
    # ComfyUI looks for the Qwen encoder in both text_encoders and clip, so the
    # catalog lists it twice on purpose. Installing both workflows must not pull
    # 8.66 GB twice.
    payload = b"shared-encoder-payload" * 4096
    digest = hashlib.sha256(payload).hexdigest()

    comfy_dir = tmp_path / "ComfyUI"
    twin = comfy_dir / "models" / "text_encoders" / "qwen.safetensors"
    twin.parent.mkdir(parents=True)
    twin.write_bytes(payload)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the file was already on disk; it must not be downloaded")

    monkeypatch.setattr(launcher_app, "tokenized_request", refuse)

    controller = launcher_app.JobController()
    controller.shared_destinations = {
        digest: [
            "models/text_encoders/qwen.safetensors",
            "models/clip/qwen.safetensors",
        ]
    }

    written = download_one_file(
        controller,
        {
            "name": "Qwen 3 8B text encoder",
            "url": "https://cdn.example/qwen.safetensors",
            "destination": "models/clip/qwen.safetensors",
            "size_bytes": len(payload),
            "sha256": digest,
            "auth": "none",
        },
    )

    linked = comfy_dir / "models" / "clip" / "qwen.safetensors"
    assert linked.read_bytes() == payload
    assert twin.read_bytes() == payload
    assert written == len(payload)
    assert not linked.with_name(linked.name + ".part").exists()
    # Progress has to land too, or the bar sits still through the whole file.
    assert controller.state.file_downloaded_bytes == len(payload)
    assert controller.state.downloaded_bytes == len(payload)
    assert controller.state.percent == 99


def test_a_twin_that_is_not_on_disk_downloads_normally(tmp_path, monkeypatch) -> None:
    payload = b"not-yet-anywhere" * 4096
    digest = hashlib.sha256(payload).hexdigest()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    server, thread = shared_payload_server(payload)
    controller = launcher_app.JobController()
    controller.shared_destinations = {
        digest: [
            "models/text_encoders/qwen.safetensors",
            "models/clip/qwen.safetensors",
        ]
    }

    try:
        download_one_file(
            controller,
            {
                "name": "Qwen 3 8B text encoder",
                "url": f"http://127.0.0.1:{server.server_port}/qwen.safetensors",
                "destination": "models/clip/qwen.safetensors",
                "size_bytes": len(payload),
                "sha256": digest,
                "auth": "none",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (comfy_dir / "models" / "clip" / "qwen.safetensors").read_bytes() == payload
    assert not (comfy_dir / "models" / "text_encoders").exists()


def test_a_twin_of_the_wrong_size_is_never_linked(tmp_path, monkeypatch) -> None:
    payload = b"the-real-thing" * 4096
    digest = hashlib.sha256(payload).hexdigest()
    wrong = b"truncated remnant of an earlier download"

    comfy_dir = tmp_path / "ComfyUI"
    twin = comfy_dir / "models" / "text_encoders" / "qwen.safetensors"
    twin.parent.mkdir(parents=True)
    twin.write_bytes(wrong)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "ARIA2C_PATH", None)

    server, thread = shared_payload_server(payload)
    controller = launcher_app.JobController()
    controller.shared_destinations = {
        digest: [
            "models/text_encoders/qwen.safetensors",
            "models/clip/qwen.safetensors",
        ]
    }

    try:
        download_one_file(
            controller,
            {
                "name": "Qwen 3 8B text encoder",
                "url": f"http://127.0.0.1:{server.server_port}/qwen.safetensors",
                "destination": "models/clip/qwen.safetensors",
                "size_bytes": len(payload),
                "sha256": digest,
                "auth": "none",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (comfy_dir / "models" / "clip" / "qwen.safetensors").read_bytes() == payload
    # The wrong-sized file is left exactly as it was, not linked and not clobbered.
    assert twin.read_bytes() == wrong


def test_huggingface_token_is_refused_for_a_foreign_host(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_only")

    url, headers = launcher_app.tokenized_request(
        {
            "name": "Legitimate model",
            "url": "https://huggingface.co/example/model/resolve/main/model.safetensors",
            "auth": "huggingface",
        }
    )
    assert headers["Authorization"] == "Bearer hf_test_only"

    # A catalog served by the API must not be able to name a host of its choosing.
    with pytest.raises(RuntimeError, match="credential refused"):
        launcher_app.tokenized_request(
            {
                "name": "Exfiltration attempt",
                "url": "https://attacker.example/collect.safetensors",
                "auth": "huggingface",
            }
        )


def test_account_reports_no_credential(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.delenv("LCT_API_BASE", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / "absent.lct")

    with TestClient(launcher_app.app) as client:
        account = client.get("/api/account").json()

    assert account == {
        "configured": False,
        "source": "none",
        "status": None,
        "service": "unconfigured",
    }


def test_signing_in_stores_the_token_and_sends_the_real_password(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with account_api(login_body={"token": "abc", "tier": "fast"}) as (base, stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            response = client.post(
                "/api/account/login",
                json={"email": " user@example.com ", "password": "hunter2"},
            )
            assert response.status_code == 200
            signed_in = response.json()
            account = client.get("/api/account").json()

    # SecretStr stringifies to '**********'; without an explicit unwrap this is what
    # the account service would receive, and login could never succeed.
    assert stub.login_bodies == [{"email": "user@example.com", "password": "hunter2"}]
    assert token_file.read_text(encoding="utf-8") == "abc"
    assert signed_in["configured"] is True
    assert signed_in["source"] == "file"
    assert signed_in["status"]["tier"] == "fast"
    assert account["configured"] is True
    assert account["source"] == "file"
    # One login means one round trip: no second /v1/status call to build the response.
    assert stub.paths == ["/v1/auth/login", "/v1/status"]


def test_no_account_route_ever_returns_the_credential(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / ".lct"
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with account_api(login_body={"token": "abc", "tier": "fast"}) as (base, _stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            bodies = [
                client.post(
                    "/api/account/login",
                    json={"email": "user@example.com", "password": "hunter2"},
                ).text,
                client.get("/api/account").text,
                client.post("/api/account/logout").text,
            ]

    for body in bodies:
        assert "abc" not in body
        assert "hunter2" not in body


def test_signing_out_clears_the_token_file(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / ".lct"
    token_file.write_text("abc", encoding="utf-8")
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.delenv("LCT_API_BASE", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with TestClient(launcher_app.app) as client:
        account = client.post("/api/account/logout").json()

    assert not token_file.exists()
    assert account == {
        "configured": False,
        "source": "none",
        "status": None,
        "service": "unconfigured",
    }


def test_a_template_licence_key_cannot_be_signed_in_or_out(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    monkeypatch.setenv("LCT_LICENSE_KEY", "from-the-template")
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with account_api(login_body={"token": "abc"}) as (base, stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            login = client.post(
                "/api/account/login",
                json={"email": "user@example.com", "password": "hunter2"},
            )
            logout = client.post("/api/account/logout")

    assert login.status_code == 409
    assert logout.status_code == 409
    assert "LCT_LICENSE_KEY" in login.json()["detail"]
    assert not token_file.exists()
    assert stub.login_bodies == []


def test_rejected_credentials_leave_the_token_file_untouched(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    token_file.write_text("existing", encoding="utf-8")
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with account_api(login_status=401) as (base, _stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            response = client.post(
                "/api/account/login",
                json={"email": "user@example.com", "password": "wrong"},
            )

    assert response.status_code == 401
    assert response.json()["detail"] == "Email or password not recognised."
    assert token_file.read_text(encoding="utf-8") == "existing"


def test_rate_limited_login_explains_the_wait(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / ".lct")

    with account_api(login_status=429) as (base, _stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            response = client.post(
                "/api/account/login",
                json={"email": "user@example.com", "password": "hunter2"},
            )

    assert response.status_code == 401
    assert response.json()["detail"] == "Too many attempts. Wait a minute."


def test_login_without_an_api_base_is_a_503_and_makes_no_request(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LCT_API_BASE", raising=False)
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / ".lct")

    def explode(*_args, **_kwargs):
        raise AssertionError("login must not reach the network without an API base.")

    monkeypatch.setattr(launcher_remote.httpx, "post", explode)

    assert launcher_remote.login("user@example.com", "hunter2") == {
        "ok": False,
        "error": "No account service configured.",
    }

    with TestClient(launcher_app.app) as client:
        response = client.post(
            "/api/account/login",
            json={"email": "user@example.com", "password": "hunter2"},
        )

    # Misconfiguration, not a rejected credential.
    assert response.status_code == 503


def test_a_malformed_login_body_never_echoes_the_password(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / ".lct")

    with account_api(login_status=401) as (base, stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            bodies = [
                # No email: rejected before any request is made.
                client.post("/api/account/login", json={"password": "hunter2"}),
                # Wrong type: coerced to text rather than raising a 422 that would
                # echo the value back.
                client.post(
                    "/api/account/login",
                    json={"email": "user@example.com", "password": 12345},
                ),
                client.post("/api/account/login", json={}),
            ]

    for response in bodies:
        assert response.status_code == 401
        assert response.json()["detail"] == "Email or password not recognised."
        assert "hunter2" not in response.text
        assert "12345" not in response.text

    # An empty email never reaches the service; the coerced password does, as text.
    assert stub.login_bodies == [{"email": "user@example.com", "password": "12345"}]


def test_a_signed_in_pod_reports_an_unreachable_status_service(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    token_file.write_text("abc", encoding="utf-8")
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)
    monkeypatch.setenv("LCT_API_BASE", f"http://127.0.0.1:{closed_port()}")

    with TestClient(launcher_app.app) as client:
        account = client.get("/api/account").json()

    # The badge relies on this: a null status must never read as a stated tier.
    assert account == {
        "configured": True,
        "source": "file",
        "status": None,
        "service": "unavailable",
    }


def test_a_revoked_credential_is_not_reported_as_an_outage(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / ".lct"
    token_file.write_text("stale-key", encoding="utf-8")
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)

    with account_api(status_status=401) as (base, _stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            account = client.get("/api/account").json()

    # The panel must offer sign-in again rather than tell the user to wait out an
    # outage that is not happening.
    assert account["configured"] is True
    assert account["service"] == "unauthenticated"
    assert account["status"] is None


def test_a_login_that_states_no_tier_still_succeeds(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / ".lct")

    with account_api(login_body={"token": "abc"}) as (base, _stub):
        monkeypatch.setenv("LCT_API_BASE", base)
        with TestClient(launcher_app.app) as client:
            response = client.post(
                "/api/account/login",
                json={"email": "user@example.com", "password": "hunter2"},
            )

    assert response.status_code == 200
    # An empty status is the service saying nothing, not the service saying standard.
    assert response.json()["status"] == {}
    assert response.json()["service"] == "ok"
    assert "abc" not in response.text


def test_the_tier_badge_never_infers_a_tier_from_silence() -> None:
    """Guards the shape of renderAccount, not its behaviour.

    There is no JS harness in this project, so this cannot prove the badge branches
    correctly - only that the pattern which caused the bug has not come back. If it
    fails, read renderAccount and decide whether the code or this test is wrong.
    """
    with TestClient(launcher_app.app) as client:
        js = client.get("/app.js").text

    render = js[
        js.index("function renderAccount") : js.index("async function loadAccount")
    ]

    # The standard label needs an explicit tier match, so it is an allowlist rather
    # than a fallback: an absent or unrecognised tier reaches neither label.
    assert 'tier === "standard"' in render
    assert "Checking subscription" in render


def test_writing_a_credential_drops_the_cached_catalog(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / ".lct"
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: token_file)
    captured: list = []

    with catalog_api(remote_catalog_bytes(), captured=captured) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        assert launcher_remote.fetch_catalog() is not None
        assert launcher_remote.fetch_catalog() is not None  # served from the cache
        assert len(captured) == 1

        launcher_remote.write_credential("a-different-key")
        assert launcher_remote.fetch_catalog() is not None

    assert len(captured) == 2
    assert captured[1].get("Authorization") == "Bearer a-different-key"


def test_writing_a_credential_does_not_reset_the_log_suppressors(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("LCT_LICENSE_KEY", raising=False)
    monkeypatch.setattr(launcher_remote, "_token_file", lambda: tmp_path / ".lct")
    monkeypatch.setenv("LCT_API_BASE", f"http://127.0.0.1:{closed_port()}")

    launcher_remote.fetch_catalog(fresh=True)
    launcher_remote.write_credential("a-key")
    capsys.readouterr()
    launcher_remote.fetch_catalog(fresh=True)

    # Same failure reason, already reported: a sign-in must not un-suppress it.
    assert "Catalog API unavailable" not in capsys.readouterr().out


def test_a_dead_status_endpoint_is_only_called_once_per_ttl(monkeypatch) -> None:
    monkeypatch.setenv("LCT_API_BASE", f"http://127.0.0.1:{closed_port()}")
    attempts: list[str] = []

    real_get = launcher_remote.httpx.get

    def counting_get(url, **kwargs):
        attempts.append(url)
        return real_get(url, **kwargs)

    monkeypatch.setattr(launcher_remote.httpx, "get", counting_get)

    assert launcher_remote.fetch_status() == {"reason": "unavailable", "data": None}
    assert launcher_remote.fetch_status() == {"reason": "unavailable", "data": None}

    assert len(attempts) == 1


STATIC = None  # resolved lazily against launcher_app.SOURCE_ROOT


def static_file(name: str) -> str:
    return (launcher_app.SOURCE_ROOT / "launcher" / "static" / name).read_text(
        encoding="utf-8"
    )


class MarkupIndex(HTMLParser):
    """Collects ids, and the classes of every element inside #job-panel."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.classes_in_job_panel: set[str] = set()
        self._depth_in_panel = 0

    def handle_starttag(self, tag, attrs) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.add(element_id)
        if self._depth_in_panel:
            self.classes_in_job_panel.update((attributes.get("class") or "").split())
            self._depth_in_panel += 1
        elif element_id == "job-panel":
            self._depth_in_panel = 1

    def handle_endtag(self, tag) -> None:
        if self._depth_in_panel:
            self._depth_in_panel -= 1


def test_every_element_app_js_reaches_for_exists_in_the_markup() -> None:
    """The contract between app.js and index.html, enforced rather than remembered.

    app.js finds elements by id and toggles classes on them, so a rename that looks
    cosmetic breaks the panel silently - and pollStatus swallows the exception, so the
    panel just freezes with a clean console. This makes that class of bug impossible.
    """
    js = static_file("app.js")
    markup = MarkupIndex()
    markup.feed(static_file("index.html"))

    # Only the two forms that really are id lookups. A bare "#" scan would
    # false-positive on location.hash and on the `#${selected}` template.
    referenced = set(re.findall(r'querySelector\(\s*["\']#([A-Za-z0-9_-]+)', js))
    referenced |= set(re.findall(r'getElementById\(\s*["\']([A-Za-z0-9_-]+)', js))

    missing = sorted(referenced - markup.ids)
    assert not missing, f"app.js reads ids that index.html does not define: {missing}"
    assert len(referenced) >= 35

    # The one selector that is not an id, and that the check above cannot see:
    #   elements.track = document.querySelector("#job-panel .progress-track")
    # used unguarded on every updatePanel.
    assert 'querySelector("#job-panel .progress-track")' in js
    assert "progress-track" in markup.classes_in_job_panel


def test_every_view_target_has_a_view_and_appears_in_view_names() -> None:
    js = static_file("app.js")
    html = static_file("index.html")

    targets = set(re.findall(r'data-view-target="([^"]+)"', html))
    views = set(re.findall(r'data-view="([^"]+)"', html))
    declared = set(
        re.findall(
            r'"([^"]+)"',
            re.search(r"const VIEW_NAMES\s*=\s*\[([^\]]*)\]", js).group(1),
        )
    )

    # A nav button with no matching view is the dead-tab bug this project has had.
    assert targets == views == declared
    # The RapidCache tab is a label change only; the view id stays "account".
    assert "account" in declared
    assert 'data-view-target="account"' in html
    assert ">\n            RapidCache\n          </button>" in html


UPSELL_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
// argv: [node, this script, app.js path, scenarios json]
const APP_JS = process.argv[2];
const scenarios = JSON.parse(process.argv[3]);

function fakeElement() {
  const node = {
    textContent: "", innerHTML: "", hidden: false, disabled: false,
    href: "", value: "", src: "", muted: false,
    style: {}, dataset: {},
    classList: {
      toggle(name, on) { if (name === "tier-pill") node.pillOn = on; },
      add() {}, remove() {}, contains() { return false; },
    },
    addEventListener(type, handler) { (node.handlers[type] ||= []).push(handler); },
    setAttribute() {}, removeAttribute() {},
    append() {}, remove() {}, focus() {}, scrollIntoView() {},
    play() { node.played = true; return Promise.resolve(); },
    querySelector() { return fakeElement(); },
    querySelectorAll() { return []; },
    handlers: {}, played: false,
  };
  return node;
}

const results = [];
for (const scenario of scenarios) {
  const nodes = {};
  const timers = { started: [], cleared: [] };
  const element = (key) => (nodes[key] ||= fakeElement());

  // The help page's screenshots. Built before app.js runs, because app.js walks them at
  // load time and a figure hidden then is the whole point of the assertion.
  const docsFigure = fakeElement();
  const docsImage = fakeElement();
  docsImage.parentElement = docsFigure;
  Object.assign(docsImage, scenario.docsImage || {});

  const document = {
    // app.js spreads the result and reads .dataset on each entry.
    querySelectorAll: (sel) => {
      if (sel.includes("docs-figure")) return [docsImage];
      const one = fakeElement();
      one.dataset.view = "workflows";
      one.dataset.viewTarget = "workflows";
      return [one];
    },
    querySelector: (sel) => element(sel),
    getElementById: (id) => element("#" + id),
    createElement: () => {
      // escapeText() sets textContent and reads innerHTML back; that is how app.js
      // escapes. Mirror it, minus the escaping, or every metric string reads as empty
      // here and an assertion about panel text would pass on a blank panel.
      const node = fakeElement();
      let text = "";
      Object.defineProperty(node, "textContent", {
        get: () => text,
        set: (value) => { text = String(value); node.innerHTML = text; },
      });
      return node;
    },
    addEventListener() {},
  };

  const sandbox = {
    document,
    console,
    Promise,
    setTimeout,
    clearTimeout,
    fetch: () => Promise.reject(new Error("offline in tests")),
    // Recorded, never scheduled: a live interval keeps node alive and would hang
    // the pytest wrapper instead of failing it.
    setInterval: (fn, ms) => { const id = timers.started.length + 1; timers.started.push(id); return id; },
    clearInterval: (id) => { timers.cleared.push(id); },
  };
  sandbox.window = {
    location: { hash: scenario.hash || "", protocol: "https:", hostname: "pod.test" },
    history: { replaceState() {} },
    matchMedia: (q) => ({ matches: Boolean(scenario.reduceMotion) }),
    setInterval: sandbox.setInterval,
    clearInterval: sandbox.clearInterval,
  };
  sandbox.globalThis = sandbox;

  const context = vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(APP_JS, "utf8"), context);

  const video = nodes["#rapidcache-video"];
  if (scenario.videoError && video && video.handlers.error) {
    video.handlers.error.forEach((h) => h());
  }
  if (scenario.docsImageError && docsImage.handlers.error) {
    docsImage.handlers.error.forEach((h) => h());
  }
  if (scenario.account) {
    context.renderAccount(scenario.account);
  }
  if (scenario.status) {
    context.updatePanel(scenario.status);
  }

  // classList is a no-op in this fake DOM, so the pill is observed through the toggle
  // rather than through a class list nothing maintains.
  const badge = nodes["#tier-badge"];
  results.push({
    name: scenario.name,
    tierText: badge ? badge.textContent : null,
    tierPill: badge ? Boolean(badge.pillOn) : null,
    debugHidden: nodes["#debug-button"] ? nodes["#debug-button"].hidden : null,
    upsellHidden: nodes["#rapidcache-upsell"].hidden,
    hintHidden: nodes["#rapidcache-signin-hint"].hidden,
    videoHidden: video ? video.hidden : null,
    videoSrc: video ? video.src : null,
    metrics: nodes["#job-metrics"] ? nodes["#job-metrics"].innerHTML : null,
    docsFigureHidden: docsFigure.hidden,
    liveTimers: timers.started.filter((id) => !timers.cleared.includes(id)).length,
  });
}
console.log(JSON.stringify(results));
"""


def run_upsell_harness(scenarios: list, app_js: Path) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available; the upsell harness needs it")
    # Its own temp directory, never next to app.js: launcher/static is served at "/" and
    # ships to every pod, so a kill -9 between write and unlink would leave the harness
    # inside the bootstrap zip. app.js is passed as an absolute path, so where the
    # harness lives does not matter to it.
    with tempfile.TemporaryDirectory() as workspace:
        harness = Path(workspace) / "_upsell_harness.cjs"
        harness.write_text(UPSELL_HARNESS, encoding="utf-8")
        finished = subprocess.run(
            [node, str(harness), str(app_js), json.dumps(scenarios)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert finished.returncode == 0, finished.stderr
    return {row["name"]: row for row in json.loads(finished.stdout)}


def account(configured: bool, service: str, tier: str, source: str = "file") -> dict:
    return {
        "configured": configured,
        "source": source,
        "service": service,
        "status": {"tier": tier} if tier else {},
    }


def test_the_upsell_never_reaches_someone_who_cannot_or_need_not_buy() -> None:
    """The eight states of the account panel, one row each.

    Getting this wrong in either direction is costly: advertise to a subscriber and we
    insult a paying customer, hide from a signed-out pod and the promo reaches nobody.
    """
    app_js = launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js"
    rows = run_upsell_harness(
        [
            # A 1.x pod: no LCT_API_BASE, so there is no way to sign in here at all.
            {"name": "no_account_service_no_upsell",
             "account": account(False, "unconfigured", "")},
            # A 2.0 pod with the base baked in and nobody signed in - the audience.
            {"name": "signed_out_2_0_image",
             "account": account(False, "unauthenticated", "")},
            {"name": "signed_in_standard",
             "account": account(True, "ok", "standard")},
            {"name": "credential_rejected",
             "account": account(True, "unauthenticated", "")},
            {"name": "template_key_standard",
             "account": account(True, "ok", "standard", source="env")},
            {"name": "signed_in_paying",
             "account": account(True, "ok", "fast")},
            {"name": "status_check_failed",
             "account": account(True, "unavailable", "")},
            {"name": "template_key_paying",
             "account": account(True, "ok", "fast", source="env")},
        ],
        app_js,
    )

    assert rows["no_account_service_no_upsell"]["upsellHidden"] is True
    assert rows["signed_out_2_0_image"]["upsellHidden"] is False
    assert rows["signed_in_standard"]["upsellHidden"] is False
    assert rows["credential_rejected"]["upsellHidden"] is False
    assert rows["template_key_standard"]["upsellHidden"] is False
    assert rows["signed_in_paying"]["upsellHidden"] is True
    assert rows["status_check_failed"]["upsellHidden"] is True
    assert rows["template_key_paying"]["upsellHidden"] is True

    # A template-key pod hides the sign-in form, so the closing line must not point at it.
    assert rows["template_key_standard"]["hintHidden"] is True
    assert rows["signed_in_standard"]["hintHidden"] is False

    # A live interval would keep node running and hang the wrapper rather than fail it.
    for row in rows.values():
        assert row["liveTimers"] == 0


def test_a_failed_video_hides_only_the_video(tmp_path) -> None:
    """The error path, run against a build that has a clip configured.

    This is the test that patches now. The card ships with RAPIDCACHE_DEMO_URL empty, so
    the 404 path has no way to run against the file as-is - it puts the clip back to prove
    that a decode failure hides the video and leaves the promo standing. The two video
    tests swap roles whenever that constant does; the other one runs the shipped file.
    """
    source = launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js"
    patched = tmp_path / "app.js"
    patched.write_text(
        source.read_text(encoding="utf-8").replace(
            'const RAPIDCACHE_DEMO_URL = "";',
            'const RAPIDCACHE_DEMO_URL = "/rapidcache-demo.mp4";',
        ),
        encoding="utf-8",
    )
    # Guards the postcondition rather than the substitution: however it got there, the
    # file under test must have a clip configured or the scenarios below prove nothing.
    assert 'RAPIDCACHE_DEMO_URL = "/rapidcache-demo.mp4"' in patched.read_text(
        encoding="utf-8"
    )

    rows = run_upsell_harness(
        [
            {"name": "video_ok",
             "account": account(True, "ok", "standard")},
            {"name": "video_404",
             "videoError": True,
             "account": account(True, "ok", "standard")},
        ],
        patched,
    )

    assert rows["video_ok"]["videoHidden"] is False
    assert rows["video_ok"]["videoSrc"] == "/rapidcache-demo.mp4"
    # A 404 or decode failure hides the video and leaves the promo standing.
    assert rows["video_404"]["videoHidden"] is True
    assert rows["video_404"]["upsellHidden"] is False


def test_an_empty_demo_url_hides_the_video_and_keeps_the_promo() -> None:
    """Production behaviour, run against app.js exactly as it ships.

    This used to patch the constant to "" to simulate the case. The card now ships that
    way, so the real file is the case - and patching would be a silent no-op, because the
    needle it used to look for no longer exists. Asserting on the shipped file is the
    stronger test: it fails if anyone re-enables the clip without revisiting these two.
    """
    app_js = launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js"

    rows = run_upsell_harness(
        [{"name": "no_url", "account": account(True, "ok", "standard")}], app_js
    )

    assert rows["no_url"]["videoHidden"] is True
    assert rows["no_url"]["videoSrc"] == ""
    assert rows["no_url"]["upsellHidden"] is False


def test_custom_node_ref_must_be_a_pinned_commit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    commands: list = []

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        commands.append(tuple(str(part) for part in command))
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    with pytest.raises(RuntimeError, match="40-character commit sha"):
        asyncio.run(
            controller._install_custom_node(
                {
                    "name": "Evil-Node",
                    "repo": "https://github.com/example/Evil-Node",
                    "ref": "--upload-pack=touch /tmp/pwned",
                }
            )
        )

    assert commands == []


def spawned_processes(monkeypatch) -> list:
    """Hand back every subprocess a controller starts, so a test can assert it died."""
    spawned: list = []
    real_exec = asyncio.create_subprocess_exec

    async def recording_exec(*command, **kwargs):
        process = await real_exec(*command, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(launcher_app.asyncio, "create_subprocess_exec", recording_exec)
    return spawned


def recording_run_process(recorded: list, on_command=None):
    """A _run_process stand-in that keeps the argv *and* the bound it was given."""

    async def fake(*command, **bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        recorded.append((normalized, bounds))
        if on_command is not None:
            answer = on_command(normalized)
            if answer is not None:
                return answer
        return 0, ""

    return fake


def test_a_hung_installer_subprocess_is_stopped_and_named(monkeypatch) -> None:
    """A customer's pod sat on one line for 46 minutes because nothing could time out.

    The child here never exits on its own, which is the shape of that hang exactly: the
    inner pip was blocked on the network with an empty build overlay, and 46 minutes of
    nothing looked identical to a working install.
    """
    spawned = spawned_processes(monkeypatch)
    controller = launcher_app.JobController()

    async def runner() -> None:
        with pytest.raises(RuntimeError, match="python"):
            await controller._run_process(
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                timeout=0.1,
            )

    asyncio.run(runner())

    assert len(spawned) == 1
    # Reaped, not merely abandoned. communicate() returned, which it cannot do while the
    # child lives, so this is the process itself answering - not a sleep long enough to
    # look convincing on this machine.
    assert spawned[0].returncode is not None


def test_a_hung_custom_node_subprocess_is_stopped_and_named(monkeypatch) -> None:
    """The Custom nodes tab reaches the same git and the same pip, and hung the same way."""
    spawned = spawned_processes(monkeypatch)
    controller = launcher_app.CustomNodeController()

    async def runner() -> None:
        with pytest.raises(RuntimeError, match="python"):
            await controller._run_process(
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                timeout=0.1,
            )

    asyncio.run(runner())

    assert len(spawned) == 1
    assert spawned[0].returncode is not None


def test_an_unbounded_run_process_behaves_exactly_as_before() -> None:
    """The default is None, so no existing caller changed meaning by gaining a keyword."""
    script = "import sys; print('from the child'); sys.exit(3)"

    for controller in (launcher_app.JobController(), launcher_app.CustomNodeController()):
        returncode, output = asyncio.run(
            controller._run_process(sys.executable, "-c", script)
        )
        assert returncode == 3
        assert "from the child" in output


def test_a_timed_out_process_still_reports_what_it_printed() -> None:
    """Raising instead of returning an exit code is what keeps the message readable.

    Callers build their failure text from the output tail. A synthetic non-zero return
    would hand them an empty one and print a mystery, for the single failure mode that
    most needs explaining - so the tail comes back on the exception instead.
    """
    controller = launcher_app.JobController()
    script = (
        "import sys, time; print('Collecting torch'); sys.stdout.flush(); time.sleep(30)"
    )

    async def runner() -> None:
        with pytest.raises(RuntimeError) as failure:
            await controller._run_process(sys.executable, "-c", script, timeout=0.5)
        message = str(failure.value)
        assert "Collecting torch" in message
        assert "did not finish within" in message

    asyncio.run(runner())


def test_every_subprocess_the_workflow_installer_starts_is_bounded(
    tmp_path,
    monkeypatch,
) -> None:
    """Not one unbounded await left on the workflow path - argv by argv."""
    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes = comfy_dir / "custom_nodes"
    comfy_dir.mkdir()
    (comfy_dir / ".git").mkdir()
    (comfy_dir / "requirements.txt").write_text("# none\n", encoding="utf-8")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")

    controller = launcher_app.JobController()
    recorded: list = []
    ref = "1289b52fbb6d64a339a4047b9ea74cf7758ccf1e"

    def answer(command):
        if "remote" in command and "get-url" in command:
            return 0, "https://github.com/kijai/ComfyUI-KJNodes\n"
        if "cat-file" in command:
            return 1, "missing"
        if "clone" in command:
            # Give the pip step something to install, so its bound is recorded too.
            destination = Path(command[-1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "requirements.txt").write_text("# none\n", encoding="utf-8")
        return None

    monkeypatch.setattr(
        controller, "_run_process", recording_run_process(recorded, answer)
    )

    custom_nodes.mkdir(parents=True, exist_ok=True)
    asyncio.run(controller._update_comfyui())
    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI-KJNodes",
                "repo": "https://github.com/kijai/ComfyUI-KJNodes",
                "ref": ref,
                "install_requirements": True,
            }
        )
    )

    assert len(recorded) >= 8
    unbounded = [command for command, bounds in recorded if not bounds.get("timeout")]
    assert unbounded == [], f"unbounded subprocess: {unbounded}"

    # The bounds themselves, so a careless edit cannot quietly let git wait half an hour.
    for command, bounds in recorded:
        timeout = bounds["timeout"]
        if "pip" in command:
            assert timeout == 1800
        elif "ls-remote" in command:
            # Shorter than the rest: GitHub answers in under two seconds, and a probe that
            # times out costs only the shortcut.
            assert timeout == 30
        elif (
            "cat-file" in command
            or "rev-parse" in command
            or ("remote" in command and "get-url" in command)
        ):
            assert timeout == 60
        else:
            assert timeout == 600


def test_every_subprocess_the_custom_nodes_tab_starts_is_bounded(
    tmp_path,
    monkeypatch,
) -> None:
    """The second copy of the same bug, bounded by the second copy of the same helper."""
    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes = comfy_dir / "custom_nodes"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(launcher_app, "validate_custom_node_url", lambda url: url)

    controller = launcher_app.CustomNodeController()
    recorded: list = []

    def answer(command):
        if "clone" in command:
            staging = Path(command[-1])
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "requirements.txt").write_text("# none\n", encoding="utf-8")
        return None

    monkeypatch.setattr(
        controller, "_run_process", recording_run_process(recorded, answer)
    )

    async def install() -> launcher_app.CustomNodeState:
        created = await controller.enqueue("https://github.com/example/Example-Node")
        if controller.worker_task:
            await controller.worker_task
        return controller.items[created["id"]]

    item = asyncio.run(install())

    assert item.status == "complete", item.error
    commands = [command for command, _bounds in recorded]
    assert any("clone" in command for command in commands)
    assert any("pip" in command for command in commands)
    assert [command for command, bounds in recorded if not bounds.get("timeout")] == []

    # And the branch that only runs when the folder is already there.
    recorded.clear()
    asyncio.run(controller._origin_url(custom_nodes / "Example-Node"))
    assert recorded and recorded[0][1] == {"timeout": 60}


def test_a_missing_build_backend_is_worth_a_retry() -> None:
    """The only thing that earns a second, expensive attempt: no backend to build with."""
    assert launcher_app.needs_build_isolation(
        "ModuleNotFoundError: No module named 'setuptools'"
    )
    assert launcher_app.needs_build_isolation("No module named 'cmake'")
    assert launcher_app.needs_build_isolation(
        "CMake must be installed to build the following extensions: dlib"
    )
    assert launcher_app.needs_build_isolation(
        "  Traceback (most recent call last):\n"
        "    File \"/tmp/pip-build-env/overlay/setup.py\", line 3, in <module>\n"
        "      import setuptools\n"
        "  ModuleNotFoundError: No module named 'setuptools'\n"
        "  [end of output]\n"
    )


def test_a_slow_index_or_a_real_error_never_earns_a_retry() -> None:
    """The retry costs the multi-gigabyte download --no-build-isolation exists to avoid.

    So a false positive here recreates the 46-minute hang. Anything ambiguous is False.
    """
    # Network: a retry doubles the wait on a pod already measured at 87-142 KB/s.
    assert not launcher_app.needs_build_isolation(
        "pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool"
        "(host='pypi.org', port=443): Read timed out."
    )
    assert not launcher_app.needs_build_isolation(
        "ConnectionResetError(104, 'Connection reset by peer')"
    )
    assert not launcher_app.needs_build_isolation(
        "Failed to establish a new connection: [Errno -3] "
        "Temporary failure in name resolution"
    )
    # A genuine build failure: the backend was there, the compile was not.
    assert not launcher_app.needs_build_isolation(
        "error: command '/usr/bin/g++' failed with exit code 1"
    )
    # A resolver conflict, which a second attempt cannot fix either.
    assert not launcher_app.needs_build_isolation(
        "ERROR: Cannot install torch==2.5.1 and torchvision==0.20 because these "
        "package versions have conflicting dependencies."
    )


def test_a_network_failure_wins_even_when_it_mentions_a_module() -> None:
    """Rule 1 before rule 2, because the expensive mistake is the retry."""
    assert not launcher_app.needs_build_isolation(
        "WARNING: Retrying after connection broken by ReadTimeoutError; "
        "ModuleNotFoundError: No module named 'setuptools'"
    )


def custom_node_pip_argv(tmp_path, monkeypatch, outputs):
    """Drive _install_custom_node's dependency step and return every pip argv it ran.

    `outputs` is one (returncode, output) per pip attempt, so a test can make the first
    one fail the way a real pod failed. No pip is launched.
    """
    custom_nodes = tmp_path / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv-cu128")
    controller = launcher_app.JobController()
    pip_runs: list[tuple[str, ...]] = []
    answers = list(outputs)

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if "pip" in normalized:
            pip_runs.append(normalized)
            return answers.pop(0)
        if "remote" in normalized and "get-url" in normalized:
            return 0, "https://github.com/ltdrdata/ComfyUI-Impact-Pack\n"
        if "clone" in normalized:
            destination = Path(normalized[-1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "requirements.txt").write_text(
                "git+https://github.com/facebookresearch/sam2\n", encoding="utf-8"
            )
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    node = {
        "name": "ComfyUI-Impact-Pack",
        "repo": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ref": "a" * 40,
        "install_requirements": True,
    }
    failure = None
    try:
        asyncio.run(controller._install_custom_node(node))
    except RuntimeError as exc:
        failure = exc
    return pip_runs, failure


def test_the_custom_node_pip_builds_against_what_is_already_installed(
    tmp_path,
    monkeypatch,
) -> None:
    """The argv is the fix. Asserted here rather than by launching a real pip."""
    pip_runs, failure = custom_node_pip_argv(tmp_path, monkeypatch, [(0, "")])

    assert failure is None
    assert len(pip_runs) == 1
    argv = pip_runs[0]
    assert "--no-build-isolation" in argv
    assert argv[argv.index("--timeout") + 1] == "15"
    assert argv[argv.index("--retries") + 1] == "3"
    # Still installing from the file, not from a name list.
    assert argv[-2] == "-r"


def test_a_missing_backend_retries_once_with_isolation_restored(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """The fallback that must exist, and must fire exactly once."""
    pip_runs, failure = custom_node_pip_argv(
        tmp_path,
        monkeypatch,
        [(1, "ModuleNotFoundError: No module named 'setuptools'"), (0, "")],
    )

    assert failure is None
    assert len(pip_runs) == 2
    assert "--no-build-isolation" in pip_runs[0]
    assert "--no-build-isolation" not in pip_runs[1]
    # The network flags stay on the retry - it is the slow attempt, not the fast one.
    assert "--timeout" in pip_runs[1] and "--retries" in pip_runs[1]
    # Never silent: this attempt is the expensive one.
    assert "build backend missing" in capsys.readouterr().out


def test_a_network_failure_does_not_retry_the_expensive_way(
    tmp_path,
    monkeypatch,
) -> None:
    """Retrying a slow index with isolation restored is how the 46 minutes happened."""
    pip_runs, failure = custom_node_pip_argv(
        tmp_path,
        monkeypatch,
        [(1, "HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.")],
    )

    assert len(pip_runs) == 1
    assert failure is not None and "Read timed out" in str(failure)


def test_the_custom_nodes_tab_pip_carries_the_same_flags(
    tmp_path,
    monkeypatch,
) -> None:
    """Same node packs, same pip, same fix - the tab must not be the slow way in."""
    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes = comfy_dir / "custom_nodes"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(launcher_app, "validate_custom_node_url", lambda url: url)

    controller = launcher_app.CustomNodeController()
    pip_runs: list[tuple[str, ...]] = []

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if "pip" in normalized:
            pip_runs.append(normalized)
            return 0, ""
        if "clone" in normalized:
            staging = Path(normalized[-1])
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "requirements.txt").write_text("pyyaml\n", encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    async def install():
        created = await controller.enqueue("https://github.com/example/Example-Node")
        if controller.worker_task:
            await controller.worker_task
        return controller.items[created["id"]]

    item = asyncio.run(install())

    assert item.status == "complete", item.error
    assert len(pip_runs) == 1
    assert "--no-build-isolation" in pip_runs[0]
    assert pip_runs[0][pip_runs[0].index("--timeout") + 1] == "15"
    assert pip_runs[0][pip_runs[0].index("--retries") + 1] == "3"


def drive_custom_nodes(tmp_path, monkeypatch, install, nodes=None):
    """Run _install_custom_nodes with a stubbed per-node install, recording the panel.

    Returns (controller, messages) - every message the panel was given, in order.
    """
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    messages: list[str] = []
    original_update = controller.update

    def recording_update(**changes) -> None:
        if "message" in changes:
            messages.append(str(changes["message"]))
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)
    monkeypatch.setattr(controller, "_install_custom_node", install)

    asyncio.run(
        controller._install_custom_nodes(
            nodes
            if nodes is not None
            else [
                {"name": "ComfyUI-KJNodes", "repo": "https://github.com/a/b", "ref": "a" * 40},
                {"name": "ComfyUI-Impact-Pack", "repo": "https://github.com/c/d", "ref": "b" * 40},
            ]
        )
    )
    return controller, messages


def test_a_node_install_does_not_inherit_the_last_download_byte_counter(
    tmp_path,
    monkeypatch,
) -> None:
    """The customer's screenshot read "8.0 MB / 357.7 MB" while installing a node.

    update() is plain setattr, so both fields kept whatever the previous model download
    left in them. There is no such file: a node install moves no file bytes.
    """

    async def install(node, *, on_step=None) -> None:
        return None

    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    # Exactly the state a finished model download leaves behind.
    controller.state.file_downloaded_bytes = 375_083_008
    controller.state.file_total_bytes = 375_083_008
    monkeypatch.setattr(controller, "_install_custom_node", install)

    asyncio.run(
        controller._install_custom_nodes(
            [{"name": "ComfyUI-KJNodes", "repo": "https://github.com/a/b", "ref": "a" * 40}]
        )
    )

    assert controller.state.file_downloaded_bytes == 0
    assert controller.state.file_total_bytes == 0


def test_the_node_panel_reports_the_step_and_the_elapsed_time(
    tmp_path,
    monkeypatch,
) -> None:
    """A frozen string for 46 minutes reads as dead, however true it is."""

    async def install(node, *, on_step=None) -> None:
        on_step("cloning")
        on_step("installing dependencies")

    _controller, messages = drive_custom_nodes(tmp_path, monkeypatch, install)

    assert any("(node 1 of 2)" in message for message in messages)
    assert any("(node 2 of 2)" in message for message in messages)
    assert any("cloning" in message for message in messages)
    assert any("installing dependencies" in message for message in messages)
    # The format the panel actually shows, elapsed stamp included.
    assert re.search(
        r"Installing ComfyUI-Impact-Pack \(node 2 of 2\) — installing dependencies, \d+s",
        "\n".join(messages),
    )


def test_no_ticker_outlives_the_node_it_was_reporting(tmp_path, monkeypatch) -> None:
    """A surviving ticker overwrites whatever message is written next."""
    before = None

    async def install(node, *, on_step=None) -> None:
        on_step("cloning")
        # Long enough for at least one tick to fire while this node is "running".
        await asyncio.sleep(1.2)

    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    monkeypatch.setattr(controller, "_install_custom_node", install)

    async def runner() -> int:
        nonlocal before
        before = len(asyncio.all_tasks())
        await controller._install_custom_nodes(
            [{"name": "ComfyUI-KJNodes", "repo": "https://github.com/a/b", "ref": "a" * 40}]
        )
        return len(asyncio.all_tasks())

    after = asyncio.run(runner())

    # Counted, not slept on: nothing is left running to overwrite the next message.
    assert after == before
    assert controller.state.message == "Finishing workflow setup…"
    # And a tick really did fire, so the assertion above is not vacuous.
    assert controller.state.percent == 99


def test_a_failed_node_keeps_its_skip_message(tmp_path, monkeypatch) -> None:
    """The ticker must be gone before the skip message is written, not after."""

    async def install(node, *, on_step=None) -> None:
        on_step("installing dependencies")
        await asyncio.sleep(1.2)
        raise RuntimeError("pip did not finish within 1800s")

    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    monkeypatch.setattr(controller, "_install_custom_node", install)

    async def runner() -> int:
        await controller._install_custom_nodes(
            [
                {"name": "ComfyUI-Impact-Pack", "repo": "https://github.com/c/d", "ref": "b" * 40},
            ]
        )
        return len(asyncio.all_tasks())

    remaining = asyncio.run(runner())

    assert remaining == 1  # the runner itself
    assert controller.state.warnings == [
        "ComfyUI-Impact-Pack: pip did not finish within 1800s"
    ]
    assert controller.state.message == "Finishing workflow setup…"


def test_each_node_prints_where_its_time_went(tmp_path, monkeypatch, capsys) -> None:
    """The measurement that decides whether the per-node pip runs should be batched.

    Printed rather than inferred: every performance question about this launcher so far
    that was answered by guessing was answered wrongly.
    """
    custom_nodes = tmp_path / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv-cu128")
    controller = launcher_app.JobController()

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if "clone" in normalized:
            destination = Path(normalized[-1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "requirements.txt").write_text("pyyaml\n", encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI-KJNodes",
                "repo": "https://github.com/kijai/ComfyUI-KJNodes",
                "ref": "a" * 40,
                "install_requirements": True,
            }
        )
    )

    printed = capsys.readouterr().out
    assert "ComfyUI-KJNodes: cloned in " in printed
    assert "dependencies in " in printed
    assert "total " in printed


def test_a_node_that_was_already_cloned_reports_no_clone_phase(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Omit a phase that did not run, rather than printing 0.0s and inviting a theory."""
    custom_nodes = tmp_path / "custom_nodes"
    destination = custom_nodes / "ComfyUI-KJNodes"
    destination.mkdir(parents=True)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv-cu128")
    controller = launcher_app.JobController()

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if "remote" in normalized and "get-url" in normalized:
            return 0, "https://github.com/kijai/ComfyUI-KJNodes\n"
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI-KJNodes",
                "repo": "https://github.com/kijai/ComfyUI-KJNodes",
                "ref": "a" * 40,
                "install_requirements": True,
            }
        )
    )

    printed = capsys.readouterr().out
    assert "ComfyUI-KJNodes: total " in printed
    assert "cloned in" not in printed


def test_the_rate_sampler_thins_a_transfer_into_a_shape() -> None:
    """Once per poll tick would be 7200 samples an hour, which nobody will read."""
    sampler = launcher_app.RateSampler(interval=10.0)
    for second in range(11):
        sampler.add(float(second), 1_000_000.0 * second)

    exported = sampler.export()
    assert len(exported) == 2
    assert exported[0][0] == 0.0
    assert exported[1][0] == 10.0


def test_the_rate_sampler_keeps_the_beginning_of_a_run_not_the_end() -> None:
    """The collapse being chased starts near 1 GiB/s. Dropping from the front would
    discard exactly the evidence this exists to collect, which is what a
    deque(maxlen=...) would have done."""
    sampler = launcher_app.RateSampler(interval=0.0, max_samples=3)
    for second in range(500):
        sampler.add(float(second), float(second))

    assert sampler.export() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_aria2_report_lines_keeps_the_summaries_and_the_complaints() -> None:
    output = "\n".join(
        [
            "02/17 09:14:01 [NOTICE] Downloading 1 item(s)",
            "[#a1b2c3 12GiB/28GiB(42%) CN:16 DL:1.0GiB ETA:16s]",
            "02/17 09:14:31 [WARN] CUID#7 - Download aborted. URI=https://example",
            "[#a1b2c3 20GiB/28GiB(71%) CN:16 DL:11MiB ETA:12m]",
            "02/17 09:20:00 [ERROR] CUID#9 - Restarting download.",
            "02/17 09:20:01 [NOTICE] Download complete",
        ]
    )

    lines = launcher_app.aria2_report_lines(output)

    assert len(lines) == 4
    assert any("DL:1.0GiB" in line for line in lines)
    assert any("DL:11MiB" in line for line in lines)
    assert any("WARN" in line for line in lines)
    assert any("ERROR" in line for line in lines)
    assert not any("Downloading 1 item(s)" in line for line in lines)


def test_aria2_report_lines_never_carries_a_presigned_signature() -> None:
    """This output is meant to be pasted into a chat window."""
    output = (
        "02/17 09:14:31 [WARN] CUID#7 - Download aborted. URI="
        "https://pub-9c2f.r2.cloudflarestorage.com/models/flux1-dev.safetensors"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE"
        "&X-Amz-Signature=1f3a9c77b2e4d6180ab55c9e2f7d3b41&X-Amz-Expires=3600"
    )

    line = launcher_app.aria2_report_lines(output)[0]

    assert "X-Amz-Signature" not in line
    assert "AKIAEXAMPLE" not in line
    # The hostname is the point of keeping the line at all.
    assert "pub-9c2f.r2.cloudflarestorage.com" in line
    assert "[redacted]" in line


def test_diagnostics_caps_its_history_and_reports_newest_first() -> None:
    store = launcher_app.Diagnostics()
    for number in range(60):
        record = store.begin_file(
            name=f"file-{number}",
            url="https://cdn.example/model.safetensors",
            size_bytes=1024,
            transport="aria2c",
            staging="container-disk",
        )
        store.finish_file(record)

    exported = store.export()
    assert len(exported["files"]) == 50
    assert exported["files"][0]["name"] == "file-59"
    assert exported["files"][-1]["name"] == "file-10"


def test_diagnostics_never_exports_a_url_or_a_path() -> None:
    """Structural, not one hand-picked field: walk everything and look."""
    store = launcher_app.Diagnostics()
    record = store.begin_file(
        name="flux1-dev.safetensors",
        url=(
            "https://pub-9c2f.r2.cloudflarestorage.com/models/flux1-dev.safetensors"
            "?X-Amz-Signature=1f3a9c77b2e4d6180ab55c9e2f7d3b41"
        ),
        size_bytes=23_802_932_552,
        transport="aria2c",
        staging="container-disk",
    )
    store.note_aria2_lines(
        launcher_app.aria2_report_lines(
            "[#a1b2 1GiB/23GiB(4%) CN:16 DL:11MiB] "
            "[WARN] URI=https://pub-9c2f.r2.cloudflarestorage.com/x?X-Amz-Signature=abc"
        )
    )
    store.finish_file(record)
    store.record_node(name="ComfyUI-Impact-Pack", total_seconds=49.5)
    # The update subtree too, with the shape git actually produces: a path in the middle
    # of a sentence, which the "starts with a slash" half of this walk would never catch.
    store.record_comfyui_update(
        workflow_id="minimax-h3",
        error=(
            "ComfyUI update failed: fatal: not a git repository: "
            "/workspace/runpod-slim/ComfyUI/.git, remote https://github.com/x/y.git"
        ),
    )

    def every_string(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from every_string(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from every_string(item)

    for text in every_string(store.export()):
        assert "http" not in text, f"a URL reached the export: {text!r}"
        assert not text.startswith("/"), f"a path reached the export: {text!r}"
    assert store.export()["files"][0]["host"] == "pub-9c2f.r2.cloudflarestorage.com"


def test_a_file_that_fails_still_lands_a_record(tmp_path, monkeypatch) -> None:
    """The in-flight record is closed by the caller, from the one place a failure lands."""
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    controller = launcher_app.JobController()

    async def explode(*_args, **_kwargs):
        raise RuntimeError("Download failed for flux (HTTP 503).")

    monkeypatch.setattr(controller, "_download_file", explode)
    monkeypatch.setattr(controller, "_wait_for_comfyui", lambda: asyncio.sleep(0))
    # An open record, as _download_file would have left one.
    store.begin_file(
        name="flux1-dev.safetensors",
        url="https://cdn.example/flux1-dev.safetensors",
        size_bytes=100,
        transport="aria2c",
        staging="beside-destination",
    )

    asyncio.run(
        controller._install_workflow(
            {
                "id": "w",
                "files": [
                    {
                        "name": "flux1-dev.safetensors",
                        "url": "https://cdn.example/flux1-dev.safetensors",
                        "destination": "models/checkpoints/flux1-dev.safetensors",
                        "size_bytes": 100,
                    }
                ],
            }
        )
    )

    exported = store.export()
    assert exported["in_flight"] is None
    assert len(exported["files"]) == 1
    assert exported["files"][0]["error"] == "Download failed for flux (HTTP 503)."


def test_a_node_that_retried_with_isolation_says_so(tmp_path, monkeypatch) -> None:
    """Today's failure belongs in the same report as the transfers."""
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    custom_nodes = tmp_path / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv-cu128")
    controller = launcher_app.JobController()
    attempts = {"pip": 0}

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if "pip" in normalized:
            attempts["pip"] += 1
            if attempts["pip"] == 1:
                return 1, "ModuleNotFoundError: No module named 'setuptools'"
            return 0, ""
        if "clone" in normalized:
            destination = Path(normalized[-1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "requirements.txt").write_text("dlib\n", encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI_FaceAnalysis",
                "repo": "https://github.com/cubiq/ComfyUI_FaceAnalysis",
                "ref": "a" * 40,
                "install_requirements": True,
            }
        )
    )

    node = store.export()["nodes"][0]
    assert node["name"] == "ComfyUI_FaceAnalysis"
    assert node["retried_with_isolation"] is True
    assert node["error"] is None


def test_the_diagnostics_endpoint_answers_with_every_documented_key(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")
    with TestClient(launcher_app.app) as client:
        response = client.get("/api/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "launcher_ref",
        "aria2c_available",
        "aria2c_supports_checksum",
        "staging_available",
        "in_flight",
        "comfyui_update",
        "files",
        "nodes",
    }
    assert isinstance(body["files"], list)
    assert isinstance(body["nodes"], list)


def test_an_in_flight_transfer_exposes_its_samples_before_it_finishes() -> None:
    """Opening this during a stall is the reading nobody has managed to take by hand."""
    store = launcher_app.Diagnostics()
    sampler = launcher_app.RateSampler(interval=0.0)
    store.begin_file(
        name="flux1-dev.safetensors",
        url="https://cdn.example/flux1-dev.safetensors",
        size_bytes=1024,
        transport="aria2c",
        staging="container-disk",
        sampler=sampler,
    )

    sampler.add(0.0, 1_073_741_824.0)
    sampler.add(10.0, 11_600_000.0)

    in_flight = store.export()["in_flight"]
    assert in_flight is not None
    assert in_flight["name"] == "flux1-dev.safetensors"
    # The collapse, live: a gigabyte a second down to single-digit megabytes.
    assert in_flight["rate_samples"] == [[0.0, 1073741824.0], [10.0, 11600000.0]]


def test_the_aria2c_argv_still_carries_every_invariant(tmp_path, monkeypatch) -> None:
    """The flags this change touches, and the ones it must not."""
    payload = b"invariant-payload" * 4096
    result = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload, destination="models/checkpoints/invariant.safetensors"),
        payload,
    )
    argv = result["argv"]

    assert "--summary-interval=30" in argv
    assert "--console-log-level=notice" in argv
    assert "--summary-interval=0" not in argv
    # Untouched, and each one is load-bearing: see the comments beside them.
    assert "--file-allocation=none" in argv
    assert "--continue=true" in argv
    assert "--auto-file-renaming=false" in argv
    assert "--allow-overwrite=true" in argv
    assert "-x16" in argv and "-s16" in argv
    assert argv[argv.index("-k") + 1] == "4M"
    assert "--lowest-speed-limit" not in " ".join(argv)


def test_a_successful_download_keeps_what_aria2c_reported(tmp_path, monkeypatch) -> None:
    """The output used to be read only inside `if process.returncode:`.

    So the exact failure being chased - a download that finishes, slowly - discarded
    everything aria2c said about it.
    """
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    payload = b"reported-payload" * 4096
    chatter = (
        "02/17 09:14:01 [NOTICE] Downloading 1 item(s)\n"
        "[#a1b2c3 1.0GiB/23GiB(4%) CN:16 DL:11MiB ETA:35m]\n"
    ).encode()

    result = drive_aria2_download(
        tmp_path,
        monkeypatch,
        mirrored_spec(payload, destination="models/checkpoints/reported.safetensors"),
        payload,
        output=chatter,
    )

    assert result["error"] is None
    record = store.export()["files"][0]
    assert record["aria2_lines"] == ["[#a1b2c3 1.0GiB/23GiB(4%) CN:16 DL:11MiB ETA:35m]"]
    assert record["transport"] == "aria2c"
    assert record["host"] == "cdn.example"
    assert record["digest"] == "aria2c-inline"
    assert record["bytes_transferred"] == len(payload)


audit_spec = importlib.util.spec_from_file_location(
    "audit_node_requirements",
    Path(__file__).resolve().parent.parent / "scripts" / "audit_node_requirements.py",
)
audit_script = importlib.util.module_from_spec(audit_spec)
# Registered before it is executed: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for a module loaded straight off a path.
sys.modules[audit_spec.name] = audit_script
audit_spec.loader.exec_module(audit_script)


def pypi_fixture(files):
    """Stand in for pypi.org/pypi/<name>/json. No unit test here touches the network."""
    return lambda _name: {"urls": files}


def test_the_audit_flags_a_vcs_requirement() -> None:
    """The line that cost a customer 46 minutes."""
    finding = audit_script.classify_requirement(
        "git+https://github.com/facebookresearch/sam2",
        "ComfyUI-Impact-Pack",
        pypi_fixture([]),
    )

    assert finding is not None
    assert finding.package == "sam2"
    assert finding.node == "ComfyUI-Impact-Pack"
    assert "VCS requirement" in finding.reason


def test_the_audit_flags_a_direct_url_requirement() -> None:
    """PEP 508 spelling of the same problem."""
    finding = audit_script.classify_requirement(
        "sam-2 @ https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip",
        "ComfyUI-Impact-Pack",
        pypi_fixture([]),
    )

    assert finding is not None
    assert "direct URL" in finding.reason


def test_the_audit_flags_a_package_with_no_wheel_at_all() -> None:
    """dlib 20.0.1: one file on PyPI, and it is a tarball."""
    finding = audit_script.classify_requirement(
        "dlib==20.0.1",
        "ComfyUI_FaceAnalysis",
        lambda _name: {
            "releases": {
                "20.0.1": [
                    {"packagetype": "sdist", "filename": "dlib-20.0.1.tar.gz"},
                ]
            }
        },
    )

    assert finding is not None
    assert finding.package == "dlib"
    assert "sdist only" in finding.reason


def test_the_audit_accepts_an_abi3_wheel_built_for_an_older_python() -> None:
    """Pinned against my own mistake, made by hand during the audit this replaces.

    I filtered for the literal string "cp312", concluded opencv-contrib-python had no
    wheel, and was wrong: its wheel is tagged cp37-abi3, and abi3 covers 3.12. Substring
    matching on wheel filenames is exactly the error this script exists to make
    impossible, so the check goes through packaging and against an explicit target.
    """
    finding = audit_script.classify_requirement(
        "opencv-contrib-python",
        "ComfyUI_LayerStyle",
        pypi_fixture(
            [
                {
                    "packagetype": "bdist_wheel",
                    "filename": (
                        "opencv_contrib_python-4.13.0.92-cp37-abi3-"
                        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                    ),
                },
                {
                    "packagetype": "sdist",
                    "filename": "opencv-contrib-python-4.13.0.92.tar.gz",
                },
            ]
        ),
    )

    assert finding is None


def test_the_audit_flags_wheels_that_are_not_for_this_pod() -> None:
    """Having a wheel is not the question. Having one the pod can install is."""
    finding = audit_script.classify_requirement(
        "some-windows-only-thing",
        "ComfyUI-KJNodes",
        pypi_fixture(
            [
                {
                    "packagetype": "bdist_wheel",
                    "filename": "some_windows_only_thing-1.0-cp312-cp312-win_amd64.whl",
                }
            ]
        ),
    )

    assert finding is not None
    assert "none for CPython 3.12" in finding.reason


def test_the_audit_ignores_comments_and_pip_options() -> None:
    for line in (
        "",
        "   ",
        "# torch is already in the image",
        "  # git+https://github.com/facebookresearch/sam2",
        "-r other-requirements.txt",
        "--extra-index-url https://download.pytorch.org/whl/cu128",
    ):
        assert audit_script.classify_requirement(line, "node", pypi_fixture([])) is None


def test_a_listed_source_build_does_not_fail_the_run(tmp_path) -> None:
    """The escape hatch, and the reason it demands a reason."""
    allowlist = tmp_path / "known-source-builds.txt"
    allowlist.write_text(
        "# a comment\n"
        "sam2   # handled at runtime by --no-build-isolation\n"
        "dlib   # built into the image\n",
        encoding="utf-8",
    )

    allowed = audit_script.read_known_source_builds(allowlist)
    findings = [
        audit_script.Finding("sam2", "ComfyUI-Impact-Pack", "VCS requirement"),
        audit_script.Finding("dlib", "ComfyUI_FaceAnalysis", "sdist only"),
        audit_script.Finding("brand-new-thing", "ComfyUI-KJNodes", "sdist only"),
    ]

    remaining = audit_script.unlisted(findings, allowed)

    assert [finding.package for finding in remaining] == ["brand-new-thing"]
    # A name with no reason is not a listing anyone can act on later.
    assert allowed["sam2"].startswith("handled at runtime")


def test_the_shipped_allowlist_says_how_each_source_build_is_paid_for() -> None:
    allowed = audit_script.read_known_source_builds(audit_script.KNOWN_SOURCE_BUILDS)

    assert set(allowed) == {"sam2", "dlib"}
    assert all(reason for reason in allowed.values())


def test_the_audit_walks_each_pinned_pack_once() -> None:
    """The catalog lists the same pack under several workflows; cloning it twice is waste."""
    catalog = json.loads(
        (Path(launcher_app.__file__).resolve().parent.parent / "catalog" / "workflows.json")
        .read_text(encoding="utf-8")
    )
    packs = audit_script.node_packs(catalog)

    listed = [
        node
        for workflow in catalog["workflows"]
        for node in workflow.get("custom_nodes", [])
    ]
    assert len(packs) < len(listed)
    assert len(packs) == len({(node["repo"], node["ref"]) for node in listed})
    # Every pack is pinned to a sha, which is what makes this audit reproducible.
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for _name, _repo, ref in packs)


def test_aria2_report_lines_drops_paths_without_mangling_aria2s_arithmetic() -> None:
    """A WARN can name the file it could not open, and the no-paths rule has no exceptions.

    The slash in "12GiB/28GiB(42%)" must survive, though: redacting that would destroy
    the only part of the line anyone reads.
    """
    lines = launcher_app.aria2_report_lines(
        "\n".join(
            [
                "[#a1b2c3 12GiB/28GiB(42%) CN:16 DL:1.0GiB ETA:16s]",
                "[WARN] Failed to open /root/.10sorlabs-scratch/ab12-flux.safetensors",
                "[ERROR] CUID#9 - /workspace/runpod-slim/ComfyUI/models is not writable",
            ]
        )
    )

    assert lines[0] == "[#a1b2c3 12GiB/28GiB(42%) CN:16 DL:1.0GiB ETA:16s]"
    assert "[WARN] Failed to open [path]" == lines[1]
    assert "10sorlabs-scratch" not in " ".join(lines)
    assert "/workspace" not in " ".join(lines)
    assert "[path]" in lines[2]


def test_nothing_in_a_diagnostics_export_looks_like_a_path_or_a_url() -> None:
    """The guard from the other side: a line that carries both, through the real record."""
    store = launcher_app.Diagnostics()
    record = store.begin_file(
        name="flux1-dev.safetensors",
        url="https://pub-9c2f.r2.cloudflarestorage.com/x?X-Amz-Signature=abc",
        size_bytes=1024,
        transport="aria2c",
        staging="container-disk",
    )
    store.note_aria2_lines(
        launcher_app.aria2_report_lines(
            "[WARN] /workspace/models/flux.part from "
            "https://pub-9c2f.r2.cloudflarestorage.com/y?X-Amz-Signature=def"
        )
    )
    store.finish_file(record)

    flattened = json.dumps(store.export())
    assert "X-Amz-Signature" not in flattened
    assert "://" not in flattened
    assert "/workspace" not in flattened


SHA_INSTALLED = "a" * 40
SHA_UPSTREAM = "b" * 40


def comfyui_update_harness(tmp_path, monkeypatch, heads, remote):
    """Run _update_comfyui against a scripted git.

    `heads` is one (returncode, output) per rev-parse, in order: the probe before the
    fetch, then the one after the reset. `remote` is the single ls-remote answer. No git
    runs. Returns (controller, recorded argv, recorded messages, recorded updates).
    """
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / ".git").mkdir(parents=True)
    (comfy_dir / "requirements.txt").write_text("torch\n", encoding="utf-8")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")

    controller = launcher_app.JobController()
    recorded: list[tuple[str, ...]] = []
    messages: list[str] = []
    updates: list[dict] = []
    remaining = list(heads)

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        recorded.append(normalized)
        if "rev-parse" in normalized:
            return remaining.pop(0) if remaining else (1, "no more heads scripted")
        if "ls-remote" in normalized:
            return remote
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    original_update = controller.update

    def recording_update(**changes) -> None:
        updates.append(changes)
        if "message" in changes:
            messages.append(str(changes["message"]))
        original_update(**changes)

    monkeypatch.setattr(controller, "update", recording_update)
    return controller, recorded, messages, updates


def ran(recorded, word: str) -> bool:
    return any(word in command for command in recorded)


def test_an_up_to_date_comfyui_is_not_fetched_reset_or_reinstalled(
    tmp_path,
    monkeypatch,
) -> None:
    """The common case: the base image ships a current ComfyUI.

    The pip install alone measures about two minutes on a pod, and MiniMax H3 runs this
    before any download - which is why that panel sat at 0% doing nothing visible.
    """
    controller, recorded, messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n")],
        remote=(0, f"{SHA_INSTALLED}\trefs/heads/master\n"),
    )

    asyncio.run(controller._update_comfyui())

    assert len(recorded) == 2
    assert "rev-parse" in recorded[0]
    assert "ls-remote" in recorded[1]
    assert not ran(recorded, "fetch")
    assert not ran(recorded, "reset")
    assert not ran(recorded, "pip")
    assert controller.state.message == "ComfyUI is already up to date."


def test_a_moved_master_still_runs_the_whole_update(tmp_path, monkeypatch) -> None:
    """The shortcut is an optimisation. When it does not apply, nothing else changed."""
    controller, recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_UPSTREAM + "\n")],
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )

    asyncio.run(controller._update_comfyui())

    verbs = []
    for command in recorded:
        for word in ("rev-parse", "ls-remote", "set-url", "fetch", "reset", "pip"):
            if word in command:
                verbs.append(word)
                break
    # The existing order, with the two probes in front and one more rev-parse after the
    # reset - which is the only new subprocess this change adds to a real update.
    assert verbs == [
        "rev-parse",
        "ls-remote",
        "set-url",
        "fetch",
        "reset",
        "rev-parse",
        "pip",
    ]


def test_a_failed_ls_remote_only_costs_the_shortcut(tmp_path, monkeypatch) -> None:
    """A probe that cannot answer must never fail the install."""
    controller, recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_UPSTREAM + "\n")],
        remote=(128, "fatal: unable to access 'https://github.com/...': Could not resolve host"),
    )

    asyncio.run(controller._update_comfyui())

    assert ran(recorded, "fetch")
    assert ran(recorded, "reset")
    assert ran(recorded, "pip")


def test_a_failed_rev_parse_only_costs_the_shortcut(tmp_path, monkeypatch) -> None:
    controller, recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(128, "fatal: ambiguous argument 'HEAD'"), (128, "fatal: again")],
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )

    asyncio.run(controller._update_comfyui())

    assert ran(recorded, "fetch")
    assert ran(recorded, "reset")
    # No pair of real shas to compare, so the requirements install is not skipped either.
    assert ran(recorded, "pip")


def test_a_reset_that_moved_nothing_does_not_reinstall_requirements(
    tmp_path,
    monkeypatch,
) -> None:
    """Same invariant as the shortcut, for when the shortcut could not be taken.

    If HEAD did not move, the requirements did not change.
    """
    controller, recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_INSTALLED + "\n")],
        # A differing remote, so the shortcut is refused and the reset actually runs.
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )

    asyncio.run(controller._update_comfyui())

    assert ran(recorded, "fetch")
    assert ran(recorded, "reset")
    assert not ran(recorded, "pip")
    assert controller.state.message == "ComfyUI was already at the latest version."


def test_a_real_update_installs_requirements_with_the_network_flags(
    tmp_path,
    monkeypatch,
) -> None:
    """And still without --no-build-isolation: ComfyUI's requirements are all wheels."""
    controller, recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_UPSTREAM + "\n")],
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )

    asyncio.run(controller._update_comfyui())

    pip = next(command for command in recorded if "pip" in command)
    assert pip[pip.index("--timeout") + 1] == "15"
    assert pip[pip.index("--retries") + 1] == "3"
    assert "--no-build-isolation" not in pip


def test_no_update_ticker_outlives_the_update(tmp_path, monkeypatch) -> None:
    """A surviving ticker overwrites whatever message is written next."""
    controller, _recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n")],
        remote=(0, f"{SHA_INSTALLED}\trefs/heads/master\n"),
    )

    async def returns_normally() -> int:
        before = len(asyncio.all_tasks())
        await controller._update_comfyui()
        return len(asyncio.all_tasks()) - before

    assert asyncio.run(returns_normally()) == 0
    assert controller.state.message == "ComfyUI is already up to date."

    # And when it raises. A .git that is not there is the method's own first check.
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "gone")
    failing, _recorded, _messages, _updates = comfyui_update_harness(
        tmp_path / "second",
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_UPSTREAM + "\n")],
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )

    async def raises() -> int:
        before = len(asyncio.all_tasks())
        monkeypatch.setattr(
            failing,
            "_run_process",
            _explode_on("fetch", failing._run_process),
        )
        with pytest.raises(RuntimeError, match="ComfyUI update failed"):
            await failing._update_comfyui()
        return len(asyncio.all_tasks()) - before

    assert asyncio.run(raises()) == 0


def _explode_on(word, inner):
    async def fake(*command, **bounds):
        normalized = tuple(str(part) for part in command)
        if word in normalized:
            return 1, "fatal: could not fetch"
        return await inner(*command, **bounds)

    return fake


def test_the_update_phase_zeroes_bytes_and_never_advances_percent(
    tmp_path,
    monkeypatch,
) -> None:
    """Zeroing is defensive here - start() already guarantees it - but percent is not.

    percent stays at 0 for the whole phase on purpose: the download phase that follows
    computes it from its own file counter, starting near zero, so any number claimed here
    would be handed straight back. A bar that goes backwards is a bug this file already
    guards against in the aria2c poller.
    """
    controller, _recorded, _messages, updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n"), (0, SHA_UPSTREAM + "\n")],
        remote=(0, f"{SHA_UPSTREAM}\trefs/heads/master\n"),
    )
    controller.state.file_downloaded_bytes = 375_083_008
    controller.state.file_total_bytes = 375_083_008

    asyncio.run(controller._update_comfyui())

    assert controller.state.file_downloaded_bytes == 0
    assert controller.state.file_total_bytes == 0
    assert controller.state.percent == 0
    assert [change["percent"] for change in updates if "percent" in change] == [0]


def test_the_update_lands_in_diagnostics(tmp_path, monkeypatch) -> None:
    """So the next time this is slow, nobody has to ask for a screenshot."""
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    assert store.export()["comfyui_update"] is None

    controller, _recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n")],
        remote=(0, f"{SHA_INSTALLED}\trefs/heads/master\n"),
    )
    controller.state.workflow_id = "minimax-h3"

    asyncio.run(controller._update_comfyui())

    record = store.export()["comfyui_update"]
    assert record["skipped"] is True
    assert record["reason"] == "already-up-to-date"
    assert record["workflow_id"] == "minimax-h3"
    assert record["requirements_seconds"] == 0.0
    assert record["error"] is None


def test_an_update_is_not_attributed_to_the_next_workflow(tmp_path, monkeypatch) -> None:
    """State left over from a previous install, shown as if it were current, is the same
    bug as the byte counter a node install used to inherit from a finished download."""
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    controller, _recorded, _messages, _updates = comfyui_update_harness(
        tmp_path,
        monkeypatch,
        heads=[(0, SHA_INSTALLED + "\n")],
        remote=(0, f"{SHA_INSTALLED}\trefs/heads/master\n"),
    )

    async def install_then_start_another():
        await controller._update_comfyui()
        assert store.export()["comfyui_update"] is not None

        async def nothing(_workflow) -> None:
            return None

        monkeypatch.setattr(controller, "_run", nothing)
        await controller.start({"id": "no-update", "title": "Workflow with no update"})
        if controller.task:
            await controller.task
        return store.export()["comfyui_update"]

    assert asyncio.run(install_then_start_another()) is None


def test_cancel_stops_a_subprocess_that_is_still_running(monkeypatch) -> None:
    """Cancel was immediate everywhere except the two places it is pressed.

    aria2c is raced and terminated, copy_into_place checks per chunk - but every git
    clone and every pip install ignored the button, which on a real pod is
    ComfyUI_FaceAnalysis at 136s compiling dlib.
    """
    spawned = spawned_processes(monkeypatch)
    controller = launcher_app.JobController()

    async def runner() -> None:
        async def press_cancel() -> None:
            await asyncio.sleep(0.2)
            controller.cancel_event.set()

        pressing = asyncio.create_task(press_cancel())
        with pytest.raises(launcher_app.InstallCancelled):
            # No timeout on purpose: cancel has to work on its own, not as a side effect
            # of a bound expiring.
            await controller._run_process(
                sys.executable, "-c", "import time; time.sleep(30)"
            )
        await pressing

    asyncio.run(runner())

    assert len(spawned) == 1
    # The process itself answering, not a sleep long enough to look convincing here.
    assert spawned[0].returncode is not None


def test_a_cancel_and_a_timeout_stay_distinguishable() -> None:
    """_install_custom_nodes re-raises one and turns the other into a skipped node."""
    assert not issubclass(launcher_app.InstallCancelled, RuntimeError)

    async def runner() -> None:
        cancelled = launcher_app.JobController()
        # Already set before the call, which is the state a cancel pressed during the
        # previous command leaves behind.
        cancelled.cancel_event.set()
        with pytest.raises(launcher_app.InstallCancelled):
            await cancelled._run_process(
                sys.executable, "-c", "import time; time.sleep(30)", timeout=600
            )

        timed_out = launcher_app.JobController()
        with pytest.raises(RuntimeError) as failure:
            await timed_out._run_process(
                sys.executable,
                "-c",
                "import sys, time; print('Collecting torch'); "
                "sys.stdout.flush(); time.sleep(30)",
                timeout=0.5,
            )
        assert not isinstance(failure.value, launcher_app.InstallCancelled)
        assert "Collecting torch" in str(failure.value)
        assert "did not finish within" in str(failure.value)

    asyncio.run(runner())


def test_no_canceller_task_survives_a_finished_subprocess() -> None:
    """Eleven of these run per install; a leaked Event.wait() each would accumulate."""
    controller = launcher_app.JobController()

    async def runner() -> int:
        before = len(asyncio.all_tasks())
        await controller._run_process(sys.executable, "-c", "print('done')")
        await controller._run_process(sys.executable, "-c", "print('done')", timeout=30)
        return len(asyncio.all_tasks()) - before

    assert asyncio.run(runner()) == 0


def node_clone_harness(tmp_path, monkeypatch, failure, on_word="clone"):
    """Drive _install_custom_node with a git that dies part way through one command.

    Returns (controller, node, destination). The stub creates the directory before it
    raises, which is what git leaves behind when it is signalled mid-clone.
    """
    custom_nodes = tmp_path / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", tmp_path / ".venv-cu128")
    controller = launcher_app.JobController()
    destination = custom_nodes / "ComfyUI-KJNodes"
    repo = "https://github.com/kijai/ComfyUI-KJNodes"

    async def fake_process(*command, **_bounds) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        if on_word in normalized:
            destination.mkdir(parents=True, exist_ok=True)
            # Far enough along to exist, never far enough to answer for itself.
            (destination / ".git").mkdir(exist_ok=True)
            raise failure
        if "remote" in normalized and "get-url" in normalized:
            return 0, repo + "\n"
        if "cat-file" in normalized:
            return 1, "missing"
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)
    node = {
        "name": "ComfyUI-KJNodes",
        "repo": repo,
        "ref": "a" * 40,
        "install_requirements": True,
    }
    return controller, node, destination


def test_a_cancelled_clone_does_not_poison_the_folder(tmp_path, monkeypatch) -> None:
    """Otherwise the next attempt finds a directory it cannot identify, for good.

    destination.exists() is true, so the clone is skipped, `git remote get-url origin`
    runs against a .git that never got that far, and the node fails with "not the expected
    Git repository" until somebody deletes it by hand. A cancel must not create a fault
    that a failure does not.
    """
    controller, node, destination = node_clone_harness(
        tmp_path, monkeypatch, launcher_app.InstallCancelled()
    )

    with pytest.raises(launcher_app.InstallCancelled):
        asyncio.run(controller._install_custom_node(node))

    assert not destination.exists()


def test_a_timed_out_clone_does_not_poison_the_folder(tmp_path, monkeypatch) -> None:
    """The same hole, and it was open before cancel existed.

    The cleanup sits under `if returncode:`, which a raise never reaches - so a clone that
    hit its 600s bound has been leaving the poisoned directory since the bound was added.
    """
    controller, node, destination = node_clone_harness(
        tmp_path,
        monkeypatch,
        RuntimeError("git did not finish within 600s and was stopped after 600s."),
    )

    with pytest.raises(RuntimeError, match="did not finish within"):
        asyncio.run(controller._install_custom_node(node))

    assert not destination.exists()


def test_a_cancelled_fetch_or_checkout_keeps_the_repository(
    tmp_path,
    monkeypatch,
) -> None:
    """A clone that completed leaves a valid repository; the next run recovers on its own.

    Deleting it here would throw away a good checkout to no purpose.
    """
    for word in ("fetch", "checkout"):
        controller, node, destination = node_clone_harness(
            tmp_path / word, monkeypatch, launcher_app.InstallCancelled(), on_word=word
        )
        # Already cloned, so _install_custom_node takes the existing-folder branch.
        destination.mkdir(parents=True, exist_ok=True)

        with pytest.raises(launcher_app.InstallCancelled):
            asyncio.run(controller._install_custom_node(node))

        assert destination.exists(), f"a cancelled {word} deleted a valid repository"


def test_a_cancel_inside_a_node_install_is_not_swallowed(tmp_path, monkeypatch) -> None:
    """The per-node handler turns failures into warnings and carries on. Not this one."""
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()

    async def cancelling_install(node, *, on_step=None) -> None:
        on_step("installing dependencies")
        raise launcher_app.InstallCancelled()

    monkeypatch.setattr(controller, "_install_custom_node", cancelling_install)

    async def runner() -> int:
        before = len(asyncio.all_tasks())
        with pytest.raises(launcher_app.InstallCancelled):
            await controller._install_custom_nodes(
                [
                    {"name": "ComfyUI-KJNodes", "repo": "https://github.com/a/b", "ref": "a" * 40},
                    {"name": "Never-Reached", "repo": "https://github.com/c/d", "ref": "b" * 40},
                ]
            )
        return len(asyncio.all_tasks()) - before

    # The ticker is cancelled on the way out, in the same finally as every other exit.
    assert asyncio.run(runner()) == 0
    assert controller.state.warnings == [], "a cancel became a skipped-node warning"
    assert controller.state.current_file == "ComfyUI-KJNodes"


def test_a_cancel_during_the_comfyui_probes_is_not_swallowed(
    tmp_path,
    monkeypatch,
) -> None:
    """The probes catch RuntimeError and nothing wider, precisely so this cannot happen.

    A bare except Exception there would return "" and let the install carry on as though
    nobody had pressed anything.
    """
    store = launcher_app.Diagnostics()
    monkeypatch.setattr(launcher_app, "diagnostics", store)
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / ".git").mkdir(parents=True)
    (comfy_dir / "requirements.txt").write_text("torch\n", encoding="utf-8")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.JobController()

    async def cancelled(*_command, **_bounds) -> tuple[int, str]:
        raise launcher_app.InstallCancelled()

    monkeypatch.setattr(controller, "_run_process", cancelled)

    async def runner() -> int:
        before = len(asyncio.all_tasks())
        with pytest.raises(launcher_app.InstallCancelled):
            await controller._update_comfyui()
        return len(asyncio.all_tasks()) - before

    assert asyncio.run(runner()) == 0

    # str(InstallCancelled()) is "", which is falsy - so without the class-name fallback
    # this record would read back as a clean, unskipped success.
    record = store.export()["comfyui_update"]
    assert record["error"] == "InstallCancelled"
    assert record["skipped"] is False


def test_a_placement_is_on_the_disk_before_it_is_renamed(tmp_path, monkeypatch) -> None:
    """os.replace on an unflushed file leaves a full-length file with no contents.

    The sidecar exists so a crash mid-copy never leaves a truncated file at the real path,
    because _download_file's already-exists check trusts its length and skips the download
    for good. Renaming a file that is still entirely in page cache reaches the same end
    through a different door: a pod stop between the rename and writeback.
    """
    payload = b"durable-payload" * 4096
    source = tmp_path / "staged.part"
    source.write_bytes(payload)
    destination = tmp_path / "models" / "placed.safetensors"
    destination.parent.mkdir(parents=True)
    pretend_free_space(monkeypatch, 300 * 1024**3)

    order: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        order.append("fsync")
        return real_fsync(fd)

    def recording_replace(src, dst, *args, **kwargs):
        order.append("replace")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(launcher_app.os, "fsync", recording_fsync)
    monkeypatch.setattr(launcher_app.os, "replace", recording_replace)

    launcher_app.copy_into_place(source, destination)

    # Order, not timing: a duration assertion here would be measuring this machine.
    assert order == ["fsync", "replace"]
    assert destination.read_bytes() == payload


def test_the_restart_bound_is_configurable_and_no_longer_two_minutes(monkeypatch) -> None:
    """A real customer install ended at 100% carrying the old warning, and was fine.

    ComfyUI importing torch and scanning seven node packs off shared storage does not
    reliably finish in 120 seconds on a loaded host.
    """
    monkeypatch.delenv("COMFYUI_RESTART_TIMEOUT", raising=False)
    assert launcher_app.restart_timeout() == 300

    monkeypatch.setenv("COMFYUI_RESTART_TIMEOUT", "45")
    assert launcher_app.restart_timeout() == 45

    # A typo in a pod template must not be the reason ComfyUI never comes back.
    monkeypatch.setenv("COMFYUI_RESTART_TIMEOUT", "banana")
    assert launcher_app.restart_timeout() == 300


def test_the_restart_failure_says_how_long_it_actually_waited(monkeypatch) -> None:
    """"within two minutes" reads as a fault. The elapsed number reads as a fact."""
    controller = launcher_app.ComfyServiceController()
    monkeypatch.setenv("COMFYUI_RESTART_TIMEOUT", "0")

    class NeverReady:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url):
            class Response:
                status_code = 200

            return Response()

        async def post(self, _url, **_kwargs):
            class Response:
                status_code = 200

            return Response()

    monkeypatch.setattr(launcher_app.httpx, "AsyncClient", lambda **_kw: NeverReady())

    asyncio.run(controller._restart())

    assert controller.state.status == "error"
    assert "did not come back within 0s" in controller.state.error
    assert "waited" in controller.state.error


def test_aria2_report_lines_keeps_both_ends_of_a_long_run() -> None:
    """The collapse is at the start, and kept[-limit:] deleted exactly that.

    On a 123-second transfer these arrive about once a second, so the old tail kept
    72%->99% and threw away the first 83 seconds - while rate_samples for that same file
    shows 542 MB/s at t=10.5s decaying to ~200 for the remainder.
    """
    output = "\n".join(f"[#a1b2c3 {n}GiB/100GiB CN:16 DL:{n}MiB]" for n in range(100))

    lines = launcher_app.aria2_report_lines(output)

    assert len(lines) == 41
    # The first entry is the first line of the run, not the 61st.
    assert lines[0] == "[#a1b2c3 0GiB/100GiB CN:16 DL:0MiB]"
    assert lines[19] == "[#a1b2c3 19GiB/100GiB CN:16 DL:19MiB]"
    assert lines[20] == "… 60 lines omitted …"
    assert lines[21] == "[#a1b2c3 80GiB/100GiB CN:16 DL:80MiB]"
    assert lines[-1] == "[#a1b2c3 99GiB/100GiB CN:16 DL:99MiB]"


def test_a_short_run_is_kept_whole_with_no_marker() -> None:
    output = "\n".join(f"[#a1b2c3 CN:16 DL:{n}MiB]" for n in range(12))

    lines = launcher_app.aria2_report_lines(output)

    assert len(lines) == 12
    assert not any("omitted" in line for line in lines)


def report_store(files=(), nodes=(), update=None):
    """A Diagnostics with records already in it, newest first as the real one keeps them."""
    store = launcher_app.Diagnostics()
    for entry in files:
        record = store.begin_file(
            name=entry.get("name", "model.safetensors"),
            url="https://cdn.example/model.safetensors",
            size_bytes=entry.get("size_bytes", 0),
            transport=entry.get("transport", "aria2c"),
            staging=entry.get("staging", "container-disk"),
        )
        record.update(
            {
                key: value
                for key, value in entry.items()
                if key not in {"name", "size_bytes", "transport", "staging"}
            }
        )
        store.finish_file(record)
    for node in nodes:
        store.record_node(**node)
    if update is not None:
        store.record_comfyui_update(**update)
    return store


def a_big_file(name="Qwen Rapid AIO", rate=604 * 1024**2, **overrides):
    entry = {
        "name": name,
        "size_bytes": 28 * 1024**3,
        "average_bytes_per_second": rate,
        "fetch_seconds": 123.0,
        "place_seconds": 109.5,
        "progress_measurable": True,
    }
    entry.update(overrides)
    return entry


def test_the_report_renders_with_nothing_to_report(monkeypatch) -> None:
    """A pod that has installed nothing must still produce something readable."""
    pretend_free_space(monkeypatch, 300 * 1024**3)

    text = launcher_app.render_install_report(launcher_app.Diagnostics().export())

    assert "10sorLabs install report" in text
    assert "VERDICT" in text
    assert "Not enough data yet" in text
    assert text.endswith("\n")


def test_small_files_never_produce_a_verdict(monkeypatch) -> None:
    """An 80 MB upscaler at 122 MB/s is normal. Judging on it flags every healthy pod."""
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(
        files=[
            a_big_file(
                name="4x-upscaler",
                size_bytes=80 * 1024**2,
                rate=122 * 1024**2,
            )
        ]
    )

    text = launcher_app.render_install_report(store.export())

    assert "Not enough data yet" in text
    assert "network is healthy" not in text
    assert "network is busy" not in text


def test_the_verdict_says_which_of_the_three_a_pod_is(monkeypatch) -> None:
    """Always with the number in it: a verdict with no figure invites an argument."""
    pretend_free_space(monkeypatch, 300 * 1024**3)

    healthy = launcher_app.render_install_report(
        report_store(files=[a_big_file(rate=604 * 1024**2)]).export()
    )
    assert "network is healthy" in healthy
    assert "604 MB/s" in healthy

    busy = launcher_app.render_install_report(
        report_store(files=[a_big_file(rate=231 * 1024**2)]).export()
    )
    assert "network is busy" in busy
    assert "231 MB/s" in busy
    assert "deploying a new one usually helps" in busy

    congested = launcher_app.render_install_report(
        report_store(files=[a_big_file(rate=49 * 1024**2)]).export()
    )
    assert "network is very busy" in congested
    assert "49 MB/s" in congested


# 8.07 GB, the Qwen 3 8B text encoder from the pod this whole commit came off.
_QWEN_BYTES = 8_664_748_032


def a_skipped_and_a_real_download():
    """The live report that reported 588 MB/s for a pod that had measured 295.

    Both rows are the same file: the first attempt was cancelled part way, the second
    downloaded it. Oldest first in the rendered table, which is why the phantom row led.
    """
    store = launcher_app.Diagnostics()
    store.record_skipped_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
        verify_seconds=31.4,
    )
    record = store.begin_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
        transport="aria2c",
        staging="container-disk",
    )
    record.update(
        {
            "fetch_seconds": 28.1,
            "place_seconds": 9.2,
            "bytes_transferred": _QWEN_BYTES,
            "average_bytes_per_second": round(_QWEN_BYTES / 28.1, 1),
        }
    )
    store.finish_file(record)
    return store


def test_a_file_that_was_already_there_is_not_shown_as_a_download(monkeypatch) -> None:
    """It never downloaded. It was recorded as an 8.07 GB transfer taking zero seconds.

    Keeping the row is right - it is often the entire explanation for an install that
    finished in seconds - but it has to say what it is. A zero in a speed column reads as
    a failure, and this file did not fail.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)

    text = launcher_app.render_install_report(a_skipped_and_a_real_download().export())

    row = next(line for line in text.splitlines() if "already present" in line)
    assert row.strip().startswith("Qwen 3 8B text encoder")
    assert "8.07 GB" in row
    # Neither a duration nor a rate, because neither was ever measured.
    assert "0.0s" not in row
    assert not row.rstrip().endswith("0")


def test_the_total_counts_only_the_bytes_that_were_downloaded(monkeypatch) -> None:
    """The defect this commit is named for.

    The skipped file's bytes reached the size column and its zero reached neither timing
    column, so the total divided two files' bytes by one file's seconds and doubled the
    throughput of the machine it was describing - in the document a customer pastes into a
    support conversation to prove what their pod did.

    294 rather than the 295 on the pod: the total is bytes over elapsed, while a row
    carries its own measured rate from unrounded seconds. What is pinned here is that the
    total is one file's worth and not two.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)

    text = launcher_app.render_install_report(a_skipped_and_a_real_download().export())

    total = next(
        line for line in text.splitlines() if line.strip().startswith("total")
    )
    assert "8.07 GB" in total
    assert "16.14 GB" not in total
    assert total.split()[-1] == "294"
    assert "588" not in text
    # The timing columns are one file's too, not a sum across a file that took no time.
    assert "28.1s" in total and "9.2s" in total


def test_the_verdict_leaves_out_the_file_it_did_not_measure(monkeypatch) -> None:
    """Asserted on the verdict itself, not on a mean that happens to come out right.

    A skipped record used to be dropped here only because its rate is 0.0 and 0.0 is
    falsy. These two files are chosen so that accident is not enough to hide a regression:
    counting the skipped one halves 604 to 302 and moves the verdict from healthy to busy,
    so the wrong answer is a different sentence rather than a different number.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(files=[a_big_file(rate=604 * 1024**2)])
    store.record_skipped_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
    )

    text = launcher_app.render_install_report(store.export())

    assert "Downloads averaged 604 MB/s. This pod's network is healthy." in text
    assert "network is busy" not in text
    assert "302 MB/s" not in text


def test_a_report_of_nothing_but_skipped_files_does_not_divide_by_zero(monkeypatch) -> None:
    """Every file already on disk is a normal second run, and it has no elapsed time at
    all. The verdict says so in words rather than announcing a rate computed from nothing.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = launcher_app.Diagnostics()
    store.record_skipped_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
        verify_seconds=31.4,
    )

    text = launcher_app.render_install_report(store.export())

    assert "Not enough data yet" in text
    total = next(
        line for line in text.splitlines() if line.strip().startswith("total")
    )
    assert total.rstrip().endswith("nothing downloaded")
    # No size, no seconds, no rate. There is no figure here that would be true.
    assert not any(character.isdigit() for character in total), total


def test_a_cancelled_transfer_does_not_haunt_the_next_install(monkeypatch) -> None:
    """Where the live 588 actually came from.

    A cancel used to leave its record open. The record survived the install, and the next
    one's begin_file filed it - carrying the file's full size and no timings, because the
    epilogue that sets them never ran. Two rows for one file, one of them a phantom.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = launcher_app.Diagnostics()
    store.begin_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
        transport="aria2c",
        staging="container-disk",
    )
    store.cancel_in_flight()
    # The next install, which downloads it for real.
    record = store.begin_file(
        name="Qwen 3 8B text encoder",
        url="https://cdn.example/qwen.safetensors",
        size_bytes=_QWEN_BYTES,
        transport="aria2c",
        staging="container-disk",
    )
    record.update(
        {
            "fetch_seconds": 28.1,
            "place_seconds": 9.2,
            "bytes_transferred": _QWEN_BYTES,
            "average_bytes_per_second": round(_QWEN_BYTES / 28.1, 1),
        }
    )
    store.finish_file(record)

    text = launcher_app.render_install_report(store.export())

    assert "cancelled" in text
    total = next(
        line for line in text.splitlines() if line.strip().startswith("total")
    )
    assert "8.07 GB" in total
    assert total.split()[-1] == "294"
    assert "588" not in text


def test_an_update_still_running_is_in_the_report(monkeypatch) -> None:
    """record_comfyui_update was only called from the finally, so pressing Debug during an
    update produced a report with no COMFYUI UPDATE section at all.

    Measured live: the panel read "Updating ComfyUI, installing requirements, 2m17s" while
    the report for that same pod said nothing about an update. It is the slowest part of a
    MiniMax install, so that is exactly when somebody presses Debug.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(files=[a_big_file()])
    store.record_comfyui_update(
        workflow_id="minimax-h3",
        in_flight=True,
        phase="installing requirements",
        fetch_seconds=3.4,
        reset_seconds=1.1,
        total_seconds=137.0,
    )

    text = launcher_app.render_install_report(store.export())

    assert "COMFYUI UPDATE" in text
    assert "still running: installing requirements, 137.0s so far" in text
    # And the steps already behind it, which is what says the pause is the pip install.
    assert "fetched in 3.4s" in text
    assert "reset in 1.1s" in text
    assert "requirements in" not in text


def test_the_finished_update_replaces_the_running_one(monkeypatch) -> None:
    """One update per install, not a log of every tick. The ticker publishes once a second
    for as long as the update runs, so appending would put a hundred of them in the report.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(files=[a_big_file()])
    for elapsed in (12.0, 74.0, 137.0):
        store.record_comfyui_update(
            workflow_id="minimax-h3",
            in_flight=True,
            phase="installing requirements",
            total_seconds=elapsed,
        )
    store.record_comfyui_update(
        workflow_id="minimax-h3",
        fetch_seconds=3.4,
        reset_seconds=1.1,
        requirements_seconds=138.2,
        total_seconds=142.9,
    )

    text = launcher_app.render_install_report(store.export())

    assert text.count("COMFYUI UPDATE") == 1
    assert "still running" not in text
    assert "137.0s so far" not in text
    assert "requirements in 138.2s" in text
    assert store.export()["comfyui_update"]["in_flight"] is False


def test_the_storage_line_reads_the_progress_flag(monkeypatch) -> None:
    """progress_measurable is a storage-type detector; see storage_kind.

    And only on aria2c records. It defaults to True and is only ever reassigned on that
    branch, so reading it off an httpx record would tell every standard-tier customer they
    are on a Volume disk whatever they actually bought.
    """
    pretend_free_space(monkeypatch, 300 * 1024**3)

    local = launcher_app.render_install_report(
        report_store(files=[a_big_file(progress_measurable=True)]).export()
    )
    assert "Volume disk (local)" in local
    assert "Network volume, which is shared storage" not in local

    network = launcher_app.render_install_report(
        report_store(files=[a_big_file(progress_measurable=False)]).export()
    )
    assert "Network volume (shared)" in network
    assert "roughly a tenth of a local Volume disk" in network

    # The standard tier: httpx never sets the flag, so it cannot be read as an answer.
    standard = launcher_app.render_install_report(
        report_store(
            files=[a_big_file(transport="httpx", progress_measurable=True)]
        ).export()
    )
    assert "storage            unknown" in standard
    assert "Network volume, which is shared storage" not in standard

    empty = launcher_app.render_install_report(launcher_app.Diagnostics().export())
    assert "storage            unknown" in empty


def test_the_volume_line_never_reports_the_host_pool_as_yours(monkeypatch) -> None:
    """Two live pods read "284316.37 GB free of 893695.00 GB" for the volume while their
    container disk correctly read 198.21 GB of 200.00 GB.

    disk_usage answers for the filesystem /workspace landed on, which for a Network volume
    is shared host storage. Telling somebody they have 893 TB costs the whole report its
    credibility, including the lines that were right.
    """
    # A container disk that resolves, so the line below it is a real measurement and not
    # the "unknown" a machine without scratch space would produce.
    monkeypatch.setattr(launcher_app, "scratch_dir", lambda: Path("/scratch"))

    pretend_free_space(monkeypatch, 284_316 * 1024**3)
    shared = launcher_app.render_install_report(report_store(files=[a_big_file()]).export())

    volume = next(line for line in shared.splitlines() if "  volume " in line)
    assert volume.strip() == (
        "volume             284316.00 GB free (shared storage, so this is the host's "
        "pool rather than your volume)"
    )
    assert "free of" not in volume
    # The container disk is a real per-pod device and measured correctly on both pods.
    # It keeps its total, and this is the line that proves the change was surgical.
    assert "container disk     284316.00 GB free of 284316.00 GB" in shared

    # A volume small enough to be a volume says nothing extra.
    pretend_free_space(monkeypatch, 284 * 1024**3)
    plausible = launcher_app.render_install_report(
        report_store(files=[a_big_file()]).export()
    )

    volume = next(line for line in plausible.splitlines() if "  volume " in line)
    assert volume.strip() == "volume             284.00 GB free"
    assert "host's pool" not in plausible


def test_a_node_that_retried_gets_explained(monkeypatch) -> None:
    """A customer reading raw numbers will worry about it. It is normal."""
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(
        files=[a_big_file()],
        nodes=[
            {
                "name": "ComfyUI_FaceAnalysis",
                "clone_seconds": 4.5,
                "dependencies_seconds": 227.0,
                "total_seconds": 231.9,
                "retried_with_isolation": True,
            }
        ],
    )

    text = launcher_app.render_install_report(store.export())

    assert "ComfyUI_FaceAnalysis needed a second install attempt" in text
    assert "That is normal and it succeeded." in text
    assert "CUSTOM NODES" in text


def test_the_report_carries_no_url_and_no_path(monkeypatch) -> None:
    """It exists to be pasted into a chat window. Scanned, not spot-checked."""
    pretend_free_space(monkeypatch, 300 * 1024**3)
    store = report_store(
        files=[a_big_file()],
        nodes=[{"name": "ComfyUI-Impact-Pack", "total_seconds": 11.5}],
        update={
            "workflow_id": "minimax-h3",
            "error": (
                "fatal: not a git repository: /workspace/runpod-slim/ComfyUI/.git via "
                "https://github.com/x/y.git?token=secret"
            ),
        },
    )
    store.note_aria2_lines(
        launcher_app.aria2_report_lines(
            "[WARN] /workspace/models/flux.part from "
            "https://pub-9c2f.r2.cloudflarestorage.com/y?X-Amz-Signature=def"
        )
    )

    text = launcher_app.render_install_report(store.export())

    assert "://" not in text
    assert "X-Amz-Signature" not in text
    for line in text.splitlines():
        for token in line.split():
            assert not token.startswith("/"), f"a path reached the report: {token!r}"


def test_the_report_endpoint_answers_as_plain_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")
    with TestClient(launcher_app.app) as client:
        response = client.get("/api/diagnostics/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "10sorLabs install report" in response.text


def test_the_debug_button_follows_the_tier_it_is_gated_to() -> None:
    """One constant decides this, and it reuses the upsell's own fast-tier condition.

    The two must not drift: a button that appears for someone who cannot be helped by it
    is worse than no button.
    """
    app_js = launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js"
    rows = run_upsell_harness(
        [
            {"name": "fast", "account": account(True, "ok", "fast")},
            {"name": "standard", "account": account(True, "ok", "standard")},
        ],
        app_js,
    )

    assert rows["fast"]["debugHidden"] is False
    assert rows["standard"]["debugHidden"] is True

    # And the constant really is the only thing holding it there.
    source = app_js.read_text(encoding="utf-8")
    assert "const DEBUG_REPORT_REQUIRES_FAST = true;" in source
    flipped = source.replace(
        "const DEBUG_REPORT_REQUIRES_FAST = true;",
        "const DEBUG_REPORT_REQUIRES_FAST = false;",
    )
    with tempfile.TemporaryDirectory() as workspace:
        # Never written beside the real app.js: launcher/static is served at "/" and ships
        # to every pod, so a kill between write and unlink would put it in the zip.
        copy = Path(workspace) / "app.js"
        copy.write_text(flipped, encoding="utf-8")
        both = run_upsell_harness(
            [
                {"name": "fast", "account": account(True, "ok", "fast")},
                {"name": "standard", "account": account(True, "ok", "standard")},
            ],
            copy,
        )
    assert both["fast"]["debugHidden"] is False
    assert both["standard"]["debugHidden"] is False


def test_only_the_fast_tier_gets_the_rapidcache_pill() -> None:
    """The pill's presence is the signal, so standard keeps its plain text treatment."""
    rows = run_upsell_harness(
        [
            {"name": "fast", "account": account(True, "ok", "fast")},
            {"name": "standard", "account": account(True, "ok", "standard")},
            {"name": "checking", "account": account(True, "unavailable", "")},
        ],
        launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js",
    )

    assert rows["fast"]["tierText"] == "RapidCache"
    assert rows["fast"]["tierPill"] is True
    assert rows["standard"]["tierText"] == "Standard downloads"
    assert rows["standard"]["tierPill"] is False
    assert rows["checking"]["tierPill"] is False


def test_the_rate_says_which_number_it_is() -> None:
    """77.5 MB/s then 1.10 GB/s reads as a wild swing. It is the network, then the disk."""
    rows = run_upsell_harness(
        [
            {
                "name": "downloading",
                "status": {
                    "status": "running",
                    "stage": "downloading",
                    "message": "Downloading Qwen Rapid AIO…",
                    "bytes_per_second": 81_000_000,
                },
            },
            {
                "name": "placing",
                "status": {
                    "status": "running",
                    "stage": "installing",
                    "message": "Placing Qwen Rapid AIO… 40%",
                    "bytes_per_second": 1_180_000_000,
                },
            },
        ],
        launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js",
    )

    assert "downloading" in rows["downloading"]["metrics"]
    assert "saving to disk" in rows["placing"]["metrics"]
    # The number is still there; the phase is added, not substituted.
    assert "/s" in rows["downloading"]["metrics"]
    assert "/s" in rows["placing"]["metrics"]


def test_the_debug_report_markup_is_served(tmp_path, monkeypatch) -> None:
    """Python, not the harness: the fake DOM answers every selector with one element, so
    a harness assertion about the page's contents would pass on an empty page."""
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")
    with TestClient(launcher_app.app) as client:
        page = client.get("/").text

    assert 'id="debug-button"' in page
    assert 'id="debug-report-text"' in page
    assert 'id="debug-copy"' in page
    assert 'id="debug-download"' in page


def test_the_help_page_is_served_as_part_of_the_page(tmp_path, monkeypatch) -> None:
    """Python, not the harness. The fake DOM answers every selector with a single element
    claiming to be "workflows", so a harness assertion about the sidebar would pass on a
    page with one entry, or none. A vacuous green test is worse than no test.

    Inline rather than fetched, for the same reason plus one worse: the harness stubs
    fetch as a rejected promise, so a fetched docs view would render empty under test
    while every assertion about it appeared to pass.
    """
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", tmp_path / "ComfyUI")
    with TestClient(launcher_app.app) as client:
        page = client.get("/").text

    assert len(re.findall(r'data-view-target="([^"]+)"', page)) == 5
    assert 'data-view-target="docs"' in page
    assert 'id="view-docs"' in page
    # The content really is in the page, not a placeholder waiting on a fetch.
    #
    # Asserted on structure and on headings, never on a sentence of body copy. The first
    # version of this pinned the opening line verbatim and broke the moment that line was
    # edited, which is a test punishing the one kind of change this page should invite.
    # Headings are the contract - they are the page's shape, and renaming one is a
    # deliberate act worth failing a test over.
    body = page.split('<article class="docs-body">')[1].split("</article>")[0]
    headings = re.findall(r"<h2>([^<]+)</h2>", body)
    assert "What RapidCache is" in headings
    assert "What a pod actually is" in headings
    assert "What we promise" in headings
    assert len(headings) >= 10
    # Prose, not an empty shell: tags stripped, this is a page rather than a stub.
    assert len(re.sub(r"<[^>]+>", " ", body).split()) > 500


def test_the_help_page_speaks_in_sentences_not_dashes() -> None:
    """An em dash reads as written rather than spoken, and this page is a person talking.

    Rewriting rather than substituting was the point: a comma in place of every dash makes
    run-ons that are worse than what you started with.
    """
    html = static_file("index.html")
    docs = html[html.index('id="view-docs"') : html.index("</article>")]

    offenders = [
        line.strip()
        for line in docs.splitlines()
        # "-->" and "<!--" are markup, not copy.
        if ("—" in line or "–" in line)
        or ("--" in line and "<!--" not in line and "-->" not in line)
    ]
    assert offenders == [], offenders


def test_every_help_page_image_resolves_or_disappears_by_itself() -> None:
    """None of the screenshots have shipped yet, and a broken image icon in the middle of
    a help page is worse than no picture at all."""
    html = static_file("index.html")
    docs_dir = launcher_app.STATIC_DIR / "docs"
    sources = re.findall(r'<img src="(docs/[^"]+)"', html)

    assert sources, "the help page references no images at all"
    for source in sources:
        target = launcher_app.STATIC_DIR / source
        if target.is_file():
            continue
        # Not shipped yet, so it must be inside a figure the error handler can hide.
        block = html[: html.index(source)]
        assert block.rstrip().endswith(
            '<img src="'
        ) or 'class="docs-figure"' in block[-400:], (
            f"{source} is missing and is not inside a .docs-figure"
        )

    js = static_file("app.js")
    assert '.docs-figure img' in js
    assert 'addEventListener("error"' in js
    assert "figure.hidden = true" in js


def test_a_figure_whose_image_already_failed_is_hidden_too() -> None:
    """The listener on its own never fired on a pod, and the live page showed broken
    image icons with their alt text underneath.

    These <img> tags are inline in index.html and app.js is deferred, so a missing file
    fires its error event during parse - before this script exists - and never fires it
    again. The old test drove the listener by hand, so it passed against exactly the code
    that was failing. This one presents what the DOM actually looks like afterwards and
    never touches the handler.
    """
    rows = run_upsell_harness(
        [
            {"name": "already-failed", "docsImage": {"complete": True, "naturalWidth": 0}},
            {"name": "loaded", "docsImage": {"complete": True, "naturalWidth": 1280}},
            {"name": "still-loading", "docsImage": {"complete": False}},
            # The listener still has to work for an image that fails after we attach.
            {"name": "fails-later", "docsImageError": True},
        ],
        launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js",
    )

    assert rows["already-failed"]["docsFigureHidden"] is True
    assert rows["fails-later"]["docsFigureHidden"] is True
    # And a picture that is fine stays on the page.
    assert rows["loaded"]["docsFigureHidden"] is False
    assert rows["still-loading"]["docsFigureHidden"] is False


def test_docs_is_a_real_view_that_the_switcher_accepts() -> None:
    """The harness, because allowedViews in selectView is the logic under test here."""
    rows = run_upsell_harness(
        [
            {"name": "docs", "hash": "#docs"},
            {"name": "nonsense", "hash": "#not-a-view"},
        ],
        launcher_app.SOURCE_ROOT / "launcher" / "static" / "app.js",
    )

    # Both scenarios load app.js, which calls selectView through initialise(). A view
    # name it rejects falls back to workflows, and an unhandled throw would fail the run.
    assert rows["docs"]["name"] == "docs"
    assert rows["nonsense"]["name"] == "nonsense"

    js = static_file("app.js")
    declared = re.search(r"const VIEW_NAMES\s*=\s*\[([^\]]*)\]", js).group(1)
    assert '"docs"' in declared
