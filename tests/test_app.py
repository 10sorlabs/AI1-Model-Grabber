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

    async def fake_install(node) -> None:
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
        (
            str(comfy_python),
            "-m",
            "pip",
            "install",
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
        "--summary-interval=0",
        "--console-log-level=warn",
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
            return b"", b""

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
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
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

  const document = {
    // app.js spreads the result and reads .dataset on each entry.
    querySelectorAll: (sel) => {
      const one = fakeElement();
      one.dataset.view = "workflows";
      one.dataset.viewTarget = "workflows";
      return [one];
    },
    querySelector: (sel) => element(sel),
    getElementById: (id) => element("#" + id),
    createElement: () => fakeElement(),
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
  if (scenario.account) {
    context.renderAccount(scenario.account);
  }

  results.push({
    name: scenario.name,
    upsellHidden: nodes["#rapidcache-upsell"].hidden,
    hintHidden: nodes["#rapidcache-signin-hint"].hidden,
    videoHidden: video ? video.hidden : null,
    videoSrc: video ? video.src : null,
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
        elif "cat-file" in command or ("remote" in command and "get-url" in command):
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
