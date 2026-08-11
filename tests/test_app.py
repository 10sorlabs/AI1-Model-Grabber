import asyncio
import contextlib
import hashlib
import importlib
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ["RUNPOD_POD_ID"] = "test-pod"

launcher_app = importlib.import_module("launcher.app")
launcher_remote = importlib.import_module("launcher.remote")


@pytest.fixture(autouse=True)
def reset_remote_catalog_state():
    """The catalog cache and its one-shot log flags outlive a single test."""
    launcher_remote._reset_state()
    yield
    launcher_remote._reset_state()


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

    async def fake_process(*command) -> tuple[int, str]:
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

    async def fake_process(*command) -> tuple[int, str]:
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


def test_remote_catalog_with_a_bad_workflow_id_is_rejected(monkeypatch) -> None:
    body = json.dumps(
        {"version": 3, "workflows": [{"id": "Not A Valid Id", "files": []}]}
    ).encode("utf-8")

    with catalog_api(body) as base:
        monkeypatch.setenv("LCT_API_BASE", base)
        with pytest.raises(RuntimeError, match="Invalid workflow id"):
            launcher_app.load_catalog(fresh=True)


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
    assert "-x16" in recorded[0]
    assert "-s16" in recorded[0]
    assert "--continue=true" in recorded[0]
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
    fake_aria2c(monkeypatch, payload, recorded)
    controller = launcher_app.JobController()

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

    destination = comfy_dir / "models" / "checkpoints" / "tampered.safetensors"
    assert not destination.exists()
    assert destination.with_name(destination.name + ".part").read_bytes() == payload


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

    download_one_file(
        controller,
        {
            "name": "Polled model",
            "url": "https://cdn.example/polled.safetensors",
            "destination": "models/checkpoints/polled.safetensors",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "auth": "none",
            "parallel": True,
        },
    )

    # The poller reported real mid-flight progress rather than 0 then done.
    assert any(0 < value < len(payload) for value in observed)
    assert any(speed > 0 for speed in speeds)
    assert 0 < controller.state.percent < 100
    # …and it was already dead when the shared epilogue wrote the verify message,
    # so the UI does not show a stale speed while the file is being hashed.
    assert controller.state.message.startswith("Verifying")
    assert controller.state.bytes_per_second == 0


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


def test_custom_node_ref_must_be_a_pinned_commit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path / "custom_nodes")
    controller = launcher_app.JobController()
    commands: list = []

    async def fake_process(*command) -> tuple[int, str]:
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
