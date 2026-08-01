import asyncio
import hashlib
import importlib
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient


os.environ["RUNPOD_POD_ID"] = "test-pod"

launcher_app = importlib.import_module("launcher.app")


def test_health_and_public_catalog() -> None:
    with TestClient(launcher_app.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.get("/api/catalog")
        assert response.status_code == 200
        workflows = response.json()["workflows"]
        assert len(workflows) == 5
        assert sum(not item.get("disabled", False) for item in workflows) == 4
        assert "files" not in workflows[0]
        assert "custom_nodes" not in workflows[0]
        assert "url" not in workflows[0]


def test_catalog_contains_installers_but_no_product_workflows() -> None:
    catalog = launcher_app.load_catalog()
    enabled = [item for item in catalog["workflows"] if not item.get("disabled")]

    assert [item["id"] for item in enabled] == [
        "image-generation",
        "dataset-generator",
        "image-edit",
        "motion-control",
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


def test_disabled_workflow_cannot_start() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.post("/api/install/workflow-05")
        assert response.status_code == 400


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
