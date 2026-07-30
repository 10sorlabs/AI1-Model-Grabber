import asyncio
import hashlib
import importlib
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient


os.environ["DEMO_DURATION_OVERRIDE"] = "0.1"
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
        assert "files" not in workflows[0]
        assert "url" not in workflows[0]


def test_demo_reaches_complete_and_keeps_comfy_url() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.post("/api/install/foundation-test")
        assert response.status_code == 200

        deadline = time.monotonic() + 3
        status = {}
        while time.monotonic() < deadline:
            status = client.get("/api/status").json()
            if status["status"] == "complete":
                break
            time.sleep(0.03)

        assert status["status"] == "complete"
        assert status["percent"] == 100
        assert status["comfy_url"] == "https://test-pod-8188.proxy.runpod.net"


def test_disabled_workflow_cannot_start() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.post("/api/install/workflow-03")
        assert response.status_code == 400


def test_frontend_is_served() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "10sorLabs Model Grabber" in response.text

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
