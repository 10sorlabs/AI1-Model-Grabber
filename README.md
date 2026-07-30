# 10sorLabs AI1 Model Grabber

A lightweight, single-page workflow launcher that runs next to stock ComfyUI on
RunPod.

## What is included

- Stock `runpod/comfyui:cuda12.8`, pinned to a known image digest.
- ComfyUI on port `8188`.
- JupyterLab on port `8888`.
- 10sorLabs Model Grabber on port `3000`.
- Resumable `.part` downloads with visible progress.
- Workflow and custom-node installation from a declarative JSON catalog.
- A button that routes from the launcher to the matching RunPod ComfyUI proxy.
- Optional launcher updates from GitHub at container startup.

The current catalog contains one simulated foundation test and four disabled
placeholders. It does not download any real models yet.

## RunPod template

Use:

```text
Container image: 10sorllabs/comfyui-workflow-launcher:1.0
HTTP ports:      3000, 8188, 8888
Container disk:  large enough for the biggest selected workflow
```

No persistent volume is required. The public service URLs follow RunPod's normal
format:

```text
https://POD_ID-3000.proxy.runpod.net
https://POD_ID-8188.proxy.runpod.net
https://POD_ID-8888.proxy.runpod.net
```

Set `JUPYTER_PASSWORD` in the RunPod template before exposing JupyterLab.

## Remote UI updates

At startup, the baked bootstrapper checks:

```text
LAUNCHER_GITHUB_REPO=10sorlabs/AI1-Model-Grabber
LAUNCHER_GITHUB_REF=main
LAUNCHER_AUTO_UPDATE=1
```

If the repository is public, no GitHub credential is required. If it is private,
the pod needs `GITHUB_TOKEN`. When GitHub is unavailable or the downloaded source
is invalid, the launcher safely falls back to the version baked into the image.

Changes to HTML, CSS, JavaScript, the Python launcher or `catalog/workflows.json`
therefore appear on newly started pods without rebuilding the image, provided
they do not introduce new Python packages.

## Workflow catalog

Workflow tiles live in `catalog/workflows.json`. A real workflow can define files
and custom nodes:

```json
{
  "id": "example-workflow",
  "title": "Example Workflow",
  "description": "Installs everything needed for the example.",
  "badge": "IMAGE",
  "accent": "#b8ff5a",
  "estimated_size": "18.4 GB",
  "files": [
    {
      "name": "example-model.safetensors",
      "url": "https://huggingface.co/owner/repo/resolve/main/model.safetensors",
      "destination": "models/diffusion_models/example-model.safetensors",
      "size_bytes": 123456789,
      "sha256": "optional-sha256",
      "auth": "huggingface"
    },
    {
      "name": "Example workflow",
      "url": "https://example.com/workflow.json",
      "destination": "user/default/workflows/example.json",
      "auth": "none"
    }
  ],
  "custom_nodes": [
    {
      "name": "Example-ComfyUI-Node",
      "repo": "https://github.com/example/Example-ComfyUI-Node.git",
      "ref": "pin-a-tag-or-commit-here",
      "install_requirements": true
    }
  ]
}
```

Supported file authentication values are `none`, `huggingface`, `civitai` and
`github`. Corresponding environment names are documented in `.env.example`.
Download URLs and authentication details are not returned by the public catalog
API.

## Security note

Do not treat a token supplied to a user-controlled pod as secret. A person with
Jupyter, SSH or container access can inspect the environment and running
processes. Never bake personal tokens into the image or commit them to GitHub.

For production gated models, prefer one of:

1. Bake distributable model files into independent Docker layers.
2. Require each pod owner to provide a fine-grained read-only token.
3. Issue short-lived download URLs from a separate trusted service.

The launcher supports `HF_TOKEN` and `CIVITAI_TOKEN` for controlled/private use,
but it cannot hide those values from the owner of the pod.

## Local launcher development

Create a Python environment and install:

```text
pip install -r requirements-launcher.txt pytest
```

Then run:

```text
DEMO_DURATION_OVERRIDE=1 python -m uvicorn launcher.app:app --port 3000
```

On Windows PowerShell, set environment variables with `$env:NAME="value"` first.

## Build manually

```text
docker build -t 10sorllabs/comfyui-workflow-launcher:1.0 .
docker push 10sorllabs/comfyui-workflow-launcher:1.0
```

Alternatively, run the included GitHub Actions workflow after adding repository
secrets named `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

