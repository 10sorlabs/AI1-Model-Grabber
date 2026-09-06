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
- A Custom Models queue for direct downloads into any ComfyUI model folder.
- A button that routes from the launcher to the matching RunPod ComfyUI proxy.
- Optional launcher updates from GitHub at container startup.

The current catalog contains four production installers:

- Krea 2 Extended (approximately 25.4 GB)
- Image Edit (approximately 17.8 GB)
- Motion Control (approximately 26.5 GB)
- MiniMax H3 (approximately 63.4 GB)

The catalog installs only models, supporting files and custom nodes. Product
workflow JSON files are deliberately not included.

Each preset continues after an individual model or custom-node failure. Failed
items are shown as warnings at the end so the remaining downloads and node
installs are not discarded. When an existing custom-node checkout does not yet
contain its pinned commit, the launcher fetches that commit before checkout.
After every preset installation, ComfyUI is automatically restarted through
ComfyUI Manager and the launcher waits until port 8188 is ready again.

## Custom models

The **Custom models** section accepts a direct HTTP(S) model link and a target
folder inside `ComfyUI/models`. It includes ComfyUI's standard model locations,
discovers existing folders on the pod and can create a new custom subfolder.

Every click adds one download to a single serial queue. Custom downloads always
start from scratch: an existing `.part` file is removed before the request. A
complete model with the same filename and recognized size/checksum is skipped
and shown under **Downloaded models** instead of being overwritten.

## Custom nodes

The **Custom nodes** section accepts GitHub repository links and processes every
click through one serial install queue. Repositories are cloned into a temporary
folder, their `requirements.txt` is installed with ComfyUI's Python environment,
and the completed folder is then moved into `ComfyUI/custom_nodes`.

An existing checkout with the same Git origin is shown as already found. Failed
clones or dependency installs remove their temporary folder instead of leaving
an incomplete custom node behind. Newly installed nodes require a ComfyUI
restart before they become available; the **Restart ComfyUI** button performs
that restart without restarting the pod.

## RunPod template

Use:

```text
Container image: 10sorllabs/comfyui-workflow-launcher:1.0
HTTP ports:      3000, 8188, 8888
Container disk:  at least 60 GB for the largest individual installer
```

No persistent volume is required. The public service URLs follow RunPod's normal
format:

```text
https://POD_ID-3000.proxy.runpod.net
https://POD_ID-8188.proxy.runpod.net
https://POD_ID-8888.proxy.runpod.net
```

Set `JUPYTER_PASSWORD` in the RunPod template before exposing JupyterLab.
Set `HF_TOKEN` when using Dataset Generator or Image Edit, and make sure the
token's account has accepted the applicable Hugging Face model licenses.
Images published by the project's GitHub Actions workflow can instead include
the dedicated `HF_TOKEN` repository secret as a baked download
credential. A runtime `HF_TOKEN` always takes precedence over the baked file.

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
  "update_comfyui": true,
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

Set `update_comfyui` to `true` when a preset requires the newest ComfyUI code.
Before downloading that preset, the launcher updates from
`Comfy-Org/ComfyUI:master` and installs its requirements into the ComfyUI Python
environment. The standard end-of-preset restart then loads the new version.

Supported file authentication values are `none`, `huggingface`, `civitai` and
`github`. Corresponding environment names are documented in `.env.example`.
Download URLs and authentication details are not returned by the public catalog
API.

## Local Windows installers

Standalone Windows installers for Dataset Generator, Krea 2 and MiniMax H3 live
in `local-installers/`. Copy the required `.bat` file directly into the main
ComfyUI directory—the folder containing `models` and `custom_nodes`—and run it
while ComfyUI is stopped. The scripts verify SHA-256 checksums, install pinned
custom-node versions and their requirements, and retain successful items if a
later item fails. Dataset Generator reads a gated-model token from `HF_TOKEN` or
prompts for one. MiniMax H3 also updates ComfyUI before installing its files.

## Security note

Do not treat a token supplied to a user-controlled pod as secret. A person with
Jupyter, SSH or container access can inspect the environment, filesystem and
running processes. Never commit tokens to GitHub.

The automated publishing workflow can deliberately place the dedicated,
fine-grained `HF_TOKEN` Actions secret in the final image so public
template users do not need to configure Hugging Face themselves. This keeps the
credential out of source control and build logs, but it does not make the token
secret from anyone who can pull or inspect the final image. Treat that token as
public, narrowly scoped and disposable.

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
