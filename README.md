<p align="center">
  <img src="launcher/static/logo.png" alt="10sorLabs" width="96">
</p>

<h1 align="center">10sorLabs AI1 Model Grabber</h1>

<p align="center">
  Deploy curated ComfyUI model stacks on RunPod without manually hunting down
  model files, custom nodes or Python dependencies.
</p>

<p align="center">
  <strong>Developed by <a href="https://10sorlabs.com">10sorLabs</a></strong>
  ·
  <a href="https://10sorlabs.com">Workflows, tutorials and AI creator tools</a>
</p>

## What it does

AI1 Model Grabber runs beside ComfyUI and turns large, multi-step setups into
guided installs. Choose a preset and the launcher downloads the reviewed model
files, installs the required custom nodes, verifies the results and restarts
ComfyUI when everything is ready.

It is designed for creators who want to spend their GPU time generating—not
debugging paths, dependencies and missing files.

### Highlights

- Ten curated image and video model presets.
- Visible download progress with resumable partial transfers.
- SHA-256 and file-size verification where available.
- Pinned custom-node revisions for reproducible installs.
- Automatic dependency installation in ComfyUI's Python environment.
- Custom model downloads into any standard or discovered model folder.
- Custom-node installation from GitHub repositories.
- Failure isolation: one failed item does not discard successful work.
- Automatic ComfyUI restart and readiness checks after installation.
- Optional launcher and catalog updates from this repository at startup.
- Built-in Hugging Face and Civitai download access in the published image.

## Quick start on RunPod

Use these template settings:

| Setting | Value |
| --- | --- |
| Container image | `10sorllabs/comfyui-workflow-launcher:1.0` |
| HTTP ports | `3000`, `8188`, `8888` |
| Container disk | 200 GB recommended for the largest presets |
| Persistent volume | Optional; mount at `/workspace` when used |
| GPU | NVIDIA |

Set `JUPYTER_PASSWORD` before exposing JupyterLab publicly.

After the pod starts, open:

| Service | URL |
| --- | --- |
| AI1 Model Grabber | `https://POD_ID-3000.proxy.runpod.net` |
| ComfyUI | `https://POD_ID-8188.proxy.runpod.net` |
| JupyterLab | `https://POD_ID-8888.proxy.runpod.net` |

Open AI1 Model Grabber first, choose a preset and start the installation. The
launcher opens the matching ComfyUI proxy when the setup is complete.

> The published image already contains the project's read-only Hugging Face and
> Civitai download credentials. Normal RunPod users do not need to add
> `HF_TOKEN` or `CIVITAI_TOKEN` manually.

## Available presets

| Preset | Estimated model size | Purpose |
| --- | ---: | --- |
| Krea 2 Extended | ~25.4 GB | Extended Krea 2 image stack |
| WAN Motion Control | ~45.4 GB | WAN motion-control generation |
| Motion Control GOD Edition | ~61.3 GB | Expanded motion-control stack |
| MiniMax H3 | ~63.4 GB | MiniMax H3 image/video setup |
| KRQW2.1 | ~44.3 GB | Krea 2 Turbo and Qwen Image 2.1 stack |
| DarkDesire H3 | ~83.2 GB | DarkDesire Krea 2 and H3 stack |
| MiniMax H3 V2 | Custom setup | CUDA 13 runtime and reviewed native setup |
| Krea2 Pro | Custom setup | Extended Krea 2 production stack |
| H3 Motion Control | Custom setup | H3 motion control with Viggle Remake |
| KRAST | Custom setup | Curated KRAST model and node stack |

Sizes cover catalog model files and do not include temporary working space,
Python packages, caches or files you add yourself.

## What is included

The image provides:

- Stock RunPod ComfyUI, pinned to a reviewed base-image digest.
- ComfyUI on port `8188`.
- JupyterLab on port `8888`.
- AI1 Model Grabber on port `3000`.
- A declarative workflow catalog.
- Preinstalled dependencies used by the catalog's custom nodes.
- Automatic GitHub-based launcher updates with a safe baked fallback.

The public catalog installs models, supporting files and custom nodes. It does
not distribute 10sorLabs' product workflow JSON files.

For complete workflows, tutorials, prompts and creator playbooks, visit
**[10sorLabs.com](https://10sorlabs.com)**.

## Using AI1 Model Grabber

### Preset installs

Each preset is processed as a queue of verified model downloads and reviewed
custom-node installs. Successful items are retained when a later item fails, and
the final screen reports anything that still needs attention.

Most third-party nodes are pinned to a 40-character commit SHA. The reviewed
first-party `ComfyUI-10SOR-Regeneration-Gate` integration is the only supported
moving-branch exception and follows `main` with fast-forward-only updates.

### Custom models

Paste a direct HTTP(S) model URL, choose a target folder and add it to the queue.
The launcher supports ComfyUI's standard model folders, discovers existing
folders and can create a new subfolder.

Downloads run serially. A valid existing file is kept; incomplete or invalid
files are replaced safely.

### Custom nodes

Paste a GitHub repository URL to clone it into `ComfyUI/custom_nodes`. The
launcher installs its `requirements.txt` in ComfyUI's Python environment and
only moves the completed checkout into place after installation succeeds.

New nodes require a ComfyUI restart. Use the launcher's **Restart ComfyUI**
button—there is no need to restart the entire pod.

## Download credentials

Images published by this repository's GitHub Actions workflow contain dedicated,
read-only `HF_TOKEN` and `CIVITAI_TOKEN` credentials. They are used for gated
or authenticated catalog downloads so public template users do not need to
configure tokens for every pod.

Runtime environment variables take precedence over the baked credentials:

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Override Hugging Face download access |
| `HUGGING_FACE_HUB_TOKEN` | Alternative Hugging Face variable |
| `CIVITAI_TOKEN` | Override Civitai download access |
| `CIVITAI_API_TOKEN` | Alternative Civitai variable |

Credentials are restricted to their matching hosts. A catalog entry cannot send
a Hugging Face, Civitai or GitHub credential to an unrelated domain.

## Remote updates

The container starts with:

```text
LAUNCHER_GITHUB_REPO=10sorlabs/AI1-Model-Grabber
LAUNCHER_GITHUB_REF=main
LAUNCHER_AUTO_UPDATE=1
```

On startup, the bootstrapper checks this public repository for launcher, catalog
and frontend updates. If GitHub is unavailable or the downloaded source is
invalid, the container falls back to the version baked into the image.

Changes that introduce new system or Python dependencies still require a new
Docker image build.

## Useful environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `LAUNCHER_PORT` | `3000` | AI1 Model Grabber port |
| `LAUNCHER_AUTO_UPDATE` | `1` | Enable GitHub updates at startup |
| `LAUNCHER_GITHUB_REPO` | `10sorlabs/AI1-Model-Grabber` | Update source |
| `LAUNCHER_GITHUB_REF` | `main` | Update branch or ref |
| `COMFYUI_DIR` | `/workspace/runpod-slim/ComfyUI` | ComfyUI installation |
| `COMFYUI_PUBLIC_URL` | Auto-detected | Explicit ComfyUI browser URL |
| `JUPYTER_PASSWORD` | Empty | Password for public JupyterLab access |
| `GITHUB_TOKEN` | Empty | Only needed for a private update repository |

See [`.env.example`](.env.example) for the complete list.

## Local Windows installers

Standalone installers live in [`local-installers/`](local-installers):

- Dataset Generator
- Krea 2
- MiniMax H3

Copy the required `.bat` file into the main ComfyUI directory—the folder that
contains `models` and `custom_nodes`—and run it while ComfyUI is stopped.

These scripts verify SHA-256 checksums, install pinned custom-node revisions and
retain completed items if a later step fails. The Dataset Generator installer
reads `HF_TOKEN` or prompts for one because local Windows installers do not
contain the Docker image's baked credential files.

## Catalog format

Preset definitions live in [`catalog/workflows.json`](catalog/workflows.json).
A minimal entry looks like this:

```json
{
  "id": "example-workflow",
  "title": "Example Workflow",
  "description": "Install the example model stack.",
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
    }
  ],
  "custom_nodes": [
    {
      "name": "Example-ComfyUI-Node",
      "repo": "https://github.com/example/Example-ComfyUI-Node.git",
      "ref": "pin-a-40-character-commit-sha-here",
      "install_requirements": true
    }
  ]
}
```

Supported file authentication values are `none`, `huggingface`, `civitai`
and `github`.

### Reviewed special setup

The catalog can select closed, code-defined `runtime_profile` values for setup
that cannot be represented as ordinary files and Python requirements. Remote
catalog data cannot execute arbitrary shell commands.

Current reviewed profiles cover:

- SageAttention builds for supported NVIDIA compute capabilities.
- MiniMax H3 V2's PyTorch CUDA 13 runtime and verification.
- NVIDIA NGX userspace-library recovery without replacing RunPod's driver stack.

`model_links` can expose one downloaded model at an additional path using a
hard link, with a copy fallback when the filesystem does not support links.

## Development

The launcher requires Python 3.12 or newer.

```text
python -m pip install -r requirements-launcher.txt pytest
python -m pytest
python -m uvicorn launcher.app:app --port 3000
```

For the short demo duration used during frontend work:

```text
DEMO_DURATION_OVERRIDE=1 python -m uvicorn launcher.app:app --port 3000
```

On PowerShell, set environment variables with `$env:NAME="value"`.

## Build and publish

Manual build:

```text
docker build -t 10sorllabs/comfyui-workflow-launcher:1.0 .
docker push 10sorllabs/comfyui-workflow-launcher:1.0
```

The GitHub Actions publishing workflow additionally requires these repository
secrets:

- `DOCKERHUB_TOKEN`
- `HF_TOKEN`
- `CIVITAI_TOKEN`

The workflow runs the launcher test suite and audits every catalog custom-node
requirement before publishing.

## Credential distribution note

A credential baked into an image available to user-controlled pods must be
treated as public. The project therefore uses narrowly scoped, read-only,
disposable accounts for shared downloads. Runtime tokens are visible to the
owner of the pod and should never provide write access or access to unrelated
private resources.

## About 10sorLabs

10sorLabs builds practical ComfyUI workflows, tutorials and creator tools for
running consistent AI image and video production without depending on closed
generation platforms.

**Developed by [10sorLabs](https://10sorlabs.com).**

Explore the complete toolkit at **[10sorLabs.com](https://10sorlabs.com)**.
