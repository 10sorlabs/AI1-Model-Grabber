# syntax=docker/dockerfile:1.7

ARG RUNPOD_COMFY_IMAGE=runpod/comfyui@sha256:7078f94dbe28d079c487c245dc3524443e2c6225a6208a1fff8c7a652c1b3a40
FROM ${RUNPOD_COMFY_IMAGE}

ARG IMAGE_VERSION=dev
ARG HF_TOKEN_REVISION=local
ARG REQUIRE_HF_DOWNLOAD_TOKEN=0

LABEL org.opencontainers.image.title="10sorLabs ComfyUI Workflow Launcher" \
      org.opencontainers.image.description="Stock RunPod ComfyUI plus the remotely updateable 10sorLabs Model Grabber" \
      org.opencontainers.image.source="https://github.com/10sorlabs/AI1-Model-Grabber" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

USER root
# LCT_API_BASE is what makes the image work with no configuration. remote._api_base()
# reads it at call time and returns "" when it is unset, which is why every pod so far
# reports service: "unconfigured" and offers no way to sign in. It stays an ENV rather
# than a default baked into the code so a customer can still point a pod somewhere else.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LAUNCHER_PORT=3000 \
    LAUNCHER_AUTO_UPDATE=1 \
    LAUNCHER_GITHUB_REPO=10sorlabs/AI1-Model-Grabber \
    LAUNCHER_GITHUB_REF=main \
    LCT_API_BASE=https://rapidcache.10sorlabs.com \
    COMFYUI_DIR=/workspace/runpod-slim/ComfyUI \
    HF_TOKEN_FILE=/opt/10sorlabs/secrets/hf_token

WORKDIR /opt/10sorlabs

RUN apt-get update \
 && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-launcher.txt /tmp/requirements-launcher.txt
RUN python3.12 -m pip install \
      --break-system-packages \
      --no-cache-dir \
      -r /tmp/requirements-launcher.txt \
    && rm /tmp/requirements-launcher.txt \
    && mv /start.sh /usr/local/bin/runpod-base-start.sh \
    && chmod +x /usr/local/bin/runpod-base-start.sh

# Pre-installed so a pod does not spend roughly three minutes in pip on every boot.
# This is a cache warm and nothing else: the launcher still installs each node pack's
# own requirements on the pod at runtime, so no behaviour depends on a package being
# present here, and anything that would not install cleanly was dropped rather than
# forced. See the header of custom-node-requirements.txt for what was left out and why.
#
# The constraints file is the load-bearing part. Several of those requirements pin numpy
# and two ask for torch unpinned; resolving either here would pull a CPU torch over the
# base image's 2.10.0+cu128, and ComfyUI would then fail to start on every pod built
# from this image. So: capture what the base image already has, install against it, and
# refuse to produce an image where torch moved. The expected version is read back out of
# the constraints file rather than written here, so the check cannot drift from the pin.
#
# Placed above the COPY of launcher/ and catalog/ deliberately - those change on almost
# every commit, and this layer is the expensive one to rebuild.
COPY docker/custom-node-requirements.txt /tmp/custom-node-requirements.txt
# Deliberately not a RUN heredoc. A heredoc body is passed to the shell byte for byte,
# so on a Windows working copy - where this file is CRLF even though the index is LF -
# the shell receives "set -eu\r" and dies. The classic continuation form is normalised
# by the Dockerfile parser and builds the same on both.
RUN set -eu; \
    python3.12 -m pip freeze \
      | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|pillow|opencv-[a-z-]+)==' \
      > /tmp/constraints.txt; \
    echo "Holding the base image at:"; \
    sed 's/^/  /' /tmp/constraints.txt; \
    python3.12 -m pip install \
      --break-system-packages \
      --no-cache-dir \
      -c /tmp/constraints.txt \
      -r /tmp/custom-node-requirements.txt; \
    expected="$(sed -n 's/^[Tt]orch==//p' /tmp/constraints.txt)"; \
    if [ -z "$expected" ]; then \
      echo "no torch pin was captured - the constraints file is not doing its job"; \
      exit 1; \
    fi; \
    actual="$(python3.12 -c 'import torch; print(torch.__version__)')"; \
    if [ "$actual" != "$expected" ]; then \
      echo "torch changed during the custom-node install: $actual != $expected"; \
      exit 1; \
    fi; \
    echo "torch intact: $actual"; \
    rm /tmp/custom-node-requirements.txt /tmp/constraints.txt

COPY launcher/ /opt/10sorlabs/launcher/
COPY catalog/ /opt/10sorlabs/catalog/
COPY docker/entrypoint.sh /start.sh
# The baked Hugging Face token stays, and removing it is not the hygiene win it looks
# like. Four files in catalog/workflows.json carry auth: "huggingface" -
# flux-2-klein-9b-fp8.safetensors and qwen_3_8b_fp8mixed.safetensors, in the
# dataset-generator and image-edit workflows. buildCatalog on the RapidCache server only
# substitutes R2 URLs when tier === "fast", so a standard-tier pod receives those
# HuggingFace URLs verbatim and app.py:519 answers them with the token from this file.
# Dropping it would break two of six workflows for every non-subscriber. Rotation is a
# separate decision, not a side effect of tidying this file.
RUN --mount=type=secret,id=hf_download_token \
    set -eu; \
    chmod +x /start.sh; \
    install -d -m 0700 /opt/10sorlabs/secrets; \
    if [ -s /run/secrets/hf_download_token ]; then \
      grep -q '^hf_' /run/secrets/hf_download_token; \
      install -m 0400 \
        /run/secrets/hf_download_token \
        /opt/10sorlabs/secrets/hf_token; \
    fi; \
    if [ "${REQUIRE_HF_DOWNLOAD_TOKEN}" = "1" ] \
       && [ ! -s /opt/10sorlabs/secrets/hf_token ]; then \
      echo "Required Hugging Face download credential was not supplied."; \
      exit 1; \
    fi; \
    printf '%s' "${HF_TOKEN_REVISION}" > /opt/10sorlabs/secrets/.revision

WORKDIR /workspace/runpod-slim

EXPOSE 3000 8188 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:3000/api/health || exit 1

ENTRYPOINT ["/start.sh"]
