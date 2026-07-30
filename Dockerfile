# syntax=docker/dockerfile:1.7

ARG RUNPOD_COMFY_IMAGE=runpod/comfyui@sha256:7078f94dbe28d079c487c245dc3524443e2c6225a6208a1fff8c7a652c1b3a40
FROM ${RUNPOD_COMFY_IMAGE}

LABEL org.opencontainers.image.title="10sorLabs ComfyUI Workflow Launcher" \
      org.opencontainers.image.description="Stock RunPod ComfyUI plus the remotely updateable 10sorLabs Model Grabber" \
      org.opencontainers.image.source="https://github.com/10sorlabs/AI1-Model-Grabber" \
      org.opencontainers.image.version="1.0"

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LAUNCHER_PORT=3000 \
    LAUNCHER_AUTO_UPDATE=1 \
    LAUNCHER_GITHUB_REPO=10sorlabs/AI1-Model-Grabber \
    LAUNCHER_GITHUB_REF=main \
    COMFYUI_DIR=/workspace/runpod-slim/ComfyUI

WORKDIR /opt/10sorlabs

COPY requirements-launcher.txt /tmp/requirements-launcher.txt
RUN python3.12 -m pip install \
      --break-system-packages \
      --no-cache-dir \
      -r /tmp/requirements-launcher.txt \
    && rm /tmp/requirements-launcher.txt \
    && mv /start.sh /usr/local/bin/runpod-base-start.sh \
    && chmod +x /usr/local/bin/runpod-base-start.sh

COPY launcher/ /opt/10sorlabs/launcher/
COPY catalog/ /opt/10sorlabs/catalog/
COPY docker/entrypoint.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3000 8188 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:3000/api/health || exit 1

ENTRYPOINT ["/start.sh"]

