const elements = {
  catalogState: document.querySelector("#catalog-state"),
  grid: document.querySelector("#workflow-grid"),
  panel: document.querySelector("#job-panel"),
  kicker: document.querySelector("#job-kicker"),
  title: document.querySelector("#job-title"),
  percent: document.querySelector("#job-percent"),
  track: document.querySelector(".progress-track"),
  fill: document.querySelector("#progress-fill"),
  message: document.querySelector("#job-message"),
  metrics: document.querySelector("#job-metrics"),
  error: document.querySelector("#job-error"),
  cancel: document.querySelector("#cancel-button"),
  comfy: document.querySelector("#comfy-button"),
};

const runningStates = new Set(["running"]);
let workflows = [];
let activeWorkflowId = null;
let pollTimer = null;

function escapeText(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** index;
  const digits = index >= 3 ? 2 : index >= 2 ? 1 : 0;
  return `${value.toFixed(digits)} ${units[index]}`;
}

function comfyUrl(serverUrl) {
  if (serverUrl) return serverUrl;
  const host = window.location.hostname;
  const proxyMatch = host.match(/^(.+)-\d+\.proxy\.runpod\.net$/);
  if (proxyMatch) {
    return `https://${proxyMatch[1]}-8188.proxy.runpod.net`;
  }
  return `${window.location.protocol}//${host}:8188`;
}

function renderWorkflows(isBusy = false) {
  elements.grid.innerHTML = workflows
    .map((workflow, index) => {
      const disabled = workflow.disabled || (isBusy && workflow.id !== activeWorkflowId);
      const selected = workflow.id === activeWorkflowId;
      return `
        <button
          class="workflow-card${selected ? " is-selected" : ""}${isBusy ? " is-busy" : ""}"
          type="button"
          data-workflow-id="${escapeText(workflow.id)}"
          ${disabled ? "disabled" : ""}
          aria-label="${escapeText(workflow.title)}"
        >
          <span class="workflow-meta">
            <span class="workflow-number">${String(index + 1).padStart(2, "0")}</span>
            <span class="workflow-badge">${escapeText(workflow.badge || "READY")}</span>
          </span>
          <h3>${escapeText(workflow.title)}</h3>
          <p class="workflow-description">${escapeText(workflow.description)}</p>
          <span class="workflow-size">${escapeText(workflow.estimated_size || "")}</span>
        </button>
      `;
    })
    .join("");

  elements.grid.querySelectorAll("[data-workflow-id]").forEach((card) => {
    card.addEventListener("click", () => startWorkflow(card.dataset.workflowId));
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status}).`);
  }
  return data;
}

async function loadCatalog() {
  try {
    const catalog = await fetchJson("/api/catalog");
    workflows = catalog.workflows || [];
    elements.catalogState.textContent = `${workflows.filter((item) => !item.disabled).length} available`;
    renderWorkflows(false);
  } catch (error) {
    elements.catalogState.textContent = "Catalog unavailable";
    elements.grid.innerHTML = `<p class="job-error">${escapeText(error.message)}</p>`;
  }
}

async function startWorkflow(workflowId) {
  try {
    activeWorkflowId = workflowId;
    renderWorkflows(true);
    const status = await fetchJson(`/api/install/${encodeURIComponent(workflowId)}`, {
      method: "POST",
    });
    updatePanel(status);
    beginPolling();
    elements.panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    renderWorkflows(false);
    showImmediateError(error.message);
  }
}

function showImmediateError(message) {
  elements.panel.hidden = false;
  elements.panel.className = "job-panel is-error";
  elements.kicker.textContent = "ERROR";
  elements.title.textContent = "Could not start workflow";
  elements.percent.textContent = "0%";
  elements.fill.style.width = "0%";
  elements.error.textContent = message;
  elements.error.hidden = false;
  elements.cancel.hidden = true;
  elements.comfy.hidden = true;
}

function updatePanel(status) {
  const percent = Math.max(0, Math.min(100, Number(status.percent || 0)));
  const isComplete = status.status === "complete";
  const isError = status.status === "error";
  const isRunning = runningStates.has(status.status);

  elements.panel.hidden = status.status === "idle";
  elements.panel.className = `job-panel${isComplete ? " is-complete" : ""}${isError ? " is-error" : ""}`;
  elements.kicker.textContent = String(status.stage || status.status).toUpperCase();
  elements.title.textContent = status.title || "Workflow setup";
  elements.percent.textContent = `${Math.round(percent)}%`;
  elements.fill.style.width = `${percent}%`;
  elements.track.setAttribute("aria-valuenow", String(Math.round(percent)));
  elements.message.textContent = status.message || "";

  const metrics = [];
  if (status.file_downloaded_bytes > 0 || status.file_total_bytes > 0) {
    const downloaded = formatBytes(status.file_downloaded_bytes);
    const total = formatBytes(status.file_total_bytes);
    metrics.push(total ? `${downloaded} / ${total}` : downloaded);
  }
  if (status.bytes_per_second > 0) {
    metrics.push(`${formatBytes(status.bytes_per_second)}/s`);
  }
  if (status.file_count > 1 && status.file_index > 0) {
    metrics.push(`File ${status.file_index} of ${status.file_count}`);
  }
  elements.metrics.textContent = metrics.filter(Boolean).join(" · ");

  elements.error.textContent = status.error || "";
  elements.error.hidden = !status.error;
  elements.cancel.hidden = !isRunning;
  elements.comfy.hidden = !isComplete;
  if (isComplete) {
    elements.comfy.href = comfyUrl(status.comfy_url);
  }

  activeWorkflowId = isRunning || isComplete ? status.workflow_id : null;
  renderWorkflows(isRunning);
}

function beginPolling() {
  window.clearInterval(pollTimer);
  pollTimer = window.setInterval(pollStatus, 500);
}

async function pollStatus() {
  try {
    const status = await fetchJson("/api/status");
    updatePanel(status);
    if (!runningStates.has(status.status)) {
      window.clearInterval(pollTimer);
    }
  } catch {
    // A short proxy interruption should not discard the visible progress.
  }
}

elements.cancel.addEventListener("click", async () => {
  elements.cancel.disabled = true;
  try {
    const status = await fetchJson("/api/cancel", { method: "POST" });
    updatePanel(status);
  } finally {
    elements.cancel.disabled = false;
  }
});

async function initialise() {
  await loadCatalog();
  try {
    const status = await fetchJson("/api/status");
    if (status.status !== "idle") {
      activeWorkflowId = status.workflow_id;
      updatePanel(status);
      if (runningStates.has(status.status)) beginPolling();
    }
  } catch {
    // Catalog errors already provide a useful first-load message.
  }
}

initialise();
