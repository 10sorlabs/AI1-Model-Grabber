// Where the RapidCache demo clip is served from. Accepts a local path or a full URL,
// so the file can move to a CDN without touching anything else. Empty disables the
// video and leaves the rest of the promo intact.
//
// Empty on purpose. At the card's 420px width the clip renders 378x213 and takes 42% of
// the card's height, and the two ComfyUI panels inside it are illegible at that size.
// The stats row carries what the clip was carrying, and the CTA opens the product page
// where it plays full size. Measured, the card is 279px tall without it - the size this
// card was asked for. The <video>, its CSS and the whole degradation path are unchanged,
// so putting the clip back is this one line.
const RAPIDCACHE_DEMO_URL = "";

// RW's call, and a considered one: the standard template is effectively free, so support
// is not part of it, while subscribers pay real money and are the more exposed customer.
// Counter-argument, recorded rather than acted on: /api/diagnostics is unauthenticated
// anyway, and a standard-tier user with a problem still ends up sending a screenshot.
// Flip to false to show it to everyone; nothing else changes.
const DEBUG_REPORT_REQUIRES_FAST = true;

const elements = {
  navLinks: [...document.querySelectorAll("[data-view-target]")],
  views: [...document.querySelectorAll("[data-view]")],
  catalogState: document.querySelector("#catalog-state"),
  grid: document.querySelector("#workflow-grid"),
  panel: document.querySelector("#job-panel"),
  kicker: document.querySelector("#job-kicker"),
  title: document.querySelector("#job-title"),
  percent: document.querySelector("#job-percent"),
  track: document.querySelector("#job-panel .progress-track"),
  fill: document.querySelector("#progress-fill"),
  message: document.querySelector("#job-message"),
  metrics: document.querySelector("#job-metrics"),
  warnings: document.querySelector("#job-warnings"),
  error: document.querySelector("#job-error"),
  cancel: document.querySelector("#cancel-button"),
  restart: document.querySelector("#restart-button"),
  comfy: document.querySelector("#comfy-button"),
  debugButton: document.querySelector("#debug-button"),
  debugReport: document.querySelector("#debug-report"),
  debugReportText: document.querySelector("#debug-report-text"),
  debugCopy: document.querySelector("#debug-copy"),
  debugDownload: document.querySelector("#debug-download"),
  modelDraftList: document.querySelector("#model-draft-list"),
  modelQueueList: document.querySelector("#model-queue-list"),
  customQueueState: document.querySelector("#custom-queue-state"),
  downloadedModelList: document.querySelector("#downloaded-model-list"),
  downloadedModelState: document.querySelector("#downloaded-model-state"),
  nodeDraftList: document.querySelector("#node-draft-list"),
  nodeQueueList: document.querySelector("#node-queue-list"),
  customNodeQueueState: document.querySelector("#custom-node-queue-state"),
  installedNodeList: document.querySelector("#installed-node-list"),
  installedNodeState: document.querySelector("#installed-node-state"),
  customNodeRestart: document.querySelector("#custom-node-restart-button"),
  customNodeRestartError: document.querySelector("#custom-node-restart-error"),
  tierBadge: document.querySelector("#tier-badge"),
  accountState: document.querySelector("#account-state"),
  accountForm: document.querySelector("#account-form"),
  accountEmail: document.querySelector("#account-email"),
  accountPassword: document.querySelector("#account-password"),
  accountError: document.querySelector("#account-error"),
  accountSummary: document.querySelector("#account-summary"),
  accountSignin: document.querySelector("#account-signin"),
  accountSignout: document.querySelector("#account-signout"),
  rapidcacheUpsell: document.querySelector("#rapidcache-upsell"),
  rapidcacheVideo: document.querySelector("#rapidcache-video"),
  rapidcacheSigninHint: document.querySelector("#rapidcache-signin-hint"),
};

// No CSS rule can stop a video from autoplaying; that has to be decided here.
const reduceMotion =
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// One list, used by both selectView and initialise. Keeping two copies is what let
// a new view silently fall through to Workflows.
const VIEW_NAMES = ["workflows", "custom-models", "custom-nodes", "account"];

const runningStates = new Set(["running"]);
let selectedView = "workflows";
let workflows = [];
let activeWorkflowId = null;
let pollTimer = null;
let customPollTimer = null;
let customNodePollTimer = null;
let comfyRestartPollTimer = null;
let modelLocations = [];
let draftSequence = 0;
let modelDrafts = [];
let nodeDraftSequence = 0;
let nodeDrafts = [];

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

function selectView(viewName, updateHash = true) {
  const allowedViews = new Set(VIEW_NAMES);
  const selected = allowedViews.has(viewName) ? viewName : "workflows";
  selectedView = selected;
  elements.views.forEach((view) => {
    view.hidden = view.dataset.view !== selected;
  });
  elements.navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.dataset.viewTarget === selected);
  });
  if (updateHash) {
    window.history.replaceState(null, "", `#${selected}`);
  }
  // The video lives in a subtree that was display:none until now.
  if (selected === "account") playRapidcacheVideo();
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
  elements.panel.dataset.stage = "error";
  elements.kicker.textContent = "ERROR";
  elements.title.textContent = "Could not start workflow";
  elements.percent.textContent = "0%";
  elements.fill.style.width = "0%";
  // Without this a failed start keeps the previous run's byte count and rate.
  elements.metrics.innerHTML = "";
  elements.warnings.textContent = "";
  elements.warnings.hidden = true;
  elements.error.textContent = message;
  elements.error.hidden = false;
  elements.cancel.hidden = true;
  elements.restart.hidden = true;
  elements.comfy.hidden = true;
}

function ratePhase(status) {
  const message = String(status.message || "");
  if (message.startsWith("Placing")) return "saving to disk";
  if (status.stage === "verifying") return "checking the file";
  if (status.stage === "downloading") return "downloading";
  return "working";
}

function updatePanel(status) {
  const percent = Math.max(0, Math.min(100, Number(status.percent || 0)));
  const isComplete = status.status === "complete";
  const isError = status.status === "error";
  const isRunning = runningStates.has(status.status);

  elements.panel.hidden = status.status === "idle";
  elements.panel.className = `job-panel${isComplete ? " is-complete" : ""}${isError ? " is-error" : ""}`;
  // Always set, never conditionally: a stale stage would keep its colour after the
  // panel moved on. className is reassigned above, so this has to be an attribute.
  elements.panel.dataset.stage = String(status.stage || status.status || "idle");
  elements.kicker.textContent = String(status.stage || status.status).toUpperCase();
  elements.title.textContent = status.title || "Workflow setup";
  elements.percent.textContent = `${Math.round(percent)}%`;
  elements.fill.style.width = `${percent}%`;
  elements.track.setAttribute("aria-valuenow", String(Math.round(percent)));
  elements.message.textContent = status.message || "";

  // Separate spans so bytes, rate and the file counter can carry different weights
  // rather than reading as one flat run of text.
  const metrics = [];
  if (status.file_downloaded_bytes > 0 || status.file_total_bytes > 0) {
    const downloaded = formatBytes(status.file_downloaded_bytes);
    const total = formatBytes(status.file_total_bytes);
    metrics.push(
      `<span class="metric-bytes">${escapeText(
        total ? `${downloaded} / ${total}` : downloaded,
      )}</span>`,
    );
  }
  if (status.bytes_per_second > 0) {
    // The phase, not just the number. Both downloading and placing render one big blue
    // rate, distinguished only by the verb in the message above, so "1.10 GB/s" straight
    // after "77.5 MB/s" reads as a wild swing. It is not: one is the pod's internet
    // connection and the other is its local disk. This has misled the author of this
    // codebase more than once, so it will certainly mislead a customer.
    //
    // Derived from what the panel already carries rather than a new status field. The
    // placement reports stage "installing" with a "Placing …" message, which is why the
    // message is consulted and not only the stage.
    metrics.push(
      `<span class="metric-rate">${escapeText(
        `${ratePhase(status)} · ${formatBytes(status.bytes_per_second)}/s`,
      )}</span>`,
    );
  }
  if (status.file_count > 1 && status.file_index > 0) {
    metrics.push(
      `<span class="metric-files">${escapeText(
        `File ${status.file_index} of ${status.file_count}`,
      )}</span>`,
    );
  }
  elements.metrics.innerHTML = metrics.join("");

  const warnings = Array.isArray(status.warnings) ? status.warnings : [];
  elements.warnings.innerHTML = warnings.length
    ? `<strong>Skipped items</strong><ul>${warnings
        .map((warning) => `<li>${escapeText(warning)}</li>`)
        .join("")}</ul>`
    : "";
  elements.warnings.hidden = warnings.length === 0;
  elements.error.textContent = status.error || "";
  elements.error.hidden = !status.error;
  elements.cancel.hidden = !isRunning;
  elements.restart.hidden = !(isComplete && status.restart_required);
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

function restartButtons() {
  return [elements.restart, elements.customNodeRestart];
}

function setRestartButtonsBusy(isBusy) {
  restartButtons().forEach((button) => {
    button.disabled = isBusy;
    button.textContent = isBusy ? "RESTARTING…" : "RESTART COMFYUI";
  });
}

async function refreshAfterComfyRestart() {
  try {
    const status = await fetchJson("/api/status");
    if (status.status !== "idle") updatePanel(status);
  } catch {
    // The workflow status is optional on the custom-nodes page.
  }
  try {
    await loadCustomNodes();
  } catch {
    // Preserve the existing custom-node display during a short proxy interruption.
  }
}

async function pollComfyRestart() {
  try {
    const state = await fetchJson("/api/comfy-restart");
    if (state.status === "restarting") return;
    window.clearInterval(comfyRestartPollTimer);
    setRestartButtonsBusy(false);
    if (state.status === "ready") {
      elements.error.hidden = true;
      elements.customNodeRestartError.hidden = true;
      await refreshAfterComfyRestart();
      return;
    }
    if (state.status === "error") {
      const message = state.error || "ComfyUI restart failed.";
      elements.error.textContent = message;
      elements.error.hidden = elements.panel.hidden;
      elements.customNodeRestartError.textContent = message;
      elements.customNodeRestartError.hidden = false;
    }
  } catch {
    // A short interruption is expected while ComfyUI is restarting.
  }
}

function beginComfyRestartPolling() {
  window.clearInterval(comfyRestartPollTimer);
  comfyRestartPollTimer = window.setInterval(pollComfyRestart, 1000);
}

async function restartComfyUI() {
  setRestartButtonsBusy(true);
  elements.error.hidden = true;
  elements.customNodeRestartError.hidden = true;
  try {
    await fetchJson("/api/comfy-restart", { method: "POST" });
    beginComfyRestartPolling();
  } catch (error) {
    setRestartButtonsBusy(false);
    elements.error.textContent = error.message;
    elements.error.hidden = elements.panel.hidden;
    elements.customNodeRestartError.textContent = error.message;
    elements.customNodeRestartError.hidden = false;
  }
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

function createDraft() {
  draftSequence += 1;
  return {
    id: `draft-${draftSequence}`,
    url: "",
    location: "",
    customFolder: "",
  };
}

function locationLabel(location) {
  return location
    .split("/")
    .map((part) => part.replaceAll("_", " "))
    .join(" / ");
}

function locationOptions(selected) {
  const standard = modelLocations
    .map((location) => `
      <option value="${escapeText(location)}" ${selected === location ? "selected" : ""}>
        ${escapeText(locationLabel(location))}
      </option>
    `)
    .join("");
  return `
    <option value="">Model location</option>
    ${standard}
    <option value="__custom__" ${selected === "__custom__" ? "selected" : ""}>+ Create custom folder</option>
  `;
}

function resolvedDraftLocation(draft) {
  return draft.location === "__custom__" ? draft.customFolder.trim() : draft.location;
}

function ensureTrailingDraft() {
  const last = modelDrafts.at(-1);
  if (!last || last.url || last.location || last.customFolder) {
    const draft = createDraft();
    modelDrafts.push(draft);
    elements.modelDraftList.append(createDraftRow(draft));
  }
}

function createDraftRow(draft) {
  const row = document.createElement("div");
  row.className = "model-draft-row";
  row.dataset.draftId = draft.id;
  row.innerHTML = `
    <label class="model-field">
      <span class="sr-only">Model link</span>
      <input class="model-input model-url-input" type="url" placeholder="Model link" autocomplete="off" />
    </label>
    <label class="model-field">
      <span class="sr-only">Model location</span>
      <select class="model-select model-location-select">${locationOptions(draft.location)}</select>
      <input
        class="model-input custom-folder-input"
        type="text"
        placeholder="Custom folder, e.g. sams"
        autocomplete="off"
        ${draft.location === "__custom__" ? "" : "hidden"}
      />
    </label>
    <button class="button button-primary download-model-button" type="button" disabled>DOWNLOAD</button>
    <p class="draft-error" hidden></p>
  `;

  const urlInput = row.querySelector(".model-url-input");
  const locationSelect = row.querySelector(".model-location-select");
  const customFolderInput = row.querySelector(".custom-folder-input");
  const downloadButton = row.querySelector(".download-model-button");
  const error = row.querySelector(".draft-error");
  urlInput.value = draft.url;
  customFolderInput.value = draft.customFolder;

  function syncDraft() {
    draft.url = urlInput.value.trim();
    draft.location = locationSelect.value;
    draft.customFolder = customFolderInput.value;
    customFolderInput.hidden = draft.location !== "__custom__";
    downloadButton.disabled = !(draft.url && resolvedDraftLocation(draft));
    error.hidden = true;
    ensureTrailingDraft();
  }

  urlInput.addEventListener("input", syncDraft);
  locationSelect.addEventListener("change", () => {
    syncDraft();
    if (draft.location === "__custom__") customFolderInput.focus();
  });
  customFolderInput.addEventListener("input", syncDraft);
  downloadButton.addEventListener("click", async () => {
    downloadButton.disabled = true;
    error.hidden = true;
    try {
      await fetchJson("/api/custom-models", {
        method: "POST",
        body: JSON.stringify({
          url: draft.url,
          location: resolvedDraftLocation(draft),
        }),
      });
      modelDrafts = modelDrafts.filter((item) => item.id !== draft.id);
      row.remove();
      ensureTrailingDraft();
      await loadCustomModels();
      beginCustomPolling();
    } catch (requestError) {
      error.textContent = requestError.message;
      error.hidden = false;
      downloadButton.disabled = false;
    }
  });

  return row;
}

function renderDrafts() {
  elements.modelDraftList.innerHTML = "";
  if (!modelDrafts.length) modelDrafts.push(createDraft());
  modelDrafts.forEach((draft) => elements.modelDraftList.append(createDraftRow(draft)));
  ensureTrailingDraft();
}

function modelStatusLabel(status) {
  if (status === "skipped") return "FOUND";
  return String(status || "queued").toUpperCase();
}

function modelJobMarkup(item) {
  const percent = Math.max(0, Math.min(100, Number(item.percent || 0)));
  const metrics = [];
  if (item.downloaded_bytes || item.total_bytes) {
    const downloaded = formatBytes(Number(item.downloaded_bytes || 0));
    const total = formatBytes(Number(item.total_bytes || 0));
    metrics.push(total ? `${downloaded} / ${total}` : downloaded);
  }
  if (item.bytes_per_second > 0) {
    metrics.push(`${formatBytes(Number(item.bytes_per_second))}/s`);
  }
  const path = `models/${item.location}/${item.filename}`;
  return `
    <article class="model-job">
      <div class="model-job-topline">
        <div class="model-job-name">
          <strong title="${escapeText(item.filename)}">${escapeText(item.filename)}</strong>
          <span title="${escapeText(path)}">${escapeText(path)} · ${escapeText(item.source_host)}</span>
        </div>
        <span class="model-status is-${escapeText(item.status)}">${escapeText(modelStatusLabel(item.status))}</span>
      </div>
      <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(percent)}">
        <div class="progress-fill" style="width: ${percent}%"></div>
      </div>
      <div class="model-job-details">
        <p>${escapeText(item.message || "")}</p>
        <p>${escapeText(metrics.join(" · "))}</p>
      </div>
      ${item.error ? `<p class="model-job-error">${escapeText(item.error)}</p>` : ""}
    </article>
  `;
}

function renderCustomModels(state) {
  modelLocations = state.locations || [];
  const queue = state.queue || [];
  const downloaded = state.downloaded || [];
  const activeCount = queue.filter((item) => ["queued", "downloading"].includes(item.status)).length;

  elements.customQueueState.textContent = `${activeCount} queued`;
  elements.modelQueueList.innerHTML = queue.map(modelJobMarkup).join("");
  elements.downloadedModelState.textContent = `${downloaded.length} ${downloaded.length === 1 ? "model" : "models"}`;
  elements.downloadedModelList.innerHTML = downloaded.length
    ? downloaded.map(modelJobMarkup).join("")
    : '<p class="empty-state">Completed downloads will appear here.</p>';

  return activeCount > 0;
}

async function loadCustomModels() {
  const state = await fetchJson("/api/custom-models");
  const hasActiveDownloads = renderCustomModels(state);
  if (!modelDrafts.length) renderDrafts();
  return hasActiveDownloads;
}

function beginCustomPolling() {
  window.clearInterval(customPollTimer);
  customPollTimer = window.setInterval(async () => {
    try {
      const hasActiveDownloads = await loadCustomModels();
      if (!hasActiveDownloads) window.clearInterval(customPollTimer);
    } catch {
      // Preserve the current queue display during a short proxy interruption.
    }
  }, 700);
}

function createNodeDraft() {
  nodeDraftSequence += 1;
  return {
    id: `node-draft-${nodeDraftSequence}`,
    url: "",
  };
}

function ensureTrailingNodeDraft() {
  const last = nodeDrafts.at(-1);
  if (!last || last.url) {
    const draft = createNodeDraft();
    nodeDrafts.push(draft);
    elements.nodeDraftList.append(createNodeDraftRow(draft));
  }
}

function createNodeDraftRow(draft) {
  const row = document.createElement("div");
  row.className = "node-draft-row";
  row.dataset.nodeDraftId = draft.id;
  row.innerHTML = `
    <label class="model-field">
      <span class="sr-only">GitHub repository link</span>
      <input
        class="model-input node-url-input"
        type="url"
        placeholder="GitHub repository link"
        autocomplete="off"
      />
    </label>
    <button class="button button-primary download-model-button" type="button" disabled>INSTALL</button>
    <p class="draft-error" hidden></p>
  `;

  const urlInput = row.querySelector(".node-url-input");
  const installButton = row.querySelector(".download-model-button");
  const error = row.querySelector(".draft-error");
  urlInput.value = draft.url;

  urlInput.addEventListener("input", () => {
    draft.url = urlInput.value.trim();
    installButton.disabled = !draft.url;
    error.hidden = true;
    ensureTrailingNodeDraft();
  });

  installButton.addEventListener("click", async () => {
    installButton.disabled = true;
    error.hidden = true;
    try {
      await fetchJson("/api/custom-nodes", {
        method: "POST",
        body: JSON.stringify({url: draft.url}),
      });
      nodeDrafts = nodeDrafts.filter((item) => item.id !== draft.id);
      row.remove();
      ensureTrailingNodeDraft();
      await loadCustomNodes();
      beginCustomNodePolling();
    } catch (requestError) {
      error.textContent = requestError.message;
      error.hidden = false;
      installButton.disabled = false;
    }
  });

  return row;
}

function renderNodeDrafts() {
  elements.nodeDraftList.innerHTML = "";
  if (!nodeDrafts.length) nodeDrafts.push(createNodeDraft());
  nodeDrafts.forEach((draft) => elements.nodeDraftList.append(createNodeDraftRow(draft)));
  ensureTrailingNodeDraft();
}

function customNodeMarkup(item) {
  const percent = Math.max(0, Math.min(100, Number(item.percent || 0)));
  const path = `custom_nodes/${item.name}`;
  return `
    <article class="model-job">
      <div class="model-job-topline">
        <div class="model-job-name">
          <strong title="${escapeText(item.name)}">${escapeText(item.name)}</strong>
          <span title="${escapeText(path)}">${escapeText(path)} · ${escapeText(item.source_host)}</span>
        </div>
        <span class="model-status is-${escapeText(item.status)}">${escapeText(modelStatusLabel(item.status))}</span>
      </div>
      <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(percent)}">
        <div class="progress-fill" style="width: ${percent}%"></div>
      </div>
      <div class="model-job-details">
        <p>${escapeText(item.message || "")}</p>
        <p>${item.restart_required ? "RESTART REQUIRED" : ""}</p>
      </div>
      ${item.error ? `<p class="model-job-error">${escapeText(item.error)}</p>` : ""}
    </article>
  `;
}

function renderCustomNodes(state) {
  const queue = state.queue || [];
  const installed = state.downloaded || [];
  const activeCount = queue.filter((item) => ["queued", "cloning", "installing"].includes(item.status)).length;

  elements.customNodeQueueState.textContent = `${activeCount} queued`;
  elements.nodeQueueList.innerHTML = queue.map(customNodeMarkup).join("");
  elements.installedNodeState.textContent = `${installed.length} ${installed.length === 1 ? "node" : "nodes"}`;
  elements.installedNodeList.innerHTML = installed.length
    ? installed.map(customNodeMarkup).join("")
    : '<p class="empty-state">Completed custom nodes will appear here.</p>';
  const needsRestart = installed.some((item) => item.restart_required);
  elements.customNodeRestart.hidden = !needsRestart || activeCount > 0;
  return activeCount > 0;
}

async function loadCustomNodes() {
  const state = await fetchJson("/api/custom-nodes");
  const hasActiveInstalls = renderCustomNodes(state);
  if (!nodeDrafts.length) renderNodeDrafts();
  return hasActiveInstalls;
}

function beginCustomNodePolling() {
  window.clearInterval(customNodePollTimer);
  customNodePollTimer = window.setInterval(async () => {
    try {
      const hasActiveInstalls = await loadCustomNodes();
      if (!hasActiveInstalls) window.clearInterval(customNodePollTimer);
    } catch {
      // Preserve the current queue display during a short proxy interruption.
    }
  }, 700);
}

function formatRenewal(value) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

// Deliberately declared above the account renderer: a test slices this file between
// that function and the loader below it and asserts on what falls inside, so nothing
// new may sit between the two. Do not quote either name literally here - the slice is
// found by string search and would land in this comment.
function setUpRapidcacheVideo() {
  const video = elements.rapidcacheVideo;
  if (!video) return;
  if (!RAPIDCACHE_DEMO_URL) {
    // No clip configured. Hide the element rather than render an empty box.
    video.hidden = true;
    return;
  }
  // A 404 or a decode failure hides the video and leaves the promo intact. The
  // listener goes on the element itself: src on a <source> child fires an error
  // that does not bubble, so it would never arrive here.
  video.addEventListener(
    "error",
    () => {
      video.hidden = true;
    },
    { once: true },
  );
  // Some browsers check the property, not just the attribute, before allowing
  // autoplay.
  video.muted = true;
  video.src = RAPIDCACHE_DEMO_URL;
  video.hidden = false;
}

function playRapidcacheVideo() {
  const video = elements.rapidcacheVideo;
  if (!video || video.hidden || reduceMotion) return;
  // Browsers do not reliably start a video inside a display:none subtree, so this is
  // attempted both when the view is shown and after the panel is unhidden. A rejected
  // play promise is an unhandled rejection unless it is caught.
  const started = video.play();
  if (started && typeof started.catch === "function") started.catch(() => {});
}

function renderAccount(account) {
  const status = account.status || {};
  const source = account.source || "none";
  const service = account.service || "ok";
  const fromTemplate = source === "env";

  // An allowlist, not a fallback. Adding a tier name to the API contract means adding
  // it here too, or the new tier silently reads as no badge at all.
  const tier = String(status.tier || "").toLowerCase();
  const knownTier = tier === "fast" || tier === "standard";

  // "Checking" only while a retry really is coming. If the service answered and named
  // a tier we do not know, nothing is in flight and the label would sit there forever.
  const showChecking = service === "unavailable";
  const showTier = service === "ok" && knownTier;

  // A label only. It never disables, hides or alters any other control.
  const isFastTier = showTier && tier === "fast";
  elements.tierBadge.textContent = showTier
    ? isFastTier
      ? "RapidCache"
      : "Standard downloads"
    : "Checking subscription…";
  elements.tierBadge.hidden = !account.configured || !(showChecking || showTier);
  // A pill on fast tier only. The pill's presence is the signal, so giving standard one
  // too - in any colour - would destroy it.
  elements.tierBadge.classList.toggle("tier-pill", isFastTier);

  // Same condition as the pill and the upsell below, so the three cannot drift apart.
  elements.debugButton.hidden = DEBUG_REPORT_REQUIRES_FAST && !isFastTier;

  // Never advertise a product on a pod that has no way to redeem it, never advertise
  // to someone who already pays, and never advertise when we cannot rule that out.
  // Showing a subscriber an advert for what they already bought is worse than showing
  // nobody anything.
  //   unconfigured -> no account service at all, so signing in is impossible here
  //   fast         -> they already have it
  //   unavailable  -> the status check failed; they may well be paying
  // This sits before the four early returns below, not after them: three of the four
  // states never reach the end of this function.
  elements.rapidcacheUpsell.hidden =
    service === "unconfigured" ||
    (showTier && tier === "fast") ||
    (account.configured && service === "unavailable");
  if (!elements.rapidcacheUpsell.hidden) {
    // The template-key branch hides the sign-in form, so pointing at it would be
    // pointing at nothing.
    elements.rapidcacheSigninHint.hidden = fromTemplate;
    if (selectedView === "account") playRapidcacheVideo();
  }

  if (fromTemplate) {
    elements.accountState.textContent = "Licence key set on this pod's template";
    elements.accountForm.hidden = true;
    elements.accountSignin.hidden = true;
    elements.accountSignout.hidden = true;
    elements.accountSummary.textContent =
      "Remove the LCT_LICENSE_KEY template variable to sign in here instead.";
    elements.accountSummary.hidden = false;
    return;
  }

  if (account.configured && service === "unauthenticated") {
    // The credential was rejected, not lost. Offer the one action that helps, and
    // keep Sign out so the dead token can be cleared rather than only overwritten.
    elements.accountState.textContent = "Signed out — sign in again";
    elements.accountForm.hidden = false;
    elements.accountSignin.hidden = false;
    elements.accountSignout.hidden = false;
    elements.accountSummary.textContent =
      "This licence key is no longer accepted. Sign in again to restore fast downloads.";
    elements.accountSummary.hidden = false;
    return;
  }

  if (account.configured) {
    elements.accountForm.hidden = true;
    elements.accountSignout.hidden = false;
    elements.accountSummary.hidden = false;
    if (service === "unavailable") {
      elements.accountState.textContent = "Account service unavailable";
      elements.accountSummary.textContent =
        "Your subscription could not be checked. Downloads will use the standard route until it can.";
      return;
    }
    const parts = [];
    if (status.email) parts.push(String(status.email));
    const renewal = formatRenewal(status.expires_at);
    if (renewal) parts.push(`Renews ${renewal}`);
    elements.accountState.textContent = "Signed in";
    elements.accountSummary.textContent = parts.join(" · ") || "Signed in on this pod.";
    return;
  }

  elements.accountState.textContent = "Not signed in";
  elements.accountForm.hidden = false;
  elements.accountSignin.hidden = false;
  elements.accountSignout.hidden = true;
  elements.accountSummary.hidden = true;
}

async function loadAccount() {
  renderAccount(await fetchJson("/api/account"));
}

async function signIn() {
  elements.accountSignin.disabled = true;
  elements.accountError.hidden = true;
  try {
    const account = await fetchJson("/api/account/login", {
      method: "POST",
      body: JSON.stringify({
        email: elements.accountEmail.value.trim(),
        password: elements.accountPassword.value,
      }),
    });
    renderAccount(account);
    await loadCatalog();
  } catch (error) {
    elements.accountError.textContent = error.message;
    elements.accountError.hidden = false;
  } finally {
    // Cleared on both paths; the value never outlives the request.
    elements.accountPassword.value = "";
    elements.accountSignin.disabled = false;
  }
}

async function signOut() {
  elements.accountSignout.disabled = true;
  elements.accountError.hidden = true;
  try {
    renderAccount(await fetchJson("/api/account/logout", { method: "POST" }));
    await loadCatalog();
  } catch (error) {
    elements.accountError.textContent = error.message;
    elements.accountError.hidden = false;
  } finally {
    elements.accountSignout.disabled = false;
  }
}

elements.navLinks.forEach((link) => {
  link.addEventListener("click", () => selectView(link.dataset.viewTarget));
});

elements.accountSignin.addEventListener("click", signIn);
elements.accountSignout.addEventListener("click", signOut);
elements.accountPassword.addEventListener("keydown", (event) => {
  // The button records the in-flight state; holding Enter would otherwise fire
  // concurrent logins and burn the account service's rate limit.
  if (event.key === "Enter" && !elements.accountSignin.disabled) signIn();
});

elements.cancel.addEventListener("click", async () => {
  elements.cancel.disabled = true;
  try {
    const status = await fetchJson("/api/cancel", { method: "POST" });
    updatePanel(status);
  } finally {
    elements.cancel.disabled = false;
  }
});

elements.restart.addEventListener("click", restartComfyUI);

elements.debugButton.addEventListener("click", async () => {
  elements.debugButton.disabled = true;
  try {
    const response = await fetch("/api/diagnostics/report");
    elements.debugReportText.textContent = await response.text();
  } catch (error) {
    elements.debugReportText.textContent = `Could not read the report: ${error}`;
  }
  elements.debugReport.hidden = false;
  elements.debugButton.disabled = false;
});

// navigator, Blob and URL are all looked up inside the handlers and never at module
// scope. The test sandbox provides none of them, so a top-level reference would throw on
// load and take every harness test with it - and navigator.clipboard is also undefined on
// a plain http:// origin, which is not hypothetical: port-forwarding to
// http://localhost:3000 is how you would try this locally.
elements.debugCopy.addEventListener("click", async () => {
  const text = elements.debugReportText.textContent || "";
  const clipboard =
    typeof navigator !== "undefined" && navigator.clipboard
      ? navigator.clipboard
      : null;
  if (clipboard) {
    try {
      await clipboard.writeText(text);
      elements.debugCopy.textContent = "Copied";
      setTimeout(() => (elements.debugCopy.textContent = "Copy"), 2000);
      return;
    } catch (error) {
      // Falls through to selecting the text, below.
    }
  }
  // No clipboard API, or it refused. Select the report so Ctrl-C still works, which is
  // the whole job - never throw and leave the button looking broken.
  if (typeof window !== "undefined" && window.getSelection && document.createRange) {
    const range = document.createRange();
    range.selectNodeContents(elements.debugReportText);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  elements.debugCopy.textContent = "Press Ctrl-C";
  setTimeout(() => (elements.debugCopy.textContent = "Copy"), 4000);
});

elements.debugDownload.addEventListener("click", () => {
  const text = elements.debugReportText.textContent || "";
  if (typeof Blob === "undefined" || typeof URL === "undefined" || !URL.createObjectURL) {
    // Nothing to fall back to that is better than what is already on screen.
    elements.debugDownload.textContent = "Select and copy instead";
    setTimeout(() => (elements.debugDownload.textContent = "Download"), 4000);
    return;
  }
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "10sorlabs-install-report.txt";
  link.click();
  URL.revokeObjectURL(url);
});
elements.customNodeRestart.addEventListener("click", restartComfyUI);

async function initialise() {
  const requestedView = window.location.hash.replace("#", "");
  const initialView = VIEW_NAMES.includes(requestedView) ? requestedView : "workflows";
  setUpRapidcacheVideo();
  selectView(initialView, false);
  await loadCatalog();
  try {
    await loadAccount();
  } catch {
    elements.accountState.textContent = "Account service unavailable";
  }
  try {
    if (await loadCustomModels()) beginCustomPolling();
  } catch {
    renderDrafts();
    elements.customQueueState.textContent = "Queue unavailable";
  }
  try {
    if (await loadCustomNodes()) beginCustomNodePolling();
  } catch {
    renderNodeDrafts();
    elements.customNodeQueueState.textContent = "Queue unavailable";
  }

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

  try {
    const restartState = await fetchJson("/api/comfy-restart");
    if (restartState.status === "restarting") {
      setRestartButtonsBusy(true);
      beginComfyRestartPolling();
    }
  } catch {
    // Restart controls remain available when the status endpoint is briefly unavailable.
  }

}

initialise();
