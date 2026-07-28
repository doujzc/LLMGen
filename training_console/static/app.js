const state = {
  health: null,
  schema: null,
  profiles: [],
  validation: null,
  validationErrors: [],
  activeStage: "retrieval",
  onlyOverrides: false,
  compareDefaults: false,
  validateTimer: null,
  busy: false,
  currentRun: null,
  loadedProfileId: "",
  loadedVersion: null,
  draft: {
    profileId: "clawhub-full-4gpu",
    dataset: "clawhub",
    command: "full",
    overrides: {},
    notes: "",
    dirty: true,
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw Object.assign(
      new Error(`服务返回了无效响应（HTTP ${response.status}）`),
      { status: response.status, payload: {} },
    );
  }
  if (!response.ok) {
    throw Object.assign(
      new Error(payload.error || `HTTP ${response.status}`),
      { status: response.status, payload },
    );
  }
  return payload;
}

function postJson(url, payload) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 3400);
}

function setBusy(busy) {
  state.busy = busy;
  [
    "#save-profile-button",
    "#submit-run-button",
    "#export-env-button",
    "#new-profile-button",
  ].forEach((selector) => {
    $(selector).disabled = busy;
  });
  $("#submit-run-button").textContent = busy
    ? "正在处理…"
    : "保存新版本并提交独立任务";
}

function fieldByKey(key) {
  return state.schema?.fields.find((field) => field.key === key);
}

function effectiveValue(key) {
  if (key === "DATASET") return state.draft.dataset;
  if (key === "PIPELINE_COMMAND") return state.draft.command;
  if (Object.hasOwn(state.draft.overrides, key)) {
    return state.draft.overrides[key];
  }
  return (
    state.validation?.resolved?.[key] ??
    state.schema?.defaults?.[key] ??
    ""
  );
}

function defaultValue(key) {
  if (key === "DATASET") return state.schema?.dataset || "clawhub";
  if (key === "PIPELINE_COMMAND") return "full";
  return state.schema?.defaults?.[key] ?? "";
}

function isOverridden(key) {
  if (key === "DATASET") return state.draft.dataset !== state.schema?.dataset;
  if (key === "PIPELINE_COMMAND") return state.draft.command !== "full";
  return Object.hasOwn(state.draft.overrides, key);
}

function validationPayload() {
  return {
    dataset: state.draft.dataset,
    command: state.draft.command,
    overrides: state.draft.overrides,
  };
}

function renderHeaderMetrics() {
  const configuredGpus =
    state.validation?.contract?.num_gpus ||
    state.schema?.defaults?.ROUTER_NUM_GPUS ||
    "—";
  $("#metric-gpus").textContent =
    state.health?.gpu_count === null || state.health?.gpu_count === undefined
      ? configuredGpus
      : `${state.health.gpu_count} / ${configuredGpus}`;
  $("#metric-profile").textContent = state.draft.profileId || "草稿";
  $("#metric-version").textContent = state.loadedVersion
    ? `v${state.loadedVersion}`
    : "未保存";
  $("#metric-levels").textContent =
    state.validation?.resolved?.NUM_LEVELS ||
    state.schema?.defaults?.NUM_LEVELS ||
    "—";
}

function renderProfiles() {
  const query = $("#profile-search").value.trim().toLocaleLowerCase();
  const container = $("#profile-list");
  container.replaceChildren();
  const matches = state.profiles.filter((profile) =>
    profile.profile_id.toLocaleLowerCase().includes(query),
  );

  if (!matches.length) {
    const empty = element("div", "profile-empty");
    empty.textContent = state.profiles.length
      ? "没有匹配的配置。"
      : "尚未保存配置。当前表单是一个草稿，保存后会在这里形成不可变版本。";
    container.append(empty);
    return;
  }

  matches.forEach((profile, profileIndex) => {
    const details = document.createElement("details");
    details.className = "profile-family";
    details.open =
      profile.profile_id === state.loadedProfileId ||
      (!state.loadedProfileId && profileIndex === 0);
    const summary = document.createElement("summary");
    summary.append(
      element("span", "", profile.profile_id),
      element("small", "", `${profile.versions.length} 个版本`),
    );
    details.append(summary);

    profile.versions.forEach((version) => {
      const button = element("button", "profile-version");
      button.type = "button";
      button.classList.toggle(
        "selected",
        profile.profile_id === state.loadedProfileId &&
          version.version === state.loadedVersion,
      );
      button.append(
        element("strong", "", `v${version.version}`),
        element(
          "span",
          "",
          version.version === profile.latest_version ? "当前版本" : "历史只读",
        ),
        element("time", "", formatTime(version.created_at)),
      );
      button.addEventListener("click", () =>
        loadProfile(profile.profile_id, version.version),
      );
      details.append(button);
    });
    container.append(details);
  });
}

function stageConfigured(stageId) {
  if (!state.validation?.resolved || !state.schema) return false;
  const fields = state.schema.fields.filter((field) => field.stage === stageId);
  return fields.every((field) => {
    if (!field.required) return true;
    return String(state.validation.resolved[field.key] || "").trim();
  });
}

function renderStages() {
  const container = $("#stage-list");
  container.replaceChildren();
  (state.schema?.stages || []).forEach((stage) => {
    const button = element("button", "stage-button");
    button.type = "button";
    button.dataset.stage = stage.id;
    button.classList.toggle("active", stage.id === state.activeStage);
    button.classList.toggle("configured", stageConfigured(stage.id));
    const indicator = element("span", "stage-indicator");
    indicator.setAttribute("aria-hidden", "true");
    const copy = element("span");
    const title = [stage.number, stage.label].filter(Boolean).join(" ");
    copy.append(
      element("strong", "", title),
      element("small", "", stage.description),
    );
    button.append(indicator, copy);
    button.addEventListener("click", () => setActiveStage(stage.id));
    container.append(button);
  });
}

function activeStage() {
  return (
    state.schema?.stages.find((stage) => stage.id === state.activeStage) ||
    state.schema?.stages[0]
  );
}

function renderWorkspaceHeader() {
  const stage = activeStage();
  if (!stage) return;
  $("#workspace-stage-number").textContent = stage.number || "BASE";
  $("#workspace-title").textContent = `${stage.label} 阶段配置`;
  $("#workspace-description").textContent = stage.description;
  const versionText = state.loadedVersion ? `v${state.loadedVersion}` : "新草稿";
  const dirtyText = state.draft.dirty ? " · 有未保存修改" : "";
  $("#draft-identity-text").textContent =
    `${state.draft.profileId || "未命名"} · ${versionText}${dirtyText}`;
  $("#draft-identity-help").textContent = state.loadedVersion
    ? `保存将创建 v${state.loadedVersion + 1}，历史版本保持不变`
    : "保存后创建不可变 v1";
  $("#profile-id").value = state.draft.profileId;
  $("#profile-notes").value = state.draft.notes;
}

function renderPhaseSwitch() {
  const routerStage = ["memorization", "alignment", "retrieval"].includes(
    state.activeStage,
  );
  $("#phase-switch").hidden = !routerStage;
  if (!routerStage) return;
  $$("#phase-switch button").forEach((button) => {
    button.classList.toggle("active", button.dataset.stage === state.activeStage);
  });
  $("#phase-memorization-value").textContent =
    `${effectiveValue("ROUTER_MEMORIZATION_EPOCHS") || "0"} epochs`;
  $("#phase-alignment-value").textContent =
    `${effectiveValue("ROUTER_ALIGNMENT_EPOCHS") || "0"} epochs`;
  $("#phase-retrieval-value").textContent =
    `${effectiveValue("ROUTER_RETRIEVAL_EPOCHS") || "0"} epochs`;
}

function optionLabel(field, value) {
  if (field.key === "DATASET") {
    return (
      state.schema.datasets.find((dataset) => dataset.id === value)?.label ||
      value
    );
  }
  const commandLabels = {
    full: "完整流程 · full",
    prepare: "01 · prepare",
    "train-tokenizer": "02 · train-tokenizer",
    "export-codes": "03 · export-codes",
    "build-router-data": "04 · build-router-data",
    "train-memorization": "05 · train-memorization",
    "train-retrieval": "06 · train-retrieval",
    evaluate: "07 · evaluate",
    diagnose: "08 · diagnose",
    "diagnose-memorization": "09 · diagnose-memorization",
    "export-web": "10 · export-web",
  };
  if (field.key === "PIPELINE_COMMAND") {
    return commandLabels[value] || value;
  }
  return value;
}

function createControl(field, value) {
  if (field.kind === "select") {
    const select = document.createElement("select");
    field.options.forEach((option) => {
      const node = document.createElement("option");
      node.value = option;
      node.textContent = optionLabel(field, option);
      node.selected = String(option) === String(value);
      select.append(node);
    });
    return select;
  }
  if (field.kind === "bool") {
    const label = element("label", "boolean-control");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = ["1", "true", "yes"].includes(
      String(value).toLocaleLowerCase(),
    );
    label.append(input, element("span", "", input.checked ? "启用" : "关闭"));
    input.addEventListener("change", () => {
      label.querySelector("span").textContent = input.checked ? "启用" : "关闭";
    });
    return label;
  }
  const input = document.createElement("input");
  input.type = "text";
  input.value = value ?? "";
  input.spellcheck = false;
  input.autocomplete = "off";
  if (field.placeholder) input.placeholder = field.placeholder;
  return input;
}

function createFieldRow(field) {
  const row = element("div", "field-row");
  row.dataset.fieldKey = field.key;
  row.classList.toggle("changed", isOverridden(field.key));
  row.classList.toggle("compare", state.compareDefaults);

  const label = element("label", "field-label");
  label.htmlFor = `field-${field.key}`;
  label.append(
    element("strong", "", field.label),
    element("small", "", field.help || field.key),
  );

  const controlWrap = element("div", "field-control");
  const control = createControl(field, effectiveValue(field.key));
  const actualInput =
    control.matches?.("input, select") ? control : control.querySelector("input");
  actualInput.id = `field-${field.key}`;
  actualInput.dataset.key = field.key;
  actualInput.setAttribute("aria-describedby", `error-${field.key}`);
  const eventName =
    actualInput.tagName === "SELECT" || actualInput.type === "checkbox"
      ? "change"
      : "input";
  actualInput.addEventListener(eventName, () =>
    handleFieldChange(field, actualInput, row),
  );
  controlWrap.append(control);

  const source = element(
    "span",
    "field-source",
    isOverridden(field.key) ? "本版本覆盖" : field.source,
  );
  const defaultNode = element(
    "span",
    "field-default",
    `默认 · ${defaultValue(field.key) || "空"}`,
  );
  const error = element("span", "field-error");
  error.id = `error-${field.key}`;
  row.append(label, controlWrap, source, defaultNode, error);
  return row;
}

function appendGroupedSections(container, fields) {
  const grouped = new Map();
  fields.forEach((field) => {
    if (!grouped.has(field.section)) grouped.set(field.section, []);
    grouped.get(field.section).push(field);
  });
  grouped.forEach((sectionFields, section) => {
    const sectionNode = element("section", "field-section");
    sectionNode.append(element("h3", "", section));
    const grid = element("div", "field-grid");
    sectionFields.forEach((field) => grid.append(createFieldRow(field)));
    sectionNode.append(grid);
    container.append(sectionNode);
  });
}

function renderFields() {
  const container = $("#field-sections");
  container.replaceChildren();
  if (!state.schema) return;
  let fields = state.schema.fields.filter(
    (field) => field.stage === state.activeStage,
  );
  if (state.onlyOverrides) {
    fields = fields.filter((field) => isOverridden(field.key));
  }
  if (!fields.length) {
    container.append(
      element(
        "div",
        "empty-fields",
        state.onlyOverrides
          ? "当前阶段没有覆盖默认值。"
          : "当前阶段没有可配置字段。",
      ),
    );
    return;
  }
  const primary = fields.filter((field) => !field.advanced);
  const advanced = fields.filter((field) => field.advanced);
  appendGroupedSections(container, primary);
  if (advanced.length) {
    const details = document.createElement("details");
    details.className = "advanced-fields";
    const summary = document.createElement("summary");
    summary.textContent = `高级设置 · ${advanced.length} 项`;
    const content = element("div", "advanced-content");
    appendGroupedSections(content, advanced);
    details.append(summary, content);
    container.append(details);
  }
  renderFieldErrors();
}

async function switchDataset(dataset) {
  state.draft.dataset = dataset;
  state.draft.overrides = {};
  state.draft.dirty = true;
  state.loadedProfileId = "";
  state.loadedVersion = null;
  state.schema = await requestJson(
    `/api/schema?dataset=${encodeURIComponent(dataset)}`,
  );
  if (!state.draft.profileId) {
    state.draft.profileId = state.schema.default_profile_id;
  }
  await validateDraft();
  renderAll();
}

function handleFieldChange(field, input, row) {
  let value;
  if (field.kind === "bool") {
    value = input.checked ? "1" : "0";
  } else {
    value = input.value;
  }
  if (field.key === "DATASET") {
    switchDataset(value).catch((error) => showToast(error.message, true));
    return;
  }
  if (field.key === "PIPELINE_COMMAND") {
    state.draft.command = value;
  } else if (value === defaultValue(field.key)) {
    delete state.draft.overrides[field.key];
  } else {
    state.draft.overrides[field.key] = value;
  }
  state.draft.dirty = true;
  const changed = isOverridden(field.key);
  row.classList.toggle("changed", changed);
  row.querySelector(".field-source").textContent = changed
    ? "本版本覆盖"
    : field.source;
  renderWorkspaceHeader();
  renderHeaderMetrics();
  renderStages();
  scheduleValidation();
}

function setActiveStage(stageId) {
  state.activeStage = stageId;
  renderStages();
  renderWorkspaceHeader();
  renderPhaseSwitch();
  renderFields();
}

function renderValidation() {
  const banner = $("#validation-banner");
  banner.className = "validation-banner";
  banner.replaceChildren();
  if (state.validationErrors.length) {
    banner.classList.add("error");
    banner.append(
      element("strong", "", `配置检查失败 · ${state.validationErrors.length} 项`),
      element("span", "", state.validationErrors[0].message),
    );
    return;
  }
  if (!state.validation) {
    banner.classList.add("loading");
    banner.append(
      element("strong", "", "正在检查配置"),
      element("span", "", "计算最终生效值与运行契约。"),
    );
    return;
  }
  const warnings = state.validation.warnings || [];
  if (warnings.length) {
    banner.classList.add("warning");
    banner.append(
      element("strong", "", `配置检查通过 · ${warnings.length} 条提醒`),
      element("span", "", warnings[0]),
    );
    return;
  }
  const gpuCount = state.validation.contract.num_gpus || "—";
  banner.append(
    element("strong", "", "配置检查通过"),
    element(
      "span",
      "",
      `${gpuCount} GPUs 已配置 · 参数类型与层级结构合法`,
    ),
  );
}

function renderFieldErrors() {
  const errorMap = new Map(
    state.validationErrors.map((error) => [error.field, error.message]),
  );
  $$(".field-row").forEach((row) => {
    const message = errorMap.get(row.dataset.fieldKey) || "";
    row.classList.toggle("invalid", Boolean(message));
    const error = row.querySelector(".field-error");
    if (error) error.textContent = message;
  });
}

function renderContract() {
  const contract = state.validation?.contract;
  const runMatchesLoadedVersion =
    !state.draft.dirty &&
    state.loadedVersion &&
    state.currentRun?.profile_id === state.draft.profileId &&
    state.currentRun?.profile_version === state.loadedVersion;
  $("#contract-profile").textContent = state.draft.profileId || "未命名";
  $("#contract-version").textContent = state.loadedVersion
    ? state.draft.dirty
      ? `基于 v${state.loadedVersion} 的新草稿`
      : `v${state.loadedVersion}（不可变）`
    : "保存时生成 v1";
  $("#contract-overrides").textContent = String(
    Object.keys(state.validation?.overrides || state.draft.overrides).length,
  );
  $("#contract-config-path").textContent = runMatchesLoadedVersion
    ? state.currentRun.config_path
    : "提交时写入独立运行目录";
  $("#contract-command").textContent = contract?.command_text || "配置校验后生成";
  $("#contract-gpus").textContent = contract
    ? `${contract.gpus.join(", ") || "未指定"} (${contract.num_gpus || "—"} GPUs)`
    : "—";
  $("#contract-deepspeed").textContent = contract?.deepspeed || "—";
  $("#contract-precision").textContent = contract?.precision || "—";
  $("#contract-codebook").textContent = contract
    ? `${contract.num_levels || "—"} 层 · ${contract.branching_factors || "—"}`
    : "—";
  $("#contract-run-dir").textContent = contract?.run_dir || "—";
  $("#contract-checkpoint-dir").textContent =
    contract?.checkpoint_dir || "—";
  $("#contract-log-dir").textContent =
    (runMatchesLoadedVersion && state.currentRun.log_path) ||
    contract?.log_dir ||
    "提交运行快照后分配";
  $("#snapshot-preview").textContent =
    state.validation?.env_text || "配置尚未通过校验。";
}

function renderIdentityAndActions() {
  const hasErrors = state.validationErrors.length > 0 || !state.validation;
  $("#save-profile-button").disabled = state.busy || hasErrors;
  $("#submit-run-button").disabled = state.busy || hasErrors;
  $("#export-env-button").disabled = state.busy || hasErrors;
  $("#save-profile-button").textContent = state.loadedVersion
    ? `保存为 v${state.loadedVersion + 1}`
    : "仅保存新版本";
}

function renderAll() {
  renderProfiles();
  renderStages();
  renderWorkspaceHeader();
  renderPhaseSwitch();
  renderFields();
  renderValidation();
  renderContract();
  renderCurrentRun();
  renderHeaderMetrics();
  renderIdentityAndActions();
}

async function validateDraft() {
  state.validation = null;
  state.validationErrors = [];
  renderValidation();
  try {
    state.validation = await postJson("/api/validate", validationPayload());
  } catch (error) {
    state.validationErrors =
      error.payload?.errors || [{ field: "configuration", message: error.message }];
  }
  renderValidation();
  renderFieldErrors();
  renderContract();
  renderHeaderMetrics();
  renderStages();
  renderPhaseSwitch();
  renderIdentityAndActions();
  return state.validation;
}

function scheduleValidation() {
  window.clearTimeout(state.validateTimer);
  state.validateTimer = window.setTimeout(validateDraft, 260);
}

async function loadProfiles() {
  const payload = await requestJson("/api/profiles");
  state.profiles = payload.profiles || [];
  renderProfiles();
}

async function loadProfile(profileId, version) {
  setBusy(true);
  try {
    const profile = await requestJson(
      `/api/profile?id=${encodeURIComponent(profileId)}&version=${version}`,
    );
    state.schema = await requestJson(
      `/api/schema?dataset=${encodeURIComponent(profile.dataset)}`,
    );
    state.loadedProfileId = profile.profile_id;
    state.loadedVersion = profile.version;
    state.draft = {
      profileId: profile.profile_id,
      dataset: profile.dataset,
      command: profile.command,
      overrides: { ...profile.overrides },
      notes: profile.notes || "",
      dirty: false,
    };
    await validateDraft();
    renderAll();
  } catch (error) {
    showToast(`无法加载配置：${error.message}`, true);
  } finally {
    setBusy(false);
    renderIdentityAndActions();
  }
}

async function saveProfile() {
  const valid = state.validation || (await validateDraft());
  if (!valid) throw new Error("请先修复配置错误");
  const sameFamily = state.draft.profileId === state.loadedProfileId;
  const result = await postJson("/api/profiles", {
    profile_id: state.draft.profileId,
    dataset: state.draft.dataset,
    command: state.draft.command,
    notes: state.draft.notes,
    overrides: state.draft.overrides,
    parent_version:
      sameFamily && state.loadedVersion ? state.loadedVersion : null,
  });
  const profile = result.profile;
  state.loadedProfileId = profile.profile_id;
  state.loadedVersion = profile.version;
  state.draft.overrides = { ...profile.overrides };
  state.draft.dirty = false;
  state.validation = result.validation;
  state.validationErrors = [];
  await loadProfiles();
  renderAll();
  showToast(`已保存 ${profile.profile_id} v${profile.version}`);
  return profile;
}

async function ensureSavedProfile() {
  if (
    state.loadedVersion &&
    !state.draft.dirty &&
    state.loadedProfileId === state.draft.profileId
  ) {
    return {
      profile_id: state.loadedProfileId,
      version: state.loadedVersion,
    };
  }
  return saveProfile();
}

async function saveOnly() {
  setBusy(true);
  try {
    await saveProfile();
  } catch (error) {
    state.validationErrors = error.payload?.errors || state.validationErrors;
    renderAll();
    showToast(error.message, true);
  } finally {
    setBusy(false);
    renderIdentityAndActions();
  }
}

async function submitRun() {
  setBusy(true);
  try {
    const profile = await ensureSavedProfile();
    const run = await postJson("/api/runs", {
      profile_id: profile.profile_id,
      version: profile.version,
    });
    state.currentRun = run;
    renderCurrentRun();
    renderContract();
    showToast(
      run.status === "saved"
        ? "已保存运行快照；当前服务未启动训练"
        : `已提交独立任务 ${run.run_id}`,
    );
  } catch (error) {
    state.validationErrors = error.payload?.errors || state.validationErrors;
    renderAll();
    showToast(error.message, true);
  } finally {
    setBusy(false);
    renderIdentityAndActions();
  }
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function exportEnv() {
  if (!state.validation?.env_text) {
    showToast("配置尚未通过校验", true);
    return;
  }
  const version = state.loadedVersion ? `v${state.loadedVersion}` : "draft";
  downloadText(
    state.validation.env_text,
    `${state.draft.profileId}-${version}.env`,
  );
  showToast("已导出当前有效配置");
}

async function copyText(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(successMessage);
  } catch {
    showToast("浏览器未允许复制，请手动选择文本", true);
  }
}

function statusLabel(status) {
  const labels = {
    queued: "等待运行器",
    starting: "正在启动",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    failed_to_start: "启动失败",
    saved: "仅保存快照",
    unknown: "状态未知",
  };
  return labels[status] || status || "—";
}

function renderCurrentRun() {
  const run = state.currentRun;
  $("#run-empty").hidden = Boolean(run);
  $("#run-detail").hidden = !run;
  if (!run) return;
  const status = $("#run-status");
  status.textContent = statusLabel(run.status);
  status.className = ["failed", "failed_to_start", "unknown"].includes(run.status)
    ? run.status === "unknown"
      ? "unknown"
      : "failed"
    : "";
  $("#run-status-source").textContent =
    run.status === "running"
      ? "独立进程持续写入"
      : "持久化状态";
  $("#run-id").textContent = run.run_id || "—";
  $("#run-profile-version").textContent =
    `${run.profile_id || "—"} · v${run.profile_version || "—"}`;
  $("#run-stage").textContent = run.stage || "—";
  $("#run-progress").textContent = run.progress_text || "—";
  $("#run-runner-pid").textContent = run.runner_pid ?? "—";
  $("#run-training-pid").textContent = run.training_pid ?? "—";
  $("#run-exit-code").textContent = run.exit_code ?? "—";
  $("#run-checkpoint").textContent = run.latest_checkpoint || "尚未产生";
  $("#run-log-path").textContent = run.log_path || "—";
  $("#run-updated").textContent = formatTime(run.updated_at);
}

async function loadRuns(showMessage = false) {
  try {
    const payload = await requestJson("/api/runs?limit=20");
    state.currentRun = payload.runs?.[0] || null;
    renderCurrentRun();
    renderContract();
    if (showMessage) showToast("已从磁盘刷新运行状态");
  } catch (error) {
    if (showMessage) showToast(`无法刷新任务：${error.message}`, true);
  }
}

async function openRunLog() {
  if (!state.currentRun) return;
  const dialog = $("#log-dialog");
  $("#log-dialog-title").textContent = state.currentRun.run_id;
  $("#run-log").textContent = "正在读取日志…";
  dialog.showModal();
  try {
    const payload = await requestJson(
      `/api/run-log?id=${encodeURIComponent(state.currentRun.run_id)}&tail=300`,
    );
    $("#run-log").textContent = payload.text || "日志文件尚未产生内容。";
  } catch (error) {
    $("#run-log").textContent = `无法读取日志：${error.message}`;
  }
}

function setContractTab(tab) {
  const contractActive = tab === "contract";
  $("#contract-tab").classList.toggle("active", contractActive);
  $("#snapshot-tab").classList.toggle("active", !contractActive);
  $("#contract-tab").setAttribute("aria-selected", String(contractActive));
  $("#snapshot-tab").setAttribute("aria-selected", String(!contractActive));
  $("#contract-panel").hidden = !contractActive;
  $("#snapshot-panel").hidden = contractActive;
}

async function createDraft(dataset, profileId, command) {
  state.schema = await requestJson(
    `/api/schema?dataset=${encodeURIComponent(dataset)}`,
  );
  state.loadedProfileId = "";
  state.loadedVersion = null;
  state.draft = {
    profileId,
    dataset,
    command,
    overrides: {},
    notes: "",
    dirty: true,
  };
  state.activeStage = command === "full" ? "retrieval" : stageForCommand(command);
  await validateDraft();
  renderAll();
}

function stageForCommand(command) {
  return (
    {
      prepare: "embedding",
      "train-tokenizer": "tokenizer",
      "export-codes": "code",
      "build-router-data": "router_data",
      "train-memorization": "memorization",
      "train-retrieval": "retrieval",
      evaluate: "evaluation",
    }[command] || "base"
  );
}

function wireEvents() {
  $("#profile-search").addEventListener("input", renderProfiles);
  $("#new-profile-button").addEventListener("click", () => {
    $("#new-profile-id").value =
      state.schema?.default_profile_id || "clawhub-full-4gpu";
    $("#new-profile-dataset").value = state.draft.dataset;
    $("#new-profile-command").value = "full";
    $("#new-profile-dialog").showModal();
  });
  $("#new-profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (event.submitter?.value === "cancel") {
      $("#new-profile-dialog").close();
      return;
    }
    const profileId = $("#new-profile-id").value.trim().toLocaleLowerCase();
    if (!$("#new-profile-id").checkValidity()) {
      $("#new-profile-id").reportValidity();
      return;
    }
    setBusy(true);
    try {
      await createDraft(
        $("#new-profile-dataset").value,
        profileId,
        $("#new-profile-command").value,
      );
      $("#new-profile-dialog").close();
      showToast("已创建未保存草稿");
    } catch (error) {
      showToast(error.message, true);
    } finally {
      setBusy(false);
      renderIdentityAndActions();
    }
  });
  $("#profile-id").addEventListener("input", (event) => {
    state.draft.profileId = event.target.value.trim().toLocaleLowerCase();
    state.draft.dirty = true;
    renderWorkspaceHeader();
    renderHeaderMetrics();
    renderContract();
  });
  $("#profile-notes").addEventListener("input", (event) => {
    state.draft.notes = event.target.value;
    state.draft.dirty = true;
    renderWorkspaceHeader();
  });
  $("#only-overrides").addEventListener("change", (event) => {
    state.onlyOverrides = event.target.checked;
    renderFields();
  });
  $("#compare-defaults").addEventListener("click", () => {
    state.compareDefaults = !state.compareDefaults;
    $("#compare-defaults").textContent = state.compareDefaults
      ? "隐藏默认值"
      : "与默认值比较";
    renderFields();
  });
  $$("#phase-switch button").forEach((button) => {
    button.addEventListener("click", () => setActiveStage(button.dataset.stage));
  });
  $("#save-profile-button").addEventListener("click", saveOnly);
  $("#submit-run-button").addEventListener("click", submitRun);
  $("#export-env-button").addEventListener("click", exportEnv);
  $("#copy-command-button").addEventListener("click", () =>
    copyText(
      state.validation?.contract?.command_text || "",
      "已复制运行命令",
    ),
  );
  $("#copy-env-button").addEventListener("click", () =>
    copyText(state.validation?.env_text || "", "已复制有效配置"),
  );
  $("#contract-tab").addEventListener("click", () => setContractTab("contract"));
  $("#snapshot-tab").addEventListener("click", () => setContractTab("snapshot"));
  $("#refresh-runs-button").addEventListener("click", () => loadRuns(true));
  $("#view-log-button").addEventListener("click", openRunLog);
  $("#close-log-dialog").addEventListener("click", () =>
    $("#log-dialog").close(),
  );
}

async function initialize() {
  wireEvents();
  try {
    const [health, schema, profiles, runs] = await Promise.all([
      requestJson("/api/health"),
      requestJson("/api/schema?dataset=clawhub"),
      requestJson("/api/profiles"),
      requestJson("/api/runs?limit=20"),
    ]);
    state.health = health;
    state.schema = schema;
    state.profiles = profiles.profiles || [];
    state.currentRun = runs.runs?.[0] || null;
    state.draft.profileId = schema.default_profile_id;
    $("#inference-link").href = health.inference_url;
    $("#service-state-text").textContent = health.launch_enabled
      ? "训练控制台已就绪"
      : "预览模式 · 不启动训练";
    if (state.profiles.length) {
      await loadProfile(
        state.profiles[0].profile_id,
        state.profiles[0].latest_version,
      );
    } else {
      await validateDraft();
      renderAll();
    }
    window.setInterval(() => {
      if (!document.hidden) loadRuns(false);
    }, 5000);
  } catch (error) {
    $(".service-state").classList.add("failed");
    $("#service-state-text").textContent = "控制台连接失败";
    state.validationErrors = [
      { field: "configuration", message: error.message },
    ];
    renderValidation();
    showToast(error.message, true);
  }
}

initialize();
