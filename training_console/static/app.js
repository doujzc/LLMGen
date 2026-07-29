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
  validationRequestId: 0,
  validationPending: false,
  busy: false,
  stopBusy: false,
  monitorRefreshing: false,
  currentPage: "configuration",
  currentRun: null,
  runs: [],
  selectedRunId: "",
  runFilter: "all",
  monitorLogRequestId: 0,
  monitorLogRunId: "",
  monitorLogUpdating: false,
  loadedProfileId: "",
  loadedVersion: null,
  loadedRevision: null,
  draft: {
    profileId: "clawhub-full-4gpu",
    dataset: "clawhub",
    command: "full",
    overrides: {},
    notes: "",
    dirty: true,
  },
};

const ACTIVE_RUN_STATUSES = new Set([
  "queued",
  "starting",
  "running",
  "stopping",
  "unknown",
]);
const PROCESS_RUN_STATUSES = new Set(["starting", "running", "stopping"]);
const ATTENTION_RUN_STATUSES = new Set([
  "failed",
  "failed_to_start",
  "stopped",
  "unknown",
]);
const PIPELINE_STAGES = [
  { id: "embedding", number: "01", label: "Embedding", markers: ["01", "embedding"] },
  { id: "tokenizer", number: "02", label: "Tokenizer", markers: ["02", "tokenizer"] },
  { id: "code", number: "03", label: "Code", markers: ["03", "code 导出"] },
  { id: "router_data", number: "04", label: "Router Data", markers: ["04", "router 数据"] },
  { id: "memorization", number: "05", label: "Memorization", markers: ["05", "memorization"] },
  { id: "alignment", number: "06a", label: "Alignment", markers: ["06a", "alignment"] },
  { id: "retrieval", number: "06b", label: "Retrieval", markers: ["06b", "retrieval", "06 retrieval"] },
  { id: "evaluation", number: "07", label: "Evaluation", markers: ["07", "评估"] },
];

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

function dismissErrorToast() {
  const toast = $("#toast");
  if (!toast.hidden && toast.classList.contains("error")) {
    window.clearTimeout(showToast.timer);
    toast.hidden = true;
  }
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
    : "保存并提交独立任务";
}

function fieldByKey(key) {
  return state.schema?.fields.find((field) => field.key === key);
}

function derivedFieldValue(field) {
  if (!field?.derived_from || !field.derived_suffix) return null;
  return `$${field.derived_from}/${field.derived_suffix}`;
}

function effectiveValue(key) {
  if (key === "DATASET") return state.draft.dataset;
  if (key === "PIPELINE_COMMAND") return state.draft.command;
  if (Object.hasOwn(state.draft.overrides, key)) {
    return state.draft.overrides[key];
  }
  const linkedValue = derivedFieldValue(fieldByKey(key));
  if (linkedValue !== null) return linkedValue;
  return (
    state.validation?.resolved?.[key] ??
    state.schema?.defaults?.[key] ??
    ""
  );
}

function defaultValue(key) {
  if (key === "DATASET") return state.schema?.dataset || "clawhub";
  if (key === "PIPELINE_COMMAND") return "full";
  const linkedValue = derivedFieldValue(fieldByKey(key));
  if (linkedValue !== null) return linkedValue;
  return state.schema?.defaults?.[key] ?? "";
}

function fieldSourceText(field) {
  if (isOverridden(field.key)) return "本版本覆盖";
  if (field.derived_from) return `联动 · ${field.derived_from}`;
  return field.source;
}

function fieldDefaultText(field) {
  if (field.derived_from && field.derived_suffix) {
    return `默认 · $${field.derived_from}/${field.derived_suffix}`;
  }
  return `默认 · ${defaultValue(field.key) || "空"}`;
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
    ? `v${state.loadedVersion} · r${state.loadedRevision || 1}`
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
      : "尚未保存配置。当前表单是一个草稿，保存后可以继续编辑。";
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
          version.version === profile.latest_version
            ? `最新 · 可编辑 · r${version.revision || 1}`
            : `可编辑 · r${version.revision || 1}`,
        ),
        element(
          "time",
          "",
          formatTime(version.updated_at || version.created_at),
        ),
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
  const versionText = state.loadedVersion
    ? `v${state.loadedVersion} · r${state.loadedRevision || 1}`
    : "新草稿";
  const dirtyText = state.draft.dirty ? " · 有未保存修改" : "";
  $("#draft-identity-text").textContent =
    `${state.draft.profileId || "未命名"} · ${versionText}${dirtyText}`;
  $("#draft-identity-help").textContent = state.loadedVersion
    ? `保存将原地更新 v${state.loadedVersion}`
    : "保存后创建可编辑配置 v1";
  $("#profile-id").value = state.draft.profileId;
  $("#profile-id").disabled = Boolean(state.loadedVersion);
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
  row.classList.toggle(
    "linked",
    Boolean(field.derived_from) && !isOverridden(field.key),
  );
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
    fieldSourceText(field),
  );
  if (field.derived_from) {
    source.title = `默认值跟随 ${field.derived_from}；手动修改后转为独立覆盖`;
  }
  const defaultNode = element(
    "span",
    "field-default",
    fieldDefaultText(field),
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
    (field) =>
      field.stage === state.activeStage ||
      (field.visible_stages || []).includes(state.activeStage),
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
  row.classList.toggle("linked", Boolean(field.derived_from) && !changed);
  row.querySelector(".field-source").textContent = fieldSourceText(field);
  invalidateDraftValidation();
  renderWorkspaceHeader();
  renderHeaderMetrics();
  renderStages();
  scheduleValidation();
}

function invalidateDraftValidation() {
  window.clearTimeout(state.validateTimer);
  dismissErrorToast();
  state.validationRequestId += 1;
  state.validationPending = false;
  state.validation = null;
  state.validationErrors = [];
  renderValidation();
  renderFieldErrors();
  renderContract();
  renderIdentityAndActions();
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
  const derivedDirectories =
    state.schema?.directory_contract?.derived || [];
  const linkedDirectoryCount = derivedDirectories.filter(
    (directory) => !Object.hasOwn(state.draft.overrides, directory.key),
  ).length;
  const overriddenDirectoryCount =
    derivedDirectories.length - linkedDirectoryCount;
  const directorySummary = overriddenDirectoryCount
    ? `${linkedDirectoryCount} 个目录跟随 RUN_DIR · ` +
      `${overriddenDirectoryCount} 个单独覆盖`
    : `${linkedDirectoryCount} 个产物目录跟随 RUN_DIR`;
  banner.append(
    element("strong", "", "配置检查通过"),
    element(
      "span",
      "",
      `${gpuCount} GPUs 已配置 · ${directorySummary}`,
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
    state.currentRun?.profile_version === state.loadedVersion &&
    (state.currentRun?.profile_revision || 1) ===
      (state.loadedRevision || 1);
  $("#contract-profile").textContent = state.draft.profileId || "未命名";
  $("#contract-version").textContent = state.loadedVersion
    ? state.draft.dirty
      ? `v${state.loadedVersion} · r${state.loadedRevision || 1} · 待保存`
      : `v${state.loadedVersion} · r${state.loadedRevision || 1}（可编辑）`
    : "保存时创建 v1 · r1";
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
  $("#contract-gpu-order").textContent = contract
    ? `${contract.cuda_device_order || "PCI_BUS_ID"} · 数字编号运行时绑定 UUID`
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
  const savedAndClean =
    state.loadedVersion &&
    !state.draft.dirty &&
    state.loadedProfileId === state.draft.profileId;
  $("#save-profile-button").disabled =
    state.busy || Boolean(savedAndClean);
  $("#submit-run-button").disabled = state.busy || hasErrors;
  $("#export-env-button").disabled = state.busy || hasErrors;
  $("#save-profile-button").textContent = state.validationErrors.length
    ? "重新检查并保存"
    : state.loadedVersion
      ? "保存修改"
      : "保存配置";
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
  renderMonitor();
  renderHeaderMetrics();
  renderIdentityAndActions();
}

async function validateDraft() {
  window.clearTimeout(state.validateTimer);
  const requestId = ++state.validationRequestId;
  state.validationPending = true;
  state.validation = null;
  state.validationErrors = [];
  renderValidation();
  let validation = null;
  let validationErrors = [];
  try {
    validation = await postJson("/api/validate", validationPayload());
  } catch (error) {
    validationErrors =
      error.payload?.errors || [{ field: "configuration", message: error.message }];
  }
  if (requestId !== state.validationRequestId) return state.validation;
  state.validationPending = false;
  state.validation = validation;
  state.validationErrors = validationErrors;
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
    state.loadedRevision = profile.revision || 1;
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
  const valid = await validateDraft();
  if (!valid) throw new Error("请先修复配置错误");
  const sameFamily = state.draft.profileId === state.loadedProfileId;
  const result = await postJson("/api/profiles", {
    profile_id: state.draft.profileId,
    dataset: state.draft.dataset,
    command: state.draft.command,
    notes: state.draft.notes,
    overrides: state.draft.overrides,
    version: sameFamily && state.loadedVersion ? state.loadedVersion : null,
    expected_revision:
      sameFamily && state.loadedVersion ? state.loadedRevision : null,
  });
  const profile = result.profile;
  state.loadedProfileId = profile.profile_id;
  state.loadedVersion = profile.version;
  state.loadedRevision = profile.revision || 1;
  state.draft.overrides = { ...profile.overrides };
  state.draft.dirty = false;
  state.validation = result.validation;
  state.validationErrors = [];
  await loadProfiles();
  renderAll();
  showToast(
    `已保存 ${profile.profile_id} v${profile.version} · ` +
      `r${profile.revision || 1}`,
  );
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
      revision: state.loadedRevision,
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
    state.runs = [run, ...state.runs.filter((item) => item.run_id !== run.run_id)];
    state.currentRun = run;
    state.selectedRunId = run.run_id;
    renderCurrentRun();
    renderContract();
    renderMonitor();
    setConsolePage("monitor");
    loadMonitorLog(false);
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
    stopping: "正在停止",
    stopped: "已停止",
    succeeded: "已完成",
    failed: "失败",
    failed_to_start: "启动失败",
    saved: "仅保存快照",
    unknown: "状态未知",
  };
  return labels[status] || status || "—";
}

function statusTone(status) {
  if (["running", "succeeded"].includes(status)) return "positive";
  if (["queued", "starting", "stopping"].includes(status)) return "pending";
  if (["failed", "failed_to_start", "unknown"].includes(status)) return "negative";
  if (status === "stopped") return "stopped";
  return "neutral";
}

function selectedRun() {
  return (
    state.runs.find((run) => run.run_id === state.selectedRunId) ||
    state.runs[0] ||
    null
  );
}

function configuredGpuIds(run) {
  return Array.isArray(run?.configured_gpus)
    ? run.configured_gpus.map(String)
    : [];
}

function gpuMatchesRunAssignment(run, gpu) {
  if (!run || !gpu) return false;
  const gpuIndex = String(gpu.index);
  const gpuUuid = String(gpu.uuid || "");
  const bindings = Array.isArray(run.gpu_bindings) ? run.gpu_bindings : [];
  if (
    bindings.some(
      (binding) =>
        String(binding.index || "") === gpuIndex ||
        (gpuUuid && String(binding.uuid || "") === gpuUuid),
    )
  ) {
    return true;
  }
  return configuredGpuIds(run).some(
    (token) =>
      token === gpuIndex ||
      (gpuUuid && (gpuUuid.startsWith(token) || token.startsWith(gpuUuid))),
  );
}

function gpuProcessesForRun(run, gpu) {
  if (!run || !Array.isArray(gpu?.processes)) return [];
  const processGroup = Number(run.training_pgid || run.training_pid || 0);
  const trainingPid = Number(run.training_pid || 0);
  if (!processGroup && !trainingPid) return [];
  return gpu.processes.filter((process) => {
    const pid = Number(process.pid || 0);
    const processGroupId = Number(process.process_group_id || 0);
    return (
      (processGroup && processGroupId === processGroup) ||
      (trainingPid && pid === trainingPid)
    );
  });
}

function persistedGpuObservation(run, gpu) {
  if (!run || !gpu) return null;
  const observations = Array.isArray(run.observed_gpu_processes)
    ? run.observed_gpu_processes
    : [];
  return (
    observations.find(
      (observation) =>
        String(observation.index || "") === String(gpu.index) ||
        (gpu.uuid && String(observation.uuid || "") === String(gpu.uuid)),
    ) || null
  );
}

function assignedGpusForRun(run, gpus = state.health?.gpus || []) {
  return gpus.filter((gpu) => gpuMatchesRunAssignment(run, gpu));
}

function observedGpusForRun(run, gpus = state.health?.gpus || []) {
  return gpus.filter(
    (gpu) =>
      gpuProcessesForRun(run, gpu).length ||
      persistedGpuObservation(run, gpu),
  );
}

function liveObservedGpusForRun(run, gpus = state.health?.gpus || []) {
  return gpus.filter((gpu) => gpuProcessesForRun(run, gpu).length);
}

function shortGpuToken(token) {
  const value = String(token || "");
  if (value.length <= 18) return value;
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function runDuration(run) {
  const started = new Date(run.started_at || run.created_at || "");
  if (Number.isNaN(started.getTime())) return "—";
  const finished = new Date(
    run.finished_at || run.stopped_at || Date.now(),
  );
  const seconds = Math.max(
    0,
    Math.floor((finished.getTime() - started.getTime()) / 1000),
  );
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  if (days) return `${days}d ${hours}h ${minutes}m`;
  if (hours) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

function runProgress(run) {
  const text = String(run.progress_text || "");
  const match = text.match(/(\d[\d,]*)\s*\/\s*(\d[\d,]*)/);
  if (match) {
    const done = Number(match[1].replaceAll(",", ""));
    const total = Number(match[2].replaceAll(",", ""));
    if (Number.isFinite(done) && total > 0) {
      return Math.max(0, Math.min(100, (done / total) * 100));
    }
  }
  if (run.status === "succeeded") return 100;
  return null;
}

function runMatchesFilter(run) {
  if (state.runFilter === "active") return ACTIVE_RUN_STATUSES.has(run.status);
  if (state.runFilter === "succeeded") return run.status === "succeeded";
  if (state.runFilter === "attention") {
    return ATTENTION_RUN_STATUSES.has(run.status);
  }
  return true;
}

function renderMonitorSummary() {
  const processActive = state.runs.filter((run) =>
    PROCESS_RUN_STATUSES.has(run.status),
  ).length;
  const queued = state.runs.filter((run) => run.status === "queued").length;
  const succeeded = state.runs.filter((run) => run.status === "succeeded").length;
  const activeTotal = processActive + queued;
  $("#monitor-active-count").textContent = String(processActive);
  $("#monitor-queued-count").textContent = String(queued);
  $("#monitor-success-count").textContent = String(succeeded);
  $("#active-run-badge").textContent = String(activeTotal);
  $("#active-run-badge").hidden = activeTotal === 0;

  const gpus = state.health?.gpus || [];
  const gpuCount = state.health?.gpu_count;
  const run = selectedRun();
  const configuredCount = configuredGpuIds(run).length;
  const hostCount =
    gpuCount === null || gpuCount === undefined ? "—" : String(gpuCount);
  $("#monitor-gpu-count").textContent = run
    ? `${configuredCount || "—"} / ${hostCount}`
    : hostCount;
  if (!gpus.length) {
    $("#monitor-gpu-summary").textContent =
      gpuCount === 0 ? "未检测到 NVIDIA GPU" : "等待 nvidia-smi";
    return;
  }
  if (run) {
    const assigned = assignedGpusForRun(run, gpus);
    const observed = observedGpusForRun(run, gpus);
    const liveObserved = liveObservedGpusForRun(run, gpus);
    const breached = observed.filter(
      (gpu) => !gpuMatchesRunAssignment(run, gpu),
    );
    if (run.gpu_contract_violation || breached.length) {
      const labels = breached.map((gpu) => gpu.index).join(", ");
      $("#monitor-gpu-summary").textContent =
        labels ? `检测到越界占用 · GPU ${labels}` : "检测到 GPU 越界占用";
      return;
    }
    if (liveObserved.length) {
      $("#monitor-gpu-summary").textContent =
        `实际占用 GPU ${liveObserved.map((gpu) => gpu.index).join(", ")}`;
      return;
    }
    if (observed.length) {
      $("#monitor-gpu-summary").textContent =
        `最后观测 GPU ${observed.map((gpu) => gpu.index).join(", ")}`;
      return;
    }
    if (assigned.length) {
      $("#monitor-gpu-summary").textContent =
        `已分配 GPU ${assigned.map((gpu) => gpu.index).join(", ")} · 等待 CUDA`;
      return;
    }
    $("#monitor-gpu-summary").textContent = "任务 GPU 契约尚未解析";
    return;
  }
  const averageUtilization = Math.round(
    gpus.reduce((total, gpu) => total + gpu.utilization, 0) / gpus.length,
  );
  const used = gpus.reduce((total, gpu) => total + gpu.memory_used_mib, 0);
  const total = gpus.reduce((sum, gpu) => sum + gpu.memory_total_mib, 0);
  $("#monitor-gpu-summary").textContent =
    `平均 ${averageUtilization}% · ${(used / 1024).toFixed(1)} / ` +
    `${(total / 1024).toFixed(1)} GiB`;
}

function renderRunList() {
  const container = $("#monitor-run-list");
  container.replaceChildren();
  const runs = state.runs.filter(runMatchesFilter);
  $("#monitor-run-count").textContent = `${state.runs.length} RUNS`;
  if (!runs.length) {
    container.append(
      element(
        "div",
        "ledger-empty",
        state.runs.length ? "当前筛选条件下没有运行记录。" : "尚未提交训练任务。",
      ),
    );
    return;
  }
  runs.forEach((run) => {
    const button = element("button", "monitor-run-card");
    button.type = "button";
    button.classList.toggle("selected", run.run_id === state.selectedRunId);
    button.dataset.status = statusTone(run.status);
    const heading = element("span", "run-card-heading");
    heading.append(
      element("strong", "", run.profile_id || "未命名配置"),
      element("span", `run-card-state ${statusTone(run.status)}`, statusLabel(run.status)),
    );
    const identity = element("code", "run-card-id", run.run_id || "—");
    const stage = element("span", "run-card-stage", run.stage || "等待阶段信息");
    const footer = element("span", "run-card-footer");
    footer.append(
      element("span", "", run.progress_text || run.command || "—"),
      element("time", "", formatTime(run.updated_at || run.created_at)),
    );
    button.append(heading, identity, stage, footer);
    button.addEventListener("click", () => {
      state.selectedRunId = run.run_id;
      renderMonitor();
      loadMonitorLog(false);
    });
    container.append(button);
  });
}

function stagesForRun(run) {
  const singleStage = {
    prepare: "embedding",
    "train-tokenizer": "tokenizer",
    "export-codes": "code",
    "build-router-data": "router_data",
    "train-memorization": "memorization",
    "train-retrieval": "retrieval",
    evaluate: "evaluation",
  }[run.command];
  return singleStage
    ? PIPELINE_STAGES.filter((stage) => stage.id === singleStage)
    : PIPELINE_STAGES;
}

function renderMonitorStageTrack(run) {
  const container = $("#monitor-stage-track");
  container.replaceChildren();
  const stages = stagesForRun(run);
  const stageText = String(
    run.stop_requested_stage || run.stage || "",
  ).toLocaleLowerCase();
  let currentIndex = stages.findIndex((stage) =>
    stage.markers.some((marker) => stageText.includes(marker)),
  );
  if (run.status === "succeeded") currentIndex = stages.length;
  stages.forEach((stage, index) => {
    const node = element("div", "monitor-stage-node");
    const completed = currentIndex === stages.length || index < currentIndex;
    const current = index === currentIndex;
    node.classList.toggle("complete", completed);
    node.classList.toggle("current", current);
    node.classList.toggle(
      "failed",
      current && ["failed", "failed_to_start", "unknown"].includes(run.status),
    );
    node.classList.toggle("stopped", current && run.status === "stopped");
    node.append(
      element("i", "", stage.number),
      element("span", "", stage.label),
    );
    container.append(node);
  });
}

function renderGpuBoard() {
  const board = $("#gpu-board");
  board.replaceChildren();
  const gpus = state.health?.gpus || [];
  const run = selectedRun();
  const requested = configuredGpuIds(run);
  const observed = observedGpusForRun(run, gpus);
  const liveObserved = liveObservedGpusForRun(run, gpus);
  const breached = observed.filter(
    (gpu) => !gpuMatchesRunAssignment(run, gpu),
  );
  $("#gpu-requested-devices").textContent = run
    ? requested.length
      ? requested.map((token) => `GPU ${token}`).join(" · ")
      : "未记录"
    : "未选择任务";
  const bindings = Array.isArray(run?.gpu_bindings) ? run.gpu_bindings : [];
  $("#gpu-runtime-devices").textContent = run
    ? bindings.length
      ? bindings
          .map((binding, index) =>
            binding.index
              ? `cuda:${binding.logical_index ?? index} → GPU ` +
                `${binding.index} → ${shortGpuToken(binding.uuid)}`
              : shortGpuToken(binding.uuid || binding.requested),
          )
          .join(" · ")
      : run.runtime_visible_devices
        ? String(run.runtime_visible_devices)
            .split(",")
            .map(shortGpuToken)
            .join(" · ")
        : "等待 Runner 解析"
    : "—";
  $("#gpu-observed-devices").textContent = run
    ? observed.length
      ? observed
          .map((gpu) => {
            const processes = gpuProcessesForRun(run, gpu);
            const persisted = persistedGpuObservation(run, gpu);
            const pids = processes.length
              ? processes.map((process) => process.pid)
              : persisted?.pids || [];
            return `GPU ${gpu.index}${pids.length ? ` · PID ${pids.join("/")}` : ""}`;
          })
          .join(" · ")
      : PROCESS_RUN_STATUSES.has(run.status)
        ? "等待本任务创建 CUDA 进程"
        : "当前无本任务 CUDA 进程"
    : "—";
  const contractStatus = $("#gpu-panel-updated");
  if (!run) {
    contractStatus.textContent = "请选择运行";
    contractStatus.className = "health-label neutral";
  } else if (run.gpu_contract_violation || breached.length) {
    const labels = breached.map((gpu) => gpu.index).join(", ");
    contractStatus.textContent =
      labels ? `越界占用 GPU ${labels}` : "已记录 GPU 越界占用";
    contractStatus.className = "health-label negative";
  } else if (liveObserved.length) {
    contractStatus.textContent = "运行绑定已核验";
    contractStatus.className = "health-label positive";
  } else if (observed.length) {
    contractStatus.textContent =
      `最后 GPU 观测与绑定一致 · ${formatTime(run.last_gpu_observed_at)}`;
    contractStatus.className = "health-label positive";
  } else if (PROCESS_RUN_STATUSES.has(run.status)) {
    contractStatus.textContent = run.gpu_binding_verified
      ? "UUID 绑定已生效 · 等待 CUDA"
      : "等待 GPU 绑定核验";
    contractStatus.className = "health-label pending";
  } else {
    contractStatus.textContent = "运行已结束";
    contractStatus.className = "health-label neutral";
  }
  if (!gpus.length) {
    board.append(
      element(
        "div",
        "gpu-empty",
        state.health?.gpu_count === 0
          ? "当前主机未检测到 NVIDIA GPU。"
          : "nvidia-smi 暂不可用；训练任务仍由磁盘状态独立监控。",
      ),
    );
    return;
  }
  gpus.forEach((gpu) => {
    const card = element("article", "gpu-card");
    const assigned = gpuMatchesRunAssignment(run, gpu);
    const runProcesses = gpuProcessesForRun(run, gpu);
    const persistedObservation = persistedGpuObservation(run, gpu);
    const occupied = runProcesses.length > 0;
    const previouslyObserved = Boolean(persistedObservation);
    const observedForRun = occupied || previouslyObserved;
    card.classList.toggle("assigned", assigned);
    card.classList.toggle("observed", observedForRun && assigned);
    card.classList.toggle("breached", observedForRun && !assigned);
    card.classList.toggle(
      "host-only",
      Boolean(run) && !assigned && !observedForRun,
    );
    const heading = element("div", "gpu-card-heading");
    const identity = element("div", "gpu-card-identity");
    let badgeLabel = "其他 GPU";
    if (!run) badgeLabel = "整机";
    else if (occupied && assigned) badgeLabel = "本任务占用";
    else if (occupied) badgeLabel = "越界占用";
    else if (previouslyObserved && assigned) badgeLabel = "最后观测";
    else if (previouslyObserved) badgeLabel = "越界记录";
    else if (assigned) badgeLabel = "已分配";
    const badge = element(
      "span",
      "gpu-contract-badge",
      badgeLabel,
    );
    identity.append(element("strong", "", `GPU ${gpu.index}`), badge);
    heading.append(
      identity,
      element("span", "", `${gpu.temperature_c}°C`),
    );
    const name = element(
      "span",
      "gpu-name",
      `${gpu.name} · ${gpu.pci_bus_id || "PCI —"}`,
    );
    const memoryPercent =
      gpu.memory_total_mib > 0
        ? Math.round((gpu.memory_used_mib / gpu.memory_total_mib) * 100)
        : 0;
    const utilization = element("div", "gpu-reading");
    utilization.append(
      element("span", "", "Compute"),
      element("strong", "", `${gpu.utilization}%`),
    );
    const utilizationTrack = element("div", "gpu-meter");
    const utilizationBar = element("span");
    utilizationBar.style.width = `${gpu.utilization}%`;
    utilizationTrack.append(utilizationBar);
    const memory = element("div", "gpu-reading");
    memory.append(
      element("span", "", "Memory"),
      element(
        "strong",
        "",
        `${(gpu.memory_used_mib / 1024).toFixed(1)} / ` +
          `${(gpu.memory_total_mib / 1024).toFixed(1)} GiB`,
      ),
    );
    const memoryTrack = element("div", "gpu-meter memory");
    const memoryBar = element("span");
    memoryBar.style.width = `${memoryPercent}%`;
    memoryTrack.append(memoryBar);
    let processText = "不属于当前任务契约";
    if (occupied) {
      processText = `本任务 PID ${runProcesses
        .map((process) => process.pid)
        .join(", ")}`;
    } else if (previouslyObserved) {
      processText =
        `最后观测 PID ${(persistedObservation.pids || []).join(", ")}`;
    } else if (assigned) {
      processText = "任务已绑定，等待 CUDA 上下文";
    }
    const processLine = element(
      "span",
      "gpu-process-line",
      processText,
    );
    card.append(
      heading,
      name,
      processLine,
      utilization,
      utilizationTrack,
      memory,
      memoryTrack,
    );
    board.append(card);
  });
}

function renderMonitorDetail() {
  const run = selectedRun();
  $("#monitor-empty").hidden = Boolean(run);
  $("#monitor-detail").hidden = !run;
  if (!run) {
    renderGpuBoard();
    return;
  }
  if (state.selectedRunId !== run.run_id) state.selectedRunId = run.run_id;
  const status = $("#monitor-status-pill");
  status.textContent = statusLabel(run.status);
  status.className = `run-pill ${statusTone(run.status)}`;
  $("#selected-run-title").textContent = run.run_id || "—";
  $("#monitor-run-subtitle").textContent =
    `${run.profile_id || "—"} · v${run.profile_version || "—"} · ` +
    `${run.dataset || "—"} / ${run.command || "—"}`;
  $("#monitor-updated").textContent =
    `最后更新 ${formatTime(run.updated_at || run.created_at)}`;

  const stopButton = $("#monitor-stop-button");
  stopButton.disabled =
    state.stopBusy ||
    !ACTIVE_RUN_STATUSES.has(run.status) ||
    run.status === "stopping";
  stopButton.textContent =
    state.stopBusy || run.status === "stopping" ? "正在停止…" : "停止训练";

  const progress = runProgress(run);
  const progressTrack = $("#monitor-progress-track");
  progressTrack.classList.toggle(
    "indeterminate",
    progress === null && PROCESS_RUN_STATUSES.has(run.status),
  );
  $("#monitor-progress-bar").style.width = `${progress ?? 0}%`;
  if (progress === null) {
    progressTrack.removeAttribute("aria-valuenow");
    $("#monitor-progress-percent").textContent = "—";
  } else {
    progressTrack.setAttribute("aria-valuenow", String(Math.round(progress)));
    $("#monitor-progress-percent").textContent = `${progress.toFixed(1)}%`;
  }
  $("#monitor-progress-text").textContent =
    run.progress_text || "等待进度日志";
  renderMonitorStageTrack(run);

  $("#monitor-duration").textContent = runDuration(run);
  $("#monitor-stage").textContent = run.stage || "—";
  $("#monitor-runner-pid").textContent = run.runner_pid ?? "—";
  $("#monitor-training-pid").textContent = run.training_pid ?? "—";
  $("#monitor-exit-code").textContent = run.exit_code ?? "—";
  $("#monitor-profile-version").textContent =
    `${run.profile_id || "—"} · v${run.profile_version || "—"} · ` +
    `r${run.profile_revision || 1}`;
  $("#monitor-checkpoint").textContent =
    run.latest_checkpoint || "尚未产生";
  $("#monitor-artifact-dir").textContent = run.artifact_run_dir || "—";
  const processHealth = $("#monitor-process-health");
  if (run.training_alive) {
    processHealth.textContent = "训练进程在线";
    processHealth.className = "health-label positive";
  } else if (run.runner_alive) {
    processHealth.textContent = "Runner 在线";
    processHealth.className = "health-label pending";
  } else if (ACTIVE_RUN_STATUSES.has(run.status)) {
    processHealth.textContent = "进程不可见";
    processHealth.className = "health-label negative";
  } else {
    processHealth.textContent = "运行已结束";
    processHealth.className = "health-label neutral";
  }
  renderGpuBoard();
}

function renderMonitor() {
  renderMonitorSummary();
  renderRunList();
  renderMonitorDetail();
}

function setConsolePage(page) {
  const monitorActive = page === "monitor";
  state.currentPage = monitorActive ? "monitor" : "configuration";
  $("#configuration-page").hidden = monitorActive;
  $("#monitor-page").hidden = !monitorActive;
  $("#configuration-view-button").classList.toggle("active", !monitorActive);
  $("#monitor-view-button").classList.toggle("active", monitorActive);
  if (monitorActive) {
    $("#monitor-view-button").setAttribute("aria-current", "page");
    $("#configuration-view-button").removeAttribute("aria-current");
  } else {
    $("#configuration-view-button").setAttribute("aria-current", "page");
    $("#monitor-view-button").removeAttribute("aria-current");
  }
  window.history.replaceState(
    null,
    "",
    monitorActive ? "#monitor" : "#configuration",
  );
  if (monitorActive) {
    renderMonitor();
    refreshMonitor(false);
  }
}

async function loadHealth(showMessage = false) {
  try {
    state.health = await requestJson("/api/health");
    renderHeaderMetrics();
    renderMonitorSummary();
    renderGpuBoard();
    if (showMessage) showToast("已刷新 GPU 与服务状态");
  } catch (error) {
    if (showMessage) showToast(`无法刷新服务状态：${error.message}`, true);
  }
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
    : run.status === "stopped"
      ? "stopped"
      : "";
  $("#run-status-source").textContent =
    run.status === "running"
      ? "独立进程持续写入"
      : "持久化状态";
  $("#run-id").textContent = run.run_id || "—";
  $("#run-profile-version").textContent =
    `${run.profile_id || "—"} · v${run.profile_version || "—"} · ` +
    `r${run.profile_revision || 1}`;
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
    const payload = await requestJson("/api/runs?limit=100");
    state.runs = payload.runs || [];
    state.currentRun = state.runs[0] || null;
    if (
      !state.selectedRunId ||
      !state.runs.some((run) => run.run_id === state.selectedRunId)
    ) {
      state.selectedRunId =
        state.runs.find((run) => ACTIVE_RUN_STATUSES.has(run.status))?.run_id ||
        state.runs[0]?.run_id ||
        "";
    }
    renderCurrentRun();
    renderContract();
    renderMonitor();
    if (showMessage) showToast("已从磁盘刷新运行状态");
  } catch (error) {
    if (showMessage) showToast(`无法刷新任务：${error.message}`, true);
  }
}

function monitorLogNearBottom(log, tolerance = 36) {
  return log.scrollHeight - log.scrollTop - log.clientHeight <= tolerance;
}

function setMonitorLogFollowing(enabled, { scroll = false } = {}) {
  const checkbox = $("#monitor-follow-log");
  const label = $("#monitor-follow-label");
  checkbox.checked = enabled;
  label.textContent = enabled ? "跟随底部" : "已暂停跟随";
  label.classList.toggle("paused", !enabled);
  if (enabled && scroll) {
    const log = $("#monitor-log");
    state.monitorLogUpdating = true;
    log.scrollTop = log.scrollHeight;
    window.requestAnimationFrame(() => {
      state.monitorLogUpdating = false;
    });
  }
}

async function loadMonitorLog(showMessage = false) {
  const run = selectedRun();
  if (!run) {
    $("#monitor-log").textContent = "尚无运行日志。";
    $("#monitor-log-updated").textContent = "尚未读取";
    return;
  }
  if (state.monitorLogRunId !== run.run_id) {
    state.monitorLogRunId = run.run_id;
    setMonitorLogFollowing(true);
  }
  const requestId = ++state.monitorLogRequestId;
  try {
    const payload = await requestJson(
      `/api/run-log?id=${encodeURIComponent(run.run_id)}&tail=600`,
    );
    if (requestId !== state.monitorLogRequestId) return;
    const log = $("#monitor-log");
    const previousScrollTop = log.scrollTop;
    const following = $("#monitor-follow-log").checked;
    state.monitorLogUpdating = true;
    log.textContent = payload.text || "日志文件尚未产生内容。";
    $("#monitor-log-updated").textContent =
      `读取于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}` +
      (following ? "" : " · 跟随已暂停");
    log.scrollTop = following
      ? log.scrollHeight
      : Math.min(
          previousScrollTop,
          Math.max(0, log.scrollHeight - log.clientHeight),
        );
    window.requestAnimationFrame(() => {
      state.monitorLogUpdating = false;
    });
    if (showMessage) showToast("已刷新持久化训练日志");
  } catch (error) {
    if (requestId !== state.monitorLogRequestId) return;
    $("#monitor-log").textContent = `无法读取日志：${error.message}`;
    $("#monitor-log-updated").textContent = "读取失败";
    if (showMessage) showToast(error.message, true);
  }
}

async function refreshMonitor(showMessage = false) {
  if (state.monitorRefreshing) return;
  state.monitorRefreshing = true;
  $("#monitor-poll-state").classList.add("refreshing");
  try {
    await Promise.all([loadRuns(false), loadHealth(false)]);
    await loadMonitorLog(false);
    if (showMessage) showToast("运行、日志与 GPU 状态均已刷新");
  } finally {
    state.monitorRefreshing = false;
    $("#monitor-poll-state").classList.remove("refreshing");
  }
}

function openStopDialog() {
  const run = selectedRun();
  if (!run || !ACTIVE_RUN_STATUSES.has(run.status)) return;
  $("#stop-run-id").textContent = run.run_id;
  $("#stop-run-stage").textContent =
    `${statusLabel(run.status)} · ${run.stage || "等待阶段信息"}`;
  $("#stop-run-dialog").showModal();
}

async function confirmStopRun() {
  const run = selectedRun();
  if (!run) return;
  state.stopBusy = true;
  renderMonitorDetail();
  $("#confirm-stop-button").disabled = true;
  try {
    const stopped = await postJson("/api/runs/stop", {
      run_id: run.run_id,
    });
    state.runs = state.runs.map((item) =>
      item.run_id === stopped.run_id ? stopped : item,
    );
    state.currentRun =
      state.runs.find((item) => item.run_id === state.currentRun?.run_id) ||
      state.runs[0] ||
      null;
    $("#stop-run-dialog").close();
    renderCurrentRun();
    renderMonitor();
    showToast(`已向 ${run.run_id} 写入停止请求`);
  } catch (error) {
    showToast(`停止失败：${error.message}`, true);
  } finally {
    state.stopBusy = false;
    $("#confirm-stop-button").disabled = false;
    renderMonitorDetail();
  }
}

async function openSelectedRunConfig() {
  const run = selectedRun();
  if (!run) return;
  const sameProfile =
    state.loadedProfileId === run.profile_id &&
    state.loadedVersion === run.profile_version;
  if (
    state.draft.dirty &&
    !sameProfile &&
    !window.confirm("当前配置有未保存修改，仍要切换到该运行的配置吗？")
  ) {
    return;
  }
  setConsolePage("configuration");
  await loadProfile(run.profile_id, run.profile_version);
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
  state.loadedRevision = null;
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
  $("#configuration-view-button").addEventListener("click", () =>
    setConsolePage("configuration"),
  );
  $("#monitor-view-button").addEventListener("click", () =>
    setConsolePage("monitor"),
  );
  $("#open-monitor-button").addEventListener("click", () =>
    setConsolePage("monitor"),
  );
  $("#monitor-empty-config-button").addEventListener("click", () =>
    setConsolePage("configuration"),
  );
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
  $("#monitor-refresh-button").addEventListener("click", () =>
    refreshMonitor(true),
  );
  $("#monitor-log-refresh").addEventListener("click", () =>
    loadMonitorLog(true),
  );
  $("#monitor-follow-log").addEventListener("change", (event) => {
    setMonitorLogFollowing(event.target.checked, {
      scroll: event.target.checked,
    });
  });
  $("#monitor-log").addEventListener("scroll", (event) => {
    if (
      state.monitorLogUpdating ||
      !$("#monitor-follow-log").checked ||
      monitorLogNearBottom(event.currentTarget)
    ) {
      return;
    }
    setMonitorLogFollowing(false);
    $("#monitor-log-updated").textContent = "手动浏览 · 跟随已暂停";
  });
  $$(".run-filters button").forEach((button) => {
    button.addEventListener("click", () => {
      state.runFilter = button.dataset.runFilter;
      $$(".run-filters button").forEach((candidate) => {
        candidate.classList.toggle("active", candidate === button);
      });
      renderRunList();
    });
  });
  $("#monitor-stop-button").addEventListener("click", openStopDialog);
  $("#monitor-config-button").addEventListener(
    "click",
    openSelectedRunConfig,
  );
  $("#close-stop-dialog").addEventListener("click", () =>
    $("#stop-run-dialog").close(),
  );
  $("#cancel-stop-button").addEventListener("click", () =>
    $("#stop-run-dialog").close(),
  );
  $("#confirm-stop-button").addEventListener("click", confirmStopRun);
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
      requestJson("/api/runs?limit=100"),
    ]);
    state.health = health;
    state.schema = schema;
    state.profiles = profiles.profiles || [];
    state.runs = runs.runs || [];
    state.currentRun = state.runs[0] || null;
    state.selectedRunId =
      state.runs.find((run) => ACTIVE_RUN_STATUSES.has(run.status))?.run_id ||
      state.runs[0]?.run_id ||
      "";
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
    renderMonitor();
    setConsolePage(
      window.location.hash === "#monitor" ? "monitor" : "configuration",
    );
    window.setInterval(() => {
      if (document.hidden) return;
      if (state.currentPage === "monitor") {
        refreshMonitor(false);
      } else {
        loadRuns(false);
      }
    }, 3000);
    window.setInterval(() => {
      if (!document.hidden && state.currentPage === "monitor") {
        renderMonitorDetail();
      }
    }, 1000);
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
