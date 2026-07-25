const state = {
  health: null,
  catalogTimer: null,
  hasResult: false,
  skillCache: new Map(),
  inputMode: "single",
  batchRows: [],
  batchFileName: "",
  batchResults: [],
  batchRun: null,
  greedyMaxPaths: "4",
};

const $ = (selector) => document.querySelector(selector);
const status = $("#status");
const statusText = $("#status-text");
const form = $("#query-form");
const queryInput = $("#query");
const submitButton = $("#submit-button");
const errorBox = $("#form-error");
const skillDialog = $("#skill-dialog");
const decodingMode = $("#decoding-mode");
const numBeams = $("#num-beams");
const beamControl = $("#beam-control");
const maxPaths = $("#max-paths");
const maxPathsControl = $("#max-paths-control");
const batchFile = $("#batch-file");
const batchSize = $("#batch-size");

const exampleQuery =
  "我周五晚上到杭州，周日返程。帮我根据天气安排一条适合拍照的两日路线，订高铁和离景点近的酒店，把行程写进日历；如果下雨就调整成室内活动，并在每天出发前提醒我带对应物品。";

function textNode(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value ?? "";
  return node;
}

function updateDecodingControls() {
  const beamEnabled = decodingMode.value === "beam_search";
  if (beamEnabled) {
    if (!maxPaths.disabled) state.greedyMaxPaths = maxPaths.value;
    maxPaths.value = "1";
  } else if (maxPaths.disabled) {
    const available = Array.from(maxPaths.options, (option) => option.value);
    maxPaths.value = available.includes(state.greedyMaxPaths)
      ? state.greedyMaxPaths
      : available[0];
  }
  maxPaths.disabled = beamEnabled;
  maxPathsControl.classList.toggle("disabled", beamEnabled);
  numBeams.disabled = !beamEnabled;
  beamControl.classList.toggle("disabled", !beamEnabled);
}

function setInputMode(mode) {
  state.inputMode = mode;
  const isBatch = mode === "batch";
  $("#single-query-input").hidden = isBatch;
  $("#batch-query-input").hidden = !isBatch;
  $("#single-mode-button").classList.toggle("active", !isBatch);
  $("#batch-mode-button").classList.toggle("active", isBatch);
  $("#single-mode-button").setAttribute("aria-selected", String(!isBatch));
  $("#batch-mode-button").setAttribute("aria-selected", String(isBatch));
  $("#example-button").hidden = isBatch;
  queryInput.required = !isBatch;
  $(".button-label").textContent = isBatch ? "批量路由" : "运行路由";
  clearError();
}

function batchSizeOptions(maximum) {
  const values = [1, 2, 4, 8, 16, 32, 64].filter(
    (value) => value <= maximum,
  );
  if (!values.includes(maximum)) values.push(maximum);
  return values.sort((left, right) => left - right);
}

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`服务返回了无效响应（HTTP ${response.status}）`);
  }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function loadHealth() {
  try {
    const health = await jsonRequest("/api/health");
    state.health = health;
    status.className = "status ready";
    statusText.textContent = `${health.device} · 模型已就绪`;
    const values = [
      health.num_skills,
      health.num_paths,
      health.num_levels,
    ];
    document.querySelectorAll("#model-stats dd").forEach((node, index) => {
      node.textContent =
        values[index] === null || values[index] === undefined
          ? "未记录"
          : Number(values[index]).toLocaleString();
    });
    maxPaths.replaceChildren();
    for (let value = 1; value <= health.max_code_paths; value += 1) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      option.selected = value === Math.min(4, health.max_code_paths);
      maxPaths.append(option);
    }
    const maxNumBeams = Math.max(2, Number(health.max_num_beams || 8));
    const beamValues = [2, 4, 8, 16, 32, 64].filter(
      (value) => value <= maxNumBeams,
    );
    if (!beamValues.includes(maxNumBeams)) beamValues.push(maxNumBeams);
    beamValues.sort((left, right) => left - right);
    numBeams.replaceChildren();
    beamValues.forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      option.selected = value === (beamValues.includes(4) ? 4 : beamValues[0]);
      numBeams.append(option);
    });
    const maxBatchSize = Math.max(1, Number(health.max_batch_size || 8));
    const availableBatchSizes = batchSizeOptions(maxBatchSize);
    batchSize.replaceChildren();
    availableBatchSizes.forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      option.selected =
        value === (availableBatchSizes.includes(2) ? 2 : availableBatchSizes[0]);
      batchSize.append(option);
    });
    updateDecodingControls();
  } catch (error) {
    status.className = "status failed";
    statusText.textContent = "模型连接失败";
    showError(error.message);
  }
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function parseQueryText(text) {
  const rows = [];
  text.replace(/^\uFEFF/, "").split(/\r?\n/).forEach((rawLine, index) => {
    const query = rawLine.trim();
    if (!query) return;
    if (query.length > 20000) {
      throw new Error(`第 ${index + 1} 行超过 20,000 个字符。`);
    }
    rows.push({ query, source_line: index + 1 });
  });
  return rows;
}

function renderBatchFile() {
  $("#batch-file-meta").hidden = false;
  $("#batch-file-name").textContent = state.batchFileName;
  $("#batch-query-count").textContent = `${state.batchRows.length} queries`;
  const preview = $("#batch-preview");
  preview.replaceChildren();
  state.batchRows.slice(0, 5).forEach((row) => {
    preview.append(textNode("li", "", `L${row.source_line} · ${row.query}`));
  });
  if (state.batchRows.length > 5) {
    preview.append(
      textNode("li", "", `… 其余 ${state.batchRows.length - 5} 条`),
    );
  }
}

async function loadBatchFile() {
  clearError();
  const file = batchFile.files?.[0];
  if (!file) return;
  try {
    const rows = parseQueryText(await file.text());
    if (!rows.length) throw new Error("TXT 中没有非空 Query。");
    const maximum = Number(state.health?.max_batch_queries || 1000);
    if (rows.length > maximum) {
      throw new Error(`TXT 包含 ${rows.length} 条 Query，服务上限为 ${maximum} 条。`);
    }
    state.batchRows = rows;
    state.batchFileName = file.name;
    renderBatchFile();
  } catch (error) {
    state.batchRows = [];
    state.batchFileName = "";
    $("#batch-file-meta").hidden = true;
    showError(error.message);
  }
}

function skillButton(skill, className = "skill-link") {
  const button = textNode("button", className, skill.name || skill.skill_id);
  button.type = "button";
  button.dataset.skillId = skill.skill_id;
  button.addEventListener("click", () => openSkill(skill.skill_id));
  return button;
}

function renderPaths(paths, beamMode = false) {
  const container = $("#paths");
  container.replaceChildren();
  paths.forEach((path, index) => {
    const card = textNode("article", "path-card");
    const top = textNode("div", "path-top");
    top.append(
      textNode(
        "span",
        "path-number",
        `${beamMode ? "CODE" : "PATH"} ${String(index + 1).padStart(2, "0")}`,
      ),
      textNode("span", "path-score", Number(path.score).toFixed(4)),
    );
    const tokens = textNode("div", "tokens");
    path.code_tokens.forEach((token) => tokens.append(textNode("code", "token", token)));
    const names = textNode("div", "decoded-names");
    path.skills.forEach((skill) => names.append(skillButton(skill)));
    card.append(top, tokens, names);
    container.append(card);
  });
  const unit = beamMode ? "code" : "path";
  $("#path-count").textContent =
    `${paths.length} ${unit}${paths.length === 1 ? "" : "s"}`;
}

function renderCandidates(candidates) {
  const body = $("#candidate-rows");
  body.replaceChildren();
  candidates.forEach((candidate, index) => {
    const row = document.createElement("tr");
    row.append(textNode("td", "rank", String(index + 1).padStart(2, "0")));
    const skill = document.createElement("td");
    skill.append(skillButton(candidate, "skill-name skill-link"));
    skill.append(textNode("span", "skill-id", candidate.skill_id));
    const domain = document.createElement("td");
    domain.append(textNode("span", "domain-tag", candidate.domain || "未分类"));
    row.append(
      skill,
      domain,
      textNode("td", "code-cell", candidate.code_text),
      textNode("td", "score-cell", Number(candidate.score).toFixed(4)),
    );
    body.append(row);
  });
}

function generationOptions() {
  const mode = decodingMode.value;
  return {
    max_code_paths: mode === "beam_search" ? 1 : Number(maxPaths.value),
    top_k: Number($("#top-k").value),
    decoding_mode: mode,
    num_beams: mode === "beam_search" ? Number(numBeams.value) : 1,
  };
}

function renderResult(result, { batch = false } = {}) {
  $("#empty-state").hidden = true;
  $("#loading-state").hidden = true;
  $("#results").hidden = false;
  $("#result-panel").classList.remove("empty");
  state.hasResult = true;
  const request = batch ? state.batchRun?.request : result.request;
  const latency = batch ? state.batchRun?.latency_ms : result.latency_ms;
  $("#latency-label").textContent = batch ? "批量总时延" : "端到端时延";
  $("#latency").textContent =
    latency === undefined ? "—" : `${Number(latency).toLocaleString()} ms`;
  const mode = request?.decoding_mode || result.decoding?.mode || "greedy";
  const beamWidth = request?.num_beams || result.decoding?.num_beams || 1;
  const beamMode = mode === "beam_search";
  $("#decode-summary").textContent =
    beamMode ? `Beam Search · Top ${beamWidth} codes` : "Greedy";
  $("#raw-output-label").textContent = beamMode
    ? "TOP-K SINGLE-LINE CODE ALTERNATIVES"
    : "RAW AUTOREGRESSIVE OUTPUT";
  $("#paths-heading").textContent = beamMode ? "Code 候选" : "生成路径";
  $("#candidate-hint").textContent = beamMode
    ? "按 Beam 的 Code 排名展开为 Skill"
    : "同一路径内按候选原始顺序展开";
  $("#result-query").textContent = result.query;
  $("#generated-text").textContent = result.generated_text || "(empty)";
  renderPaths(result.paths, beamMode);
  renderCandidates(result.candidates);
  $("#raw-json").textContent = JSON.stringify(result, null, 2);
}

function renderSingleResult(result) {
  state.batchResults = [];
  state.batchRun = null;
  $("#batch-toolbar").hidden = true;
  renderResult(result);
}

function selectBatchResult(index) {
  const result = state.batchResults[index];
  if (!result) return;
  $("#batch-result-select").value = String(index);
  renderResult(result, { batch: true });
}

function renderBatchRun(run) {
  state.batchRun = run;
  state.batchResults = run.results;
  const selector = $("#batch-result-select");
  selector.replaceChildren();
  run.results.forEach((result, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    const preview =
      result.query.length > 52 ? `${result.query.slice(0, 52)}…` : result.query;
    option.textContent =
      `${String(index + 1).padStart(3, "0")} · ` +
      `L${result.source_line ?? index + 1} · ${preview}`;
    selector.append(option);
  });
  $("#batch-summary").textContent =
    `${run.file_name} · ${run.num_queries} queries`;
  $("#batch-throughput").textContent =
    `${Number(run.latency_ms).toLocaleString()} ms · ` +
    `${Number(run.queries_per_second).toLocaleString()} query/s · ` +
    `batch size ${run.request.batch_size}`;
  $("#batch-toolbar").hidden = false;
  selectBatchResult(0);
}

function downloadBatchResults() {
  if (!state.batchResults.length) return;
  const jsonl = `${state.batchResults.map((row) => JSON.stringify(row)).join("\n")}\n`;
  const blobUrl = URL.createObjectURL(
    new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" }),
  );
  const fileStem = (state.batchRun?.file_name || "queries")
    .replace(/\.[^.]+$/, "")
    .replace(/[^\w.-]+/g, "_");
  const link = document.createElement("a");
  link.href = blobUrl;
  link.download = `${fileStem}.results.jsonl`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
}

function showLoading(message) {
  $("#empty-state").hidden = true;
  $("#results").hidden = true;
  $("#loading-state").hidden = false;
  $("#loading-copy").textContent = message;
}

async function runSingleInference(options) {
  const query = queryInput.value.trim();
  if (!query) {
    showError("请输入测试 Query。");
    queryInput.focus();
    return false;
  }
  showLoading(
    options.decoding_mode === "beam_search"
      ? `正在搜索单行 Skill Code 的 Top ${options.num_beams} 候选…`
      : "正在约束生成多条 Skill Code 并解码候选，请稍候。",
  );
  const result = await jsonRequest("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...options }),
  });
  renderSingleResult(result);
  return true;
}

async function runBatchInference(options) {
  if (!state.batchRows.length) {
    showError("请选择包含 Query 的 TXT 文件。");
    return false;
  }
  const rows = state.batchRows.map((row) => ({ ...row }));
  const fileName = state.batchFileName;
  const size = Number(batchSize.value);
  const results = [];
  let serverLatencyMs = 0;
  const started = performance.now();
  for (let start = 0; start < rows.length; start += size) {
    const end = Math.min(start + size, rows.length);
    const decodingLabel =
      options.decoding_mode === "beam_search"
        ? `单行 Code Top ${options.num_beams}`
        : "多路径 Greedy";
    showLoading(
      `正在处理 ${start + 1}–${end} / ${rows.length} 条 Query · ${decodingLabel}…`,
    );
    const response = await jsonRequest("/api/infer-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries: rows.slice(start, end).map((row) => row.query),
        batch_size: size,
        ...options,
      }),
    });
    serverLatencyMs += Number(response.latency_ms);
    response.results.forEach((result, localIndex) => {
      const globalIndex = start + localIndex;
      result.query_id = `query-${String(globalIndex + 1).padStart(6, "0")}`;
      result.batch_index = globalIndex;
      result.source_line = rows[globalIndex].source_line;
      results.push(result);
    });
  }
  const latencyMs = Math.round((performance.now() - started) * 100) / 100;
  renderBatchRun({
    file_name: fileName,
    num_queries: results.length,
    latency_ms: latencyMs,
    server_latency_ms: Math.round(serverLatencyMs * 100) / 100,
    queries_per_second:
      Math.round((results.length / Math.max(latencyMs / 1000, 1e-9)) * 1000) /
      1000,
    request: { ...options, batch_size: size },
    results,
  });
  return true;
}

async function runInference(event) {
  event.preventDefault();
  clearError();
  submitButton.disabled = true;
  $(".button-label").textContent = "推理中…";
  try {
    const completed =
      state.inputMode === "batch"
        ? await runBatchInference(generationOptions())
        : await runSingleInference(generationOptions());
    if (completed && window.innerWidth < 1000) {
      $("#result-panel").scrollIntoView({ behavior: "smooth" });
    }
  } catch (error) {
    $("#loading-state").hidden = true;
    $("#results").hidden = !state.hasResult;
    $("#empty-state").hidden = state.hasResult;
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    $(".button-label").textContent =
      state.inputMode === "batch" ? "批量路由" : "运行路由";
  }
}

function renderCatalog(payload) {
  $("#catalog-total").textContent = `${payload.total} results`;
  const container = $("#catalog-results");
  container.replaceChildren();
  payload.skills.slice(0, 12).forEach((skill) => {
    const item = textNode("button", "catalog-item");
    item.type = "button";
    item.addEventListener("click", () => openSkill(skill.skill_id));
    item.append(
      textNode("strong", "", skill.name || skill.skill_id),
      textNode("code", "", skill.code_text),
      textNode(
        "span",
        "catalog-description",
        skill.description || skill.capability_zh || skill.skill_id,
      ),
    );
    container.append(item);
  });
}

function detailTag(value) {
  return textNode("span", "detail-tag", value);
}

function renderSkillDetail(skill) {
  $("#detail-name").textContent = skill.name || skill.skill_id;
  $("#detail-id").textContent = skill.skill_id;
  $("#detail-capability").textContent =
    skill.capability_zh || skill.description || "暂无能力说明";
  $("#detail-description").textContent = skill.description || "暂无原始描述";
  $("#detail-text").textContent =
    skill.text || skill.description || skill.capability_zh || "暂无候选文本";
  $("#detail-code").textContent = skill.code_text || "";

  const tags = $("#detail-tags");
  tags.replaceChildren();
  if (skill.domain) tags.append(detailTag(`领域 · ${skill.domain}`));
  if (skill.mobile_fit) tags.append(detailTag(`手机适配 · ${skill.mobile_fit}`));
  if (skill.rank !== undefined && skill.rank !== null) {
    tags.append(detailTag(`ClawHub 排名 · ${skill.rank}`));
  }
  (skill.roles || []).forEach((role) => tags.append(detailTag(`role · ${role}`)));

  const source = $("#detail-source");
  source.hidden = !skill.source_url;
  if (skill.source_url) source.href = skill.source_url;
}

async function openSkill(skillId) {
  try {
    let skill = state.skillCache.get(skillId);
    if (!skill) {
      skill = await jsonRequest(`/api/skill?id=${encodeURIComponent(skillId)}`);
      state.skillCache.set(skillId, skill);
    }
    renderSkillDetail(skill);
    if (!skillDialog.open) skillDialog.showModal();
  } catch (error) {
    showError(`无法加载 Skill 详情：${error.message}`);
  }
}

async function loadCatalog() {
  const query = $("#catalog-query").value.trim();
  try {
    renderCatalog(
      await jsonRequest(`/api/catalog?q=${encodeURIComponent(query)}&limit=12`),
    );
  } catch (error) {
    $("#catalog-total").textContent = "加载失败";
  }
}

$("#example-button").addEventListener("click", () => {
  queryInput.value = exampleQuery;
  queryInput.focus();
});
$("#single-mode-button").addEventListener("click", () => setInputMode("single"));
$("#batch-mode-button").addEventListener("click", () => setInputMode("batch"));
batchFile.addEventListener("change", loadBatchFile);
form.addEventListener("submit", runInference);
decodingMode.addEventListener("change", updateDecodingControls);
$("#batch-result-select").addEventListener("change", (event) => {
  selectBatchResult(Number(event.target.value));
});
$("#download-batch").addEventListener("click", downloadBatchResults);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") form.requestSubmit();
});
$("#catalog-query").addEventListener("input", () => {
  window.clearTimeout(state.catalogTimer);
  state.catalogTimer = window.setTimeout(loadCatalog, 180);
});
$("#dialog-close").addEventListener("click", () => skillDialog.close());
skillDialog.addEventListener("click", (event) => {
  if (event.target === skillDialog) skillDialog.close();
});

async function initialize() {
  await loadHealth();
  await loadCatalog();
}

initialize();
