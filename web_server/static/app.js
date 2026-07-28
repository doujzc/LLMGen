const state = {
  health: null,
  catalogTimer: null,
  fullCatalogTimer: null,
  hasResult: false,
  skillCache: new Map(),
  inputMode: "single",
  batchRows: [],
  batchFileName: "",
  batchResults: [],
  batchRun: null,
  greedyMaxPaths: "4",
  allSkills: [],
  allSkillsLoaded: false,
  catalogCodePrefix: [],
  selectedMainSkillId: "",
  selectedCatalogSkillId: "",
};

const $ = (selector) => document.querySelector(selector);
const status = $("#status");
const statusText = $("#status-text");
const form = $("#query-form");
const queryInput = $("#query");
const submitButton = $("#submit-button");
const errorBox = $("#form-error");
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

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(4) : "—";
}

function skillTokens(skill) {
  if (Array.isArray(skill.tokens)) return skill.tokens.map(String);
  return String(skill.code_text || "").match(/<[^>]+>/g) || [];
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

function detailTag(value) {
  return textNode("span", "detail-tag", value);
}

function renderSkillDetail(skill, detailSelector, emptySelector) {
  const detail = $(detailSelector);
  const empty = $(emptySelector);
  empty.hidden = true;
  detail.hidden = false;
  detail.querySelector('[data-detail="name"]').textContent =
    skill.name || skill.skill_id;
  detail.querySelector('[data-detail="id"]').textContent = skill.skill_id;
  detail.querySelector('[data-detail="capability"]').textContent =
    skill.capability_zh || skill.description || "暂无能力说明";
  detail.querySelector('[data-detail="description"]').textContent =
    skill.description || "暂无原始描述";
  detail.querySelector('[data-detail="code"]').textContent =
    skill.code_text || "";

  const candidateText = detail.querySelector('[data-detail="text"]');
  if (candidateText) {
    candidateText.textContent =
      skill.text || skill.description || skill.capability_zh || "暂无候选文本";
  }

  const tags = detail.querySelector('[data-detail="tags"]');
  tags.replaceChildren();
  if (skill.domain) tags.append(detailTag(`领域 · ${skill.domain}`));
  if (skill.mobile_fit) tags.append(detailTag(`手机适配 · ${skill.mobile_fit}`));
  if (skill.rank !== undefined && skill.rank !== null) {
    tags.append(detailTag(`ClawHub 排名 · ${skill.rank}`));
  }
  (skill.roles || []).forEach((role) => tags.append(detailTag(`role · ${role}`)));

  const source = detail.querySelector('[data-detail="source"]');
  source.hidden = !skill.source_url;
  if (skill.source_url) source.href = skill.source_url;
}

async function resolveSkill(skillOrId) {
  if (typeof skillOrId === "object" && skillOrId !== null) {
    state.skillCache.set(skillOrId.skill_id, skillOrId);
    return skillOrId;
  }
  const skillId = String(skillOrId);
  let skill = state.skillCache.get(skillId);
  if (!skill) {
    skill = await jsonRequest(`/api/skill?id=${encodeURIComponent(skillId)}`);
    state.skillCache.set(skillId, skill);
  }
  return skill;
}

function updateSelectedSkillRows(skillId, context) {
  const selector =
    context === "catalog"
      ? "#all-skill-results .all-skill-row"
      : ".candidate-row, .catalog-item";
  document.querySelectorAll(selector).forEach((node) => {
    node.classList.toggle("selected", node.dataset.skillId === skillId);
  });
}

async function showSkill(skillOrId, context = "main") {
  try {
    const skill = await resolveSkill(skillOrId);
    if (context === "catalog") {
      state.selectedCatalogSkillId = skill.skill_id;
      renderSkillDetail(skill, "#catalog-detail", "#catalog-detail-empty");
    } else {
      state.selectedMainSkillId = skill.skill_id;
      renderSkillDetail(skill, "#main-detail", "#main-detail-empty");
    }
    updateSelectedSkillRows(skill.skill_id, context);
  } catch (error) {
    showError(`无法加载 Skill 详情：${error.message}`);
  }
}

function renderCandidates(candidates) {
  const container = $("#candidate-rows");
  container.replaceChildren();
  $("#candidate-count").textContent =
    `${candidates.length} candidate${candidates.length === 1 ? "" : "s"}`;

  if (!candidates.length) {
    container.append(textNode("p", "empty-list", "没有解码出候选 Skill。"));
    return;
  }

  candidates.forEach((candidate, index) => {
    state.skillCache.set(candidate.skill_id, candidate);
    const row = textNode("button", "candidate-row");
    row.type = "button";
    row.dataset.skillId = candidate.skill_id;
    row.setAttribute(
      "aria-label",
      `查看 ${candidate.name || candidate.skill_id} 详情`,
    );

    const copy = textNode("span", "candidate-copy");
    const nameLine = textNode("span", "candidate-name-line");
    nameLine.append(
      textNode("strong", "candidate-name", candidate.name || candidate.skill_id),
    );
    const meta = textNode("span", "candidate-meta");
    meta.append(
      textNode("span", "candidate-domain", candidate.domain || "未分类"),
    );
    const codePath = textNode("span", "candidate-code-path");
    const tokens = skillTokens(candidate);
    if (tokens.length) {
      tokens.forEach((token) => {
        codePath.append(textNode("code", "candidate-token", token));
      });
    } else {
      codePath.append(
        textNode("code", "candidate-token", candidate.code_text || "NO CODE"),
      );
    }
    meta.append(codePath);
    copy.append(
      nameLine,
      textNode(
        "span",
        "candidate-description",
        candidate.capability_zh ||
          candidate.description ||
          candidate.skill_id,
      ),
      meta,
    );
    const score = textNode("span", "candidate-score-block");
    score.append(
      textNode("strong", "candidate-score", formatScore(candidate.score)),
      textNode("small", "candidate-score-label", "model score"),
    );
    row.append(
      textNode("span", "candidate-rank", String(index + 1).padStart(2, "0")),
      copy,
      score,
    );
    row.addEventListener("click", () => showSkill(candidate, "main"));
    container.append(row);
  });

  showSkill(candidates[0], "main");
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
  $("#decode-summary").textContent = beamMode
    ? `Beam Search · Top ${beamWidth} codes`
    : "Greedy Autoregressive";
  $("#candidate-hint").textContent = beamMode
    ? "按 Beam 概率排序，Code 路径随候选一并展示"
    : "按生成顺序与模型得分展开，Code 路径随候选一并展示";
  const candidateCount = (result.candidates || []).length;
  $("#result-subtitle").textContent =
    `${candidateCount} 个候选 Skill · 点击候选查看详情`;
  renderCandidates(result.candidates || []);
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
      : "正在进行 Greedy Autoregressive 路由并解码候选 Skill…",
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
        : "Greedy Autoregressive";
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
    if (completed && window.innerWidth < 720) {
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
  $("#catalog-total").textContent = `${payload.total}`;
  const container = $("#catalog-results");
  container.replaceChildren();
  payload.skills.slice(0, 8).forEach((skill) => {
    state.skillCache.set(skill.skill_id, skill);
    const item = textNode("button", "catalog-item");
    item.type = "button";
    item.dataset.skillId = skill.skill_id;
    item.addEventListener("click", () => showSkill(skill, "main"));
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
  if (!payload.skills.length) {
    container.append(textNode("p", "empty-list", "没有匹配的 Skill。"));
  }
  if (state.selectedMainSkillId) {
    updateSelectedSkillRows(state.selectedMainSkillId, "main");
  }
}

async function loadCatalog() {
  const query = $("#catalog-query").value.trim();
  try {
    renderCatalog(
      await jsonRequest(`/api/catalog?q=${encodeURIComponent(query)}&limit=8`),
    );
  } catch {
    $("#catalog-total").textContent = "加载失败";
  }
}

function naturalTokenSort(left, right) {
  return left.localeCompare(right, "zh-CN", {
    numeric: true,
    sensitivity: "base",
  });
}

function buildTree(skills) {
  const root = { count: 0, children: new Map() };
  skills.forEach((skill) => {
    const tokens = skillTokens(skill);
    root.count += 1;
    let node = root;
    tokens.forEach((token) => {
      if (!node.children.has(token)) {
        node.children.set(token, { count: 0, children: new Map() });
      }
      node = node.children.get(token);
      node.count += 1;
    });
  });
  return root;
}

function samePrefix(left, right) {
  return left.length === right.length &&
    left.every((token, index) => token === right[index]);
}

function selectedBranch(prefix) {
  return prefix.every(
    (token, index) => state.catalogCodePrefix[index] === token,
  );
}

function setCatalogCodePrefix(prefix) {
  state.catalogCodePrefix = [...prefix];
  renderCodeTree();
  renderAllSkills();
}

function appendTreeChildren(container, node, prefix, level = 0) {
  const entries = [...node.children.entries()].sort(([left], [right]) =>
    naturalTokenSort(left, right),
  );
  entries.forEach(([token, child], index) => {
    const nextPrefix = [...prefix, token];
    if (child.children.size) {
      const details = document.createElement("details");
      details.open =
        selectedBranch(nextPrefix) &&
        (state.catalogCodePrefix.length > nextPrefix.length || level === 0) ||
        (state.catalogCodePrefix.length === 0 && level === 0 && index < 3);
      const summary = document.createElement("summary");
      summary.classList.toggle(
        "selected",
        samePrefix(state.catalogCodePrefix, nextPrefix),
      );
      summary.append(
        textNode("code", "tree-token", token),
        textNode("span", "tree-count", String(child.count)),
      );
      summary.addEventListener("click", () => {
        window.setTimeout(() => setCatalogCodePrefix(nextPrefix), 0);
      });
      details.append(summary);
      const children = textNode("div", "tree-children");
      appendTreeChildren(children, child, nextPrefix, level + 1);
      details.append(children);
      container.append(details);
      return;
    }

    const button = textNode("button", "tree-node-button");
    button.type = "button";
    button.classList.toggle(
      "selected",
      samePrefix(state.catalogCodePrefix, nextPrefix),
    );
    button.append(
      textNode("code", "tree-token", token),
      textNode("span", "tree-count", String(child.count)),
    );
    button.addEventListener("click", () => setCatalogCodePrefix(nextPrefix));
    container.append(button);
  });
}

function renderCodeTree() {
  const container = $("#code-tree");
  container.replaceChildren();
  const tree = buildTree(state.allSkills);
  const rootButton = textNode("button", "tree-root");
  rootButton.type = "button";
  rootButton.classList.toggle("selected", !state.catalogCodePrefix.length);
  rootButton.append(
    textNode("span", "", "全部 Code 路径"),
    textNode("span", "tree-count", String(tree.count)),
  );
  rootButton.addEventListener("click", () => setCatalogCodePrefix([]));
  container.append(rootButton);
  appendTreeChildren(container, tree, []);
  $("#clear-code-filter").hidden = !state.catalogCodePrefix.length;
}

function catalogSearchText(skill) {
  return [
    skill.skill_id,
    skill.name,
    skill.description,
    skill.capability_zh,
    skill.domain,
    skill.code_text,
  ]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase();
}

function filteredCatalogSkills() {
  const needle = $("#catalog-page-query").value.trim().toLocaleLowerCase();
  return state.allSkills.filter((skill) => {
    const tokens = skillTokens(skill);
    const codeMatch = state.catalogCodePrefix.every(
      (token, index) => tokens[index] === token,
    );
    const textMatch = !needle || catalogSearchText(skill).includes(needle);
    return codeMatch && textMatch;
  });
}

function renderAllSkills() {
  const skills = filteredCatalogSkills();
  const container = $("#all-skill-results");
  container.replaceChildren();
  $("#filtered-skill-count").textContent =
    `${skills.length.toLocaleString()} Skills`;
  $("#catalog-page-total").textContent =
    `${skills.length.toLocaleString()} / ${state.allSkills.length.toLocaleString()}`;
  $("#active-code-filter").textContent = state.catalogCodePrefix.length
    ? state.catalogCodePrefix.join(" → ")
    : "全部 Code 路径";

  if (!skills.length) {
    container.append(textNode("p", "empty-list", "没有匹配当前筛选的 Skill。"));
    return;
  }

  const fragment = document.createDocumentFragment();
  skills.forEach((skill) => {
    state.skillCache.set(skill.skill_id, skill);
    const row = textNode("button", "all-skill-row");
    row.type = "button";
    row.dataset.skillId = skill.skill_id;
    row.classList.toggle(
      "selected",
      state.selectedCatalogSkillId === skill.skill_id,
    );
    const copy = textNode("span", "all-skill-copy");
    copy.append(
      textNode("strong", "", skill.name || skill.skill_id),
      textNode(
        "span",
        "",
        skill.capability_zh || skill.description || skill.skill_id,
      ),
    );
    row.append(
      copy,
      textNode("span", "all-skill-domain", skill.domain || "未分类"),
      textNode("code", "all-skill-code", skill.code_text),
    );
    row.addEventListener("click", () => showSkill(skill, "catalog"));
    fragment.append(row);
  });
  container.append(fragment);
}

async function loadFullCatalog() {
  if (state.allSkillsLoaded) return;
  const expected = Math.max(1, Number(state.health?.num_skills || 1000));
  const payload = await jsonRequest(`/api/catalog?q=&limit=${expected}`);
  state.allSkills = payload.skills;
  state.allSkillsLoaded = true;
  payload.skills.forEach((skill) => state.skillCache.set(skill.skill_id, skill));
}

async function openCatalogPage() {
  clearError();
  const railQuery = $("#catalog-query").value.trim();
  $("#catalog-page-query").value = railQuery;
  $("#router-view").hidden = true;
  $("#catalog-page").hidden = false;
  window.scrollTo({ top: 0, behavior: "instant" });
  try {
    await loadFullCatalog();
    renderCodeTree();
    renderAllSkills();
  } catch (error) {
    $("#all-skill-results").replaceChildren(
      textNode("p", "empty-list", `无法加载候选集：${error.message}`),
    );
  }
}

function closeCatalogPage() {
  $("#catalog-page").hidden = true;
  $("#router-view").hidden = false;
  $("#open-catalog-page").focus();
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
  if (event.key === "Escape" && !$("#catalog-page").hidden) {
    closeCatalogPage();
    return;
  }
  if (
    (event.metaKey || event.ctrlKey) &&
    event.key === "Enter" &&
    $("#catalog-page").hidden
  ) {
    form.requestSubmit();
  }
});
$("#catalog-query").addEventListener("input", () => {
  window.clearTimeout(state.catalogTimer);
  state.catalogTimer = window.setTimeout(loadCatalog, 180);
});
$("#open-catalog-page").addEventListener("click", openCatalogPage);
$("#close-catalog-page").addEventListener("click", closeCatalogPage);
$("#clear-code-filter").addEventListener("click", () => setCatalogCodePrefix([]));
$("#catalog-page-query").addEventListener("input", () => {
  window.clearTimeout(state.fullCatalogTimer);
  state.fullCatalogTimer = window.setTimeout(renderAllSkills, 120);
});

async function initialize() {
  await loadHealth();
  await loadCatalog();
}

initialize();
