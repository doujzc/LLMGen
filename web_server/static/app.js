const state = {
  health: null,
  catalogTimer: null,
  hasResult: false,
  skillCache: new Map(),
};

const $ = (selector) => document.querySelector(selector);
const status = $("#status");
const statusText = $("#status-text");
const form = $("#query-form");
const queryInput = $("#query");
const submitButton = $("#submit-button");
const errorBox = $("#form-error");
const skillDialog = $("#skill-dialog");

const exampleQuery =
  "我周五晚上到杭州，周日返程。帮我根据天气安排一条适合拍照的两日路线，订高铁和离景点近的酒店，把行程写进日历；如果下雨就调整成室内活动，并在每天出发前提醒我带对应物品。";

function textNode(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value ?? "";
  return node;
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
    const isRetrieval = health.supervision_phase === "retrieval";
    $("#supervision-stat-label").textContent = isRetrieval
      ? "有 Retrieval 监督"
      : "有训练目标监督";
    $("#supervision-filter-label").textContent = isRetrieval
      ? "仅看有 Retrieval 正样本的 Skills"
      : "仅看有训练目标样本的 Skills";
    const values = [
      health.num_skills,
      health.num_supervised_skills,
      health.num_paths,
      health.num_levels,
    ];
    document.querySelectorAll("#model-stats dd").forEach((node, index) => {
      node.textContent =
        values[index] === null || values[index] === undefined
          ? "未记录"
          : Number(values[index]).toLocaleString();
    });
    const supervisedOnly = $("#supervised-only");
    supervisedOnly.disabled = health.num_supervised_skills === null;
    if (supervisedOnly.disabled) supervisedOnly.checked = false;
    const maxPaths = $("#max-paths");
    maxPaths.replaceChildren();
    for (let value = 1; value <= health.max_code_paths; value += 1) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      option.selected = value === Math.min(4, health.max_code_paths);
      maxPaths.append(option);
    }
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

function skillButton(skill, className = "skill-link") {
  const button = textNode("button", className, skill.name || skill.skill_id);
  button.type = "button";
  button.dataset.skillId = skill.skill_id;
  button.addEventListener("click", () => openSkill(skill.skill_id));
  return button;
}

function renderPaths(paths) {
  const container = $("#paths");
  container.replaceChildren();
  paths.forEach((path, index) => {
    const card = textNode("article", "path-card");
    const top = textNode("div", "path-top");
    top.append(
      textNode("span", "path-number", `PATH ${String(index + 1).padStart(2, "0")}`),
      textNode("span", "path-score", Number(path.score).toFixed(4)),
    );
    const tokens = textNode("div", "tokens");
    path.code_tokens.forEach((token) => tokens.append(textNode("code", "token", token)));
    const names = textNode("div", "decoded-names");
    path.skills.forEach((skill) => names.append(skillButton(skill)));
    card.append(top, tokens, names);
    container.append(card);
  });
  $("#path-count").textContent = `${paths.length} path${paths.length === 1 ? "" : "s"}`;
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

function renderResult(result) {
  $("#empty-state").hidden = true;
  $("#loading-state").hidden = true;
  $("#results").hidden = false;
  $("#result-panel").classList.remove("empty");
  state.hasResult = true;
  $("#latency").textContent = `${Number(result.latency_ms).toLocaleString()} ms`;
  $("#generated-text").textContent = result.generated_text || "(empty)";
  renderPaths(result.paths);
  renderCandidates(result.candidates);
  $("#raw-json").textContent = JSON.stringify(result, null, 2);
}

async function runInference(event) {
  event.preventDefault();
  clearError();
  const query = queryInput.value.trim();
  if (!query) {
    showError("请输入测试 Query。");
    queryInput.focus();
    return;
  }
  $("#empty-state").hidden = true;
  $("#results").hidden = true;
  $("#loading-state").hidden = false;
  submitButton.disabled = true;
  $(".button-label").textContent = "推理中…";
  try {
    const result = await jsonRequest("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        max_code_paths: Number($("#max-paths").value),
        top_k: Number($("#top-k").value),
      }),
    });
    renderResult(result);
    if (window.innerWidth < 1000) $("#result-panel").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    $("#loading-state").hidden = true;
    $("#results").hidden = !state.hasResult;
    $("#empty-state").hidden = state.hasResult;
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    $(".button-label").textContent = "运行路由";
  }
}

function renderCatalog(payload) {
  $("#catalog-total").textContent = `${payload.total} results`;
  const supervisedOnly = $("#supervised-only");
  if (!payload.supervision_available) {
    supervisedOnly.checked = false;
    supervisedOnly.disabled = true;
  }
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
        skill.has_train_target ? "supervision-badge supervised" : "supervision-badge",
        supervisionStatus(skill),
      ),
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

function supervisionSampleLabel() {
  return state.health?.supervision_phase === "retrieval"
    ? "Retrieval 正样本"
    : "训练目标样本";
}

function supervisionStatus(skill) {
  if (skill.has_train_target === true) {
    return `${supervisionSampleLabel()} ${skill.train_target_count}`;
  }
  if (skill.has_train_target === false) {
    return `无${supervisionSampleLabel()}`;
  }
  return "监督状态未记录";
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
  if (skill.has_train_target === true) {
    tags.append(
      detailTag(`${supervisionSampleLabel()} · ${skill.train_target_count}`),
    );
  } else if (skill.has_train_target === false) {
    tags.append(detailTag(supervisionStatus(skill)));
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
  const supervisedOnly = $("#supervised-only").checked;
  try {
    renderCatalog(
      await jsonRequest(
        `/api/catalog?q=${encodeURIComponent(query)}&limit=12&supervised_only=${supervisedOnly}`,
      ),
    );
  } catch (error) {
    $("#catalog-total").textContent = "加载失败";
  }
}

$("#example-button").addEventListener("click", () => {
  queryInput.value = exampleQuery;
  queryInput.focus();
});
form.addEventListener("submit", runInference);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") form.requestSubmit();
});
$("#catalog-query").addEventListener("input", () => {
  window.clearTimeout(state.catalogTimer);
  state.catalogTimer = window.setTimeout(loadCatalog, 180);
});
$("#supervised-only").addEventListener("change", loadCatalog);
$("#dialog-close").addEventListener("click", () => skillDialog.close());
skillDialog.addEventListener("click", (event) => {
  if (event.target === skillDialog) skillDialog.close();
});

async function initialize() {
  await loadHealth();
  await loadCatalog();
}

initialize();
