const state = { health: null, catalogTimer: null };

const $ = (selector) => document.querySelector(selector);
const status = $("#status");
const statusText = $("#status-text");
const form = $("#query-form");
const queryInput = $("#query");
const submitButton = $("#submit-button");
const errorBox = $("#form-error");

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
    const values = [health.num_skills, health.num_paths, health.num_levels];
    document.querySelectorAll("#model-stats dd").forEach((node, index) => {
      node.textContent = Number(values[index]).toLocaleString();
    });
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
    const names = path.skills.map((skill) => skill.name || skill.skill_id).join(" · ");
    card.append(top, tokens, textNode("div", "decoded-names", names));
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
    skill.append(
      textNode("span", "skill-name", candidate.name || candidate.skill_id),
      textNode("span", "skill-id", candidate.skill_id),
    );
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
  $("#results").hidden = false;
  $("#result-panel").classList.remove("empty");
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
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    $(".button-label").textContent = "运行路由";
  }
}

function renderCatalog(payload) {
  $("#catalog-total").textContent = `${payload.total} results`;
  const container = $("#catalog-results");
  container.replaceChildren();
  payload.skills.slice(0, 12).forEach((skill) => {
    const item = textNode("article", "catalog-item");
    item.append(
      textNode("strong", "", skill.name || skill.skill_id),
      textNode("code", "", skill.code_text),
      textNode("p", "", skill.description || skill.capability_zh || skill.skill_id),
    );
    container.append(item);
  });
}

async function loadCatalog() {
  const query = $("#catalog-query").value.trim();
  try {
    renderCatalog(await jsonRequest(`/api/catalog?q=${encodeURIComponent(query)}&limit=12`));
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

loadHealth();
loadCatalog();
