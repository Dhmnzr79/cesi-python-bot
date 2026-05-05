function q(id) {
  return document.getElementById(id);
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function getJson(path, clientId) {
  const sep = path.includes("?") ? "&" : "?";
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 6000);
  let r;
  try {
    r = await fetch(`${path}${sep}client_id=${encodeURIComponent(clientId)}`, {
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(t);
  }
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${path} -> ${r.status} ${txt.slice(0, 200)}`);
  }
  return r.json();
}

function setError(id, e) {
  q(id).innerHTML = `<div class="muted">Ошибка: ${esc(String(e))}</div>`;
}

function renderOverview(data) {
  const kpis = [
    ["Сообщений", data.user_turns ?? 0, ""],
    ["Диалогов", data.conversations ?? 0, ""],
    ["Лидов", data.leads ?? 0, "good"],
    ["Конверсия в лид", `${(data.conversion_percent ?? 0).toFixed(1)}%`, "good"],
    ["Ошибок", data.errors ?? 0, (data.errors ?? 0) > 0 ? "bad" : "good"],
    ["Fallback/no_candidates", data.fallbacks_total ?? 0, (data.fallbacks_total ?? 0) > 0 ? "warn" : "good"],
    ["OpenAI USD", `$${(data.estimated_usd ?? 0).toFixed(4)}`, ""],
    ["Средняя latency", `${Math.round(data.avg_latency_ms ?? 0)} ms`, ""],
  ];
  q("overview").innerHTML = kpis
    .map(([label, value, tone]) => `<div class="kpi"><span class="kpi-label">${esc(label)}</span><span class="kpi-value ${tone}">${esc(value)}</span></div>`)
    .join("");
}

function renderCosts(data) {
  const rows = data.items || [];
  if (!rows.length) {
    q("costs").innerHTML = `<div class="muted">Сегодня вызовов LLM пока нет.</div>`;
    return;
  }
  const body = rows
    .map((r) => `<tr><td>${esc(r.call_type)}</td><td>${esc(r.calls)}</td><td>${esc(r.prompt_tokens)}</td><td>${esc(r.completion_tokens)}</td><td>$${Number(r.estimated_usd || 0).toFixed(4)}</td></tr>`)
    .join("");
  q("costs").innerHTML = `
    <div class="qa-line"><strong>Итого:</strong> $${Number(data.estimated_usd_total || 0).toFixed(4)}</div>
    <table class="simple-table">
      <thead><tr><th>Тип</th><th>Вызовы</th><th>Prompt</th><th>Completion</th><th>USD</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderDialogs(data) {
  const rows = data.items || [];
  if (!rows.length) {
    q("dialogs").innerHTML = `<div class="muted">Диалогов пока нет.</div>`;
    return;
  }
  q("dialogs").innerHTML = `<div class="list">${rows.map((r) => {
    const status = r.status || "ok";
    return `
      <div class="row-card">
        <div class="row-head">
          <span>${esc(r.last_ts || "-")} · ${esc(r.sid || "-")} · turns: ${esc(r.turn_number || r.turns || 0)}</span>
          <span class="tag ${esc(status)}">${esc(status)}</span>
        </div>
        <div class="qa-line"><strong>Вопрос:</strong> ${esc(r.last_user_text || "-")}</div>
        <div class="qa-line"><strong>Ответ:</strong> ${esc(r.last_bot_text || "-")}</div>
        <div class="qa-line muted">route: ${esc(r.last_route || "-")} · lead: ${r.has_lead ? "yes" : "no"}</div>
      </div>
    `;
  }).join("")}</div>`;
}

function renderProblems(data) {
  const rows = data.items || [];
  if (!rows.length) {
    q("problems").innerHTML = `<div class="muted">Критичных проблем за период не найдено.</div>`;
    return;
  }
  const body = rows.map((r) => `
    <tr>
      <td>${esc(r.ts || "-")}</td>
      <td>${esc(r.priority || "-")}</td>
      <td>${esc(r.sid || "-")}</td>
      <td>${esc(r.user_text || "-")}</td>
      <td>${esc(r.route || "-")}</td>
      <td>${esc(r.reason || "-")}</td>
      <td>${esc(r.doc_id || "-")}</td>
    </tr>
  `).join("");
  q("problems").innerHTML = `
    <table class="simple-table">
      <thead><tr><th>Время</th><th>Приоритет</th><th>SID</th><th>Вопрос</th><th>Route</th><th>Причина</th><th>Doc</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function maskPhone(phone) {
  const p = String(phone || "").trim();
  if (!p) return "-";
  if (p.length <= 4) return "***";
  return `${p.slice(0, 2)}***${p.slice(-2)}`;
}

function renderLeads(data) {
  const rows = data.items || [];
  if (!rows.length) {
    q("leads").innerHTML = `<div class="muted">Лидов пока нет.</div>`;
    return;
  }
  const body = rows.map((r) => `
    <tr>
      <td>${esc(r.captured_at || "-")}</td>
      <td>${esc(r.name || "-")}</td>
      <td>${esc(maskPhone(r.phone))}</td>
      <td>${esc(r.topic || "-")}</td>
      <td>${esc(r.sid || "-")}</td>
      <td>${esc(r.delivery_status || "-")}</td>
      <td>${esc(r.client_id || "-")}</td>
    </tr>
  `).join("");
  q("leads").innerHTML = `
    <table class="simple-table">
      <thead><tr><th>Время</th><th>Имя</th><th>Телефон</th><th>Тема</th><th>SID</th><th>Доставка</th><th>client_id</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderEvents(data) {
  q("events").textContent = JSON.stringify(data, null, 2);
}

function readDevFilters() {
  const p = new URLSearchParams();
  const eventType = (q("eventTypeFilter").value || "").trim();
  const sid = (q("sidFilter").value || "").trim();
  const requestId = (q("requestIdFilter").value || "").trim();
  if (eventType) p.set("event_type", eventType);
  if (sid) p.set("sid", sid);
  if (requestId) p.set("request_id", requestId);
  p.set("limit", "100");
  return p.toString();
}

async function refreshAll() {
  const clientId = (q("clientId").value || "default").trim() || "default";
  q("overview").innerHTML = `<div class="kpi"><span class="kpi-label">Загрузка...</span><span class="kpi-value">...</span></div>`;
  q("costs").innerHTML = `<div class="muted">Загрузка...</div>`;
  q("dialogs").innerHTML = `<div class="muted">Загрузка...</div>`;
  q("problems").innerHTML = `<div class="muted">Загрузка...</div>`;
  q("leads").innerHTML = `<div class="muted">Загрузка...</div>`;

  const eventQuery = readDevFilters();
  await Promise.all([
    getJson("/api/overview", clientId).then(renderOverview).catch((e) => setError("overview", e)),
    getJson("/api/costs", clientId).then(renderCosts).catch((e) => setError("costs", e)),
    getJson("/api/dialogs?limit=20", clientId).then(renderDialogs).catch((e) => setError("dialogs", e)),
    getJson("/api/problems?limit=40", clientId).then(renderProblems).catch((e) => setError("problems", e)),
    getJson("/api/leads?limit=40", clientId).then(renderLeads).catch((e) => setError("leads", e)),
    getJson(`/api/events?${eventQuery}`, clientId).then(renderEvents).catch((e) => {
      q("events").textContent = JSON.stringify({ error: String(e) }, null, 2);
    }),
  ]);
}

window.addEventListener("DOMContentLoaded", () => {
  q("refreshBtn").addEventListener("click", refreshAll);
  q("clientId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") refreshAll();
  });
  for (const id of ["eventTypeFilter", "sidFilter", "requestIdFilter"]) {
    q(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter") refreshAll();
    });
  }
  refreshAll();
});

