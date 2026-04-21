/**
 * Временная dev-страница: инспектор /ask + /__debug/retrieval.
 * Удаляется вместе с папкой static/debug/.
 */

const STORAGE_SID = "clinic_ask_insp_sid";
const STORAGE_TOKEN = "clinic_ask_insp_debug_token";

function getStored(key) {
  try {
    return localStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function setStored(key, val) {
  try {
    localStorage.setItem(key, val);
  } catch {
    /* ignore */
  }
}

function el(id) {
  const n = document.getElementById(id);
  if (!n) throw new Error("missing #" + id);
  return n;
}

function addMessage(role, text) {
  const chat = el("askInspChat");
  const div = document.createElement("div");
  div.className = "ask-insp__msg ask-insp__msg--" + role;
  div.textContent = (role === "user" ? "Вы: " : "Бот: ") + (text || "");
  chat.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearDynamic() {
  el("askInspQuick").textContent = "";
  el("askInspFollowups").textContent = "";
  el("askInspCta").textContent = "";
  el("askInspVideo").textContent = "";
  el("askInspSituation").textContent = "";
}

function renderChipList(container, list, onPick) {
  if (!Array.isArray(list) || !list.length) return;
  for (const item of list) {
    if (!item || !item.ref) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = item.label || item.ref;
    b.addEventListener("click", () => onPick(item));
    container.appendChild(b);
  }
}

function renderVideoBlock(data) {
  const box = el("askInspVideo");
  box.textContent = "";
  const v = data && data.video;
  if (!v) {
    box.textContent = "Нет поля video.";
    return;
  }
  if (v.key) {
    const p = document.createElement("p");
    p.textContent = "video.key: " + String(v.key);
    box.appendChild(p);
  }
  if (v.title || v.url) {
    const p = document.createElement("p");
    p.textContent = [v.title, v.url].filter(Boolean).join(" — ");
    box.appendChild(p);
  }
  if (!v.key && !v.title && !v.url) {
    box.textContent = "Пустой объект video.";
  }
}

function renderSituation(data) {
  const box = el("askInspSituation");
  box.textContent = "";
  const sit = data && data.situation;
  if (!sit || sit.show !== true) {
    box.textContent = "Ситуация не предложена (show !== true).";
    return;
  }
  const mode = sit.mode || "normal";
  if (mode === "pending") {
    const hint = document.createElement("p");
    hint.className = "ask-insp__lead ask-insp__tight";
    hint.textContent = "Режим pending: опишите ситуацию и отправьте.";
    box.appendChild(hint);
    const ta = document.createElement("textarea");
    ta.id = "askInspSitTa";
    ta.setAttribute("aria-label", "Текст ситуации");
    box.appendChild(ta);
    const row = document.createElement("div");
    row.className = "ask-insp__row";
    const sendB = document.createElement("button");
    sendB.type = "button";
    sendB.textContent = "Отправить ситуацию";
    const backB = document.createElement("button");
    backB.type = "button";
    backB.textContent = "Назад к диалогу";
    row.appendChild(sendB);
    row.appendChild(backB);
    box.appendChild(row);
    sendB.addEventListener("click", () => {
      const t = ta.value.trim();
      if (!t) return;
      void sendAsk({ q: t });
    });
    backB.addEventListener("click", () => {
      void sendAsk({ situation_action: "back", q: "" });
    });
    ta.focus();
    return;
  }
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = "Рассказать о ситуации";
  b.addEventListener("click", () => {
    void sendAsk({ situation_action: "start", q: "" });
  });
  box.appendChild(b);
}

/** Совпадает с дефолтом `DEBUG_TOKEN` в `config.py`, если в .env не задано иное */
const DEFAULT_DEBUG_TOKEN = "dev-debug";

async function loadDebugCandidates(qForDebug, clientId, debugToken) {
  const pre = el("askInspDebug");
  pre.textContent = "Загрузка…";
  const url =
    "/__debug/retrieval?q=" +
    encodeURIComponent(qForDebug || "") +
    "&client_id=" +
    encodeURIComponent(clientId || "default");
  const token = String(debugToken || "").trim() || DEFAULT_DEBUG_TOKEN;
  const headers = { "X-Debug-Token": token };
  try {
    const res = await fetch(url, { headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      pre.textContent = JSON.stringify({ error: data.error || res.status, body: data }, null, 2);
      return;
    }
    pre.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    pre.textContent = String(e);
  }
}

async function sendAsk(opts = {}) {
  const qInput = el("askInspQ");
  const clientId = el("askInspClient").value.trim() || "default";
  const token = el("askInspToken").value.trim();
  if (token) setStored(STORAGE_TOKEN, token);

  const q = opts && opts.q !== undefined ? opts.q : qInput.value.trim();
  const ref = opts && opts.ref !== undefined ? opts.ref : null;
  const ctaAction = opts && opts.cta_action !== undefined ? opts.cta_action : null;
  const situationAction = opts && opts.situation_action !== undefined ? opts.situation_action : null;
  const action = opts && opts.action !== undefined ? opts.action : null;

  if (!q && !ref && !ctaAction && !situationAction && !action) return;

  if (q) {
    addMessage("user", q);
    qInput.value = "";
  }

  clearDynamic();
  el("askInspRaw").textContent = "Загрузка…";

  let sid = getStored(STORAGE_SID);
  if (!sid) {
    sid = "insp_" + Date.now();
    setStored(STORAGE_SID, sid);
  }
  el("askInspSid").textContent = sid;

  const payload = { sid, client_id: clientId };
  if (q) payload.q = q;
  if (ref) payload.ref = ref;
  if (ctaAction) payload.cta_action = ctaAction;
  if (situationAction) payload.situation_action = situationAction;
  if (action) payload.action = action;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    el("askInspRaw").textContent = JSON.stringify(data, null, 2);
    addMessage("bot", data.answer || "(пустой ответ)");

    if (data.meta && data.meta.sid) {
      setStored(STORAGE_SID, data.meta.sid);
      el("askInspSid").textContent = data.meta.sid;
    }

    renderChipList(el("askInspQuick"), data.quick_replies || [], (item) => {
      void sendAsk({ ref: item.ref, q: "" });
    });

    const fups = data.meta && Array.isArray(data.meta.followups) ? data.meta.followups : [];
    renderChipList(el("askInspFollowups"), fups, (item) => {
      void sendAsk({ ref: item.ref, q: "" });
    });

    const cta = data.cta;
    if (cta && cta.text) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = cta.text;
      b.addEventListener("click", () => {
        addMessage("user", "[CTA] " + cta.text);
        void sendAsk({ cta_action: cta.action || "lead", q: "" });
      });
      el("askInspCta").appendChild(b);
    }

    renderVideoBlock(data);
    renderSituation(data);

    await loadDebugCandidates(q || ref || "", clientId, token || getStored(STORAGE_TOKEN));
  } catch (e) {
    el("askInspRaw").textContent = String(e);
    el("askInspDebug").textContent = "";
    addMessage("bot", "Ошибка запроса к /ask");
  }
}

function init() {
  const tok = getStored(STORAGE_TOKEN);
  if (tok) el("askInspToken").value = tok;
  const sid = getStored(STORAGE_SID);
  if (sid) el("askInspSid").textContent = sid;

  el("askInspSend").addEventListener("click", () => {
    void sendAsk({});
  });

  el("askInspReset").addEventListener("click", () => {
    const next = "insp_" + Date.now();
    setStored(STORAGE_SID, next);
    el("askInspSid").textContent = next;
    el("askInspChat").textContent = "";
    clearDynamic();
    el("askInspRaw").textContent = "";
    el("askInspDebug").textContent = "";
  });

  el("askInspSaveToken").addEventListener("click", () => {
    const t = el("askInspToken").value.trim();
    setStored(STORAGE_TOKEN, t);
  });

  el("askInspQ").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      void sendAsk({});
    }
  });
}

init();
