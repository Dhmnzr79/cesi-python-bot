import { postAsk, streamAsk } from "./api.js";

const STORAGE_SID = "clinic_widget_sid";
const DEFAULT_AVATAR_URL = "/static/avatar.png";

/** @param {unknown} meta */
function leadMetaPhoneStep(meta) {
  return Boolean(
    meta && typeof meta === "object" && meta.lead_flow && meta.lead_step === "phone"
  );
}

/** 10 цифр после «7» (пользователь может ввести 9… или 8… или уже +7…) */
function extractNational10Digits(raw) {
  let d = String(raw || "").replace(/\D/g, "");
  if (!d.length) return "";
  if (d.startsWith("8")) d = "7" + d.slice(1);
  if (d.startsWith("7")) return d.slice(1, 11);
  return d.slice(0, 10);
}

/** Отображение: +7(000) 000-00-00 */
function formatRuMobileDisplay(nationalUpTo10) {
  const n = nationalUpTo10.replace(/\D/g, "").slice(0, 10);
  if (!n.length) return "+7";
  let s = "+7(" + n.slice(0, 3);
  if (n.length <= 3) return s;
  s += ") " + n.slice(3, 6);
  if (n.length <= 6) return s;
  s += "-" + n.slice(6, 8);
  if (n.length <= 8) return s;
  s += "-" + n.slice(8, 10);
  return s;
}

function ruPhoneToBackendE164(inputVal) {
  const n = extractNational10Digits(inputVal);
  if (n.length !== 10) return "";
  return "+7" + n;
}

/**
 * @typedef {Object} StarterPrompt
 * @property {string} label
 * @property {string} q
 */

/**
 * @typedef {Object} WidgetConfig
 * @property {string} [apiBase]
 * @property {string} clientId
 * @property {string} botName
 * @property {string} [avatarUrl]
 * @property {string} onlineLabel
 * @property {string} welcomeText
 * @property {StarterPrompt[]} starterPrompts
 */

/**
 * @param {import("./api.js").postAsk} _
 * @param {unknown} data
 */
function botTurnFromPayload(data) {
  if (!data || typeof data !== "object") return null;
  const meta = /** @type {Record<string, unknown>} */ (data.meta || {});
  const followups = Array.isArray(meta.followups) ? meta.followups : [];
  const quickReplies = Array.isArray(data.quick_replies) ? data.quick_replies : [];
  const sit = data.situation && typeof data.situation === "object" ? data.situation : null;
  const ctaRaw = data.cta;
  const cta =
    ctaRaw && typeof ctaRaw === "object" && ctaRaw.text
      ? { text: String(ctaRaw.text), action: String(ctaRaw.action || "lead") }
      : null;
  const videoKey =
    data.video && typeof data.video === "object" && data.video.key
      ? String(data.video.key)
      : null;

  return {
    role: "bot",
    text: String(data.answer || "").trim(),
    followups: followups.filter((x) => x && x.ref),
    quickReplies: quickReplies.filter((x) => x && x.ref),
    linksDismissed: false,
    videoKey,
    situation: sit ? { show: Boolean(sit.show), mode: sit.mode || "normal" } : null,
    cta,
    trailingDismissed: false,
  };
}

function dismissTrailingsAll(messages) {
  for (const m of messages) {
    if (m.role === "bot") m.trailingDismissed = true;
  }
}

function dismissLinksAll(messages) {
  for (const m of messages) {
    if (m.role === "bot") m.linksDismissed = true;
  }
}

/**
 * Создаёт «живую» bubble в feed перед typing-wrap и скрывает typing indicator.
 * Вызывается лениво — только при первом text_delta.
 * @param {HTMLElement} feed
 * @param {string} resolvedAvatarUrl
 * @returns {HTMLElement} row — корневой элемент bubble
 */
function _createLiveBubble(feed, resolvedAvatarUrl) {
  const typingWrap = feed.querySelector(".clinic-shell__typing-wrap");
  const row = document.createElement("div");
  row.className = "clinic-row clinic-row--bot";
  row.setAttribute("data-live-bubble", "");
  const av = document.createElement("img");
  av.className = "clinic-row__avatar";
  av.src = resolvedAvatarUrl;
  av.alt = "";
  av.width = 32;
  av.height = 32;
  const bubble = document.createElement("div");
  bubble.className = "clinic-msg clinic-msg--bot";
  const body = document.createElement("div");
  body.className = "clinic-msg__body";
  bubble.appendChild(body);
  row.appendChild(av);
  row.appendChild(bubble);
  feed.insertBefore(row, typingWrap);
  if (typingWrap) typingWrap.classList.remove("is-visible");
  return row;
}

/**
 * Обновляет текст в живой bubble и скроллит вниз.
 * @param {HTMLElement} row
 * @param {string} text
 * @param {HTMLElement} feed
 */
function _updateLiveBubble(row, text, feed) {
  const body = row.querySelector(".clinic-msg__body");
  if (body) body.textContent = text;
  feed.scrollTop = feed.scrollHeight;
}

/**
 * @param {HTMLElement} root
 * @param {WidgetConfig} config
 */
export function mountWidget(root, config) {
  const apiBase = config.apiBase ?? "";
  const clientId = config.clientId || "default";
  const resolvedAvatarUrl = (config.avatarUrl || "").trim() || DEFAULT_AVATAR_URL;

  const state = {
    isOpen: false,
    isExpanded: false,
    messages: [],
    lastPayload: null,
    pending: false,
    unread: false,
    started: false,
    errorLine: "",
  };

  root.innerHTML = `
    <div class="clinic-shell" data-clinic-root>
      <button type="button" class="clinic-shell__launcher" data-clinic-launcher aria-expanded="false" aria-controls="clinic-panel">
        <span class="clinic-shell__unread" data-clinic-unread aria-hidden="true"></span>
        <span class="clinic-shell__avatar-fallback" data-clinic-avatar-fb>
          <img class="clinic-shell__avatar-fallback-img" alt="" width="40" height="40" data-clinic-avatar />
        </span>
        <span class="clinic-shell__launcher-text">
          <span class="clinic-shell__name" data-clinic-name></span>
          <span class="clinic-shell__online" data-clinic-online></span>
        </span>
      </button>
      <div class="clinic-shell__panel" id="clinic-panel" role="dialog" aria-modal="true" aria-label="Чат" data-clinic-panel>
        <header class="clinic-shell__header">
          <div class="clinic-shell__header-main">
            <span class="clinic-shell__avatar-fallback clinic-shell__avatar-fallback--sm" data-clinic-header-fb>
              <img class="clinic-shell__avatar-fallback-img" alt="" width="36" height="36" data-clinic-header-avatar />
            </span>
            <span class="clinic-shell__launcher-text">
              <span class="clinic-shell__name" data-clinic-header-name></span>
              <span class="clinic-shell__online" data-clinic-header-online></span>
            </span>
          </div>
          <div class="clinic-shell__header-actions">
            <button type="button" class="clinic-shell__session-reset" data-clinic-reset-session title="Очистить sid и историю (временно для отладки)">Сброс</button>
            <button type="button" class="clinic-btn-icon clinic-btn-ghost" data-clinic-expand aria-expanded="false" title="Шире">⤢</button>
            <button type="button" class="clinic-btn-icon clinic-btn-ghost" data-clinic-close title="Свернуть">✕</button>
          </div>
        </header>
        <div class="clinic-shell__feed" data-clinic-feed></div>
        <div class="clinic-shell__composer">
          <div class="clinic-shell__error" data-clinic-err hidden></div>
          <div class="clinic-shell__composer-inner">
            <textarea class="clinic-shell__textarea" rows="2" data-clinic-input placeholder="Напишите вопрос…"></textarea>
            <button type="button" class="clinic-btn-send" data-clinic-send disabled aria-label="Отправить">➤</button>
          </div>
        </div>
      </div>
      <div class="clinic-shell__video-overlay" data-clinic-video-overlay hidden>
        <div class="clinic-shell__video-card" role="document">
          <p data-clinic-video-title></p>
          <button type="button" class="clinic-btn-ghost" data-clinic-video-close>Закрыть</button>
        </div>
      </div>
    </div>
  `;

  const shell = root.querySelector("[data-clinic-root]");
  const launcher = root.querySelector("[data-clinic-launcher]");
  const panel = root.querySelector("[data-clinic-panel]");
  const feed = root.querySelector("[data-clinic-feed]");
  const input = root.querySelector("[data-clinic-input]");
  const sendBtn = root.querySelector("[data-clinic-send]");
  const errBox = root.querySelector("[data-clinic-err]");
  const btnResetSession = root.querySelector("[data-clinic-reset-session]");
  const unreadDot = root.querySelector("[data-clinic-unread]");
  const btnExpand = root.querySelector("[data-clinic-expand]");
  const btnClose = root.querySelector("[data-clinic-close]");
  const videoOverlay = root.querySelector("[data-clinic-video-overlay]");
  const videoTitle = root.querySelector("[data-clinic-video-title]");
  const videoClose = root.querySelector("[data-clinic-video-close]");

  const avatarImg = root.querySelector("[data-clinic-avatar]");
  const hAvatar = root.querySelector("[data-clinic-header-avatar]");

  root.querySelector("[data-clinic-name]").textContent = config.botName;
  root.querySelector("[data-clinic-online]").textContent = config.onlineLabel;
  root.querySelector("[data-clinic-header-name]").textContent = config.botName;
  root.querySelector("[data-clinic-header-online]").textContent = config.onlineLabel;

  const alt = (config.botName || "Бот").trim();
  avatarImg.alt = alt;
  hAvatar.alt = alt;
  avatarImg.src = resolvedAvatarUrl;
  hAvatar.src = resolvedAvatarUrl;

  setOpen(false);

  function getSid() {
    try {
      return localStorage.getItem(STORAGE_SID) || "";
    } catch {
      return "";
    }
  }

  function setSid(sid) {
    if (!sid) return;
    try {
      localStorage.setItem(STORAGE_SID, sid);
    } catch {
      /* ignore */
    }
  }

  function clearStoredSid() {
    try {
      localStorage.removeItem(STORAGE_SID);
    } catch {
      /* ignore */
    }
  }

  function resetSession() {
    if (state.pending) return;
    clearStoredSid();
    state.messages = [];
    state.lastPayload = null;
    state.started = false;
    state.unread = false;
    unreadDot.classList.remove("is-visible");
    setError("");
    videoOverlay.hidden = true;
    input.value = "";
    renderFeed();
  }

  function setOpen(open) {
    state.isOpen = open;
    shell.classList.toggle("is-open", open);
    launcher.setAttribute("aria-expanded", open ? "true" : "false");
    panel.setAttribute("aria-hidden", open ? "false" : "true");
    if (open) {
      state.unread = false;
      unreadDot.classList.remove("is-visible");
      input.focus();
    } else {
      launcher.focus();
    }
  }

  function setExpanded(on) {
    state.isExpanded = on;
    shell.classList.toggle("is-expanded", on);
    btnExpand.setAttribute("aria-expanded", on ? "true" : "false");
  }

  function setError(msg) {
    state.errorLine = msg || "";
    if (msg) {
      errBox.textContent = msg;
      errBox.hidden = false;
    } else {
      errBox.textContent = "";
      errBox.hidden = true;
    }
  }

  /**
   * @param {HTMLElement} bubble
   * @param {object} m
   * @param {number} msgIndex
   */
  function renderInlineLinks(bubble, m, msgIndex) {
    if (m.linksDismissed) return;
    const items = [];
    for (const f of m.followups || []) {
      items.push({ label: (f.label || f.ref || "").trim(), ref: f.ref });
    }
    for (const r of m.quickReplies || []) {
      items.push({ label: (r.label || r.ref || "").trim(), ref: r.ref });
    }
    if (!items.length) return;

    const box = document.createElement("div");
    box.className = "clinic-msg__links";
    for (const it of items) {
      if (!it.ref) continue;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "clinic-msg__link";
      const arrow = document.createElement("span");
      arrow.className = "clinic-msg__link-arrow";
      arrow.setAttribute("aria-hidden", "true");
      arrow.textContent = "→";
      const lab = document.createElement("span");
      lab.className = "clinic-msg__link-text";
      lab.textContent = it.label || it.ref;
      btn.appendChild(arrow);
      btn.appendChild(lab);
      btn.addEventListener("click", () => {
        const target = state.messages[msgIndex];
        if (target && target.role === "bot") target.linksDismissed = true;
        dismissTrailingsAll(state.messages);
        const echo = (it.label || it.ref || "").trim();
        void sendAsk({ ref: it.ref, q: "", userEcho: echo, _linkOnly: true });
      });
      box.appendChild(btn);
    }
    bubble.appendChild(box);
  }

  /**
   * @param {HTMLElement} wrap
   * @param {object} m
   * @param {number} msgIndex
   */
  function renderTrail(wrap, m, msgIndex) {
    if (m.role !== "bot" || m.trailingDismissed) return;

    const trail = document.createElement("div");
    trail.className = "clinic-turn__trail";

    if (m.videoKey) {
      const key = m.videoKey;
      const vr = document.createElement("button");
      vr.type = "button";
      vr.className = "clinic-turn__btn clinic-turn__btn--video";
      vr.innerHTML =
        '<span class="clinic-turn__btn-label">Видео</span><span class="clinic-turn__btn-play" aria-hidden="true">▶</span>';
      vr.setAttribute("aria-label", `Видео, ${key}`);
      vr.addEventListener("click", () => {
        videoTitle.textContent = `Видео (key: ${key}). Плеер — по договорённости в ТЗ.`;
        videoOverlay.hidden = false;
      });
      trail.appendChild(vr);
    }

    const sit = m.situation;
    if (sit && sit.show && sit.mode === "normal") {
      const sb = document.createElement("button");
      sb.type = "button";
      sb.className = "clinic-turn__btn clinic-turn__btn--situation";
      sb.textContent = "Рассказать о ситуации";
      sb.addEventListener("click", () => {
        dismissTrailingsAll(state.messages);
        dismissLinksAll(state.messages);
        void sendAsk({ action: "situation", q: "", userEcho: "Рассказать о ситуации" });
      });
      trail.appendChild(sb);
    }

    if (sit && sit.show && sit.mode === "pending") {
      const back = document.createElement("button");
      back.type = "button";
      back.className = "clinic-turn__btn clinic-turn__btn--ghost-wide";
      back.textContent = "Назад к диалогу";
      back.addEventListener("click", () => {
        dismissTrailingsAll(state.messages);
        dismissLinksAll(state.messages);
        void sendAsk({ situation_action: "back", q: "", userEcho: "Назад к диалогу" });
      });
      trail.appendChild(back);
    }

    if (m.cta && m.cta.text) {
      const c = document.createElement("button");
      c.type = "button";
      c.className = "clinic-turn__btn clinic-turn__btn--cta";
      c.textContent = m.cta.text;
      c.addEventListener("click", () => {
        dismissTrailingsAll(state.messages);
        dismissLinksAll(state.messages);
        const echo = (m.cta.text || "Запись").trim();
        void sendAsk({ cta_action: "lead", q: "", userEcho: echo });
      });
      trail.appendChild(c);
    }

    if (trail.children.length) wrap.appendChild(trail);
  }

  function renderFeed() {
    feed.textContent = "";
    const typing = document.createElement("div");
    typing.className = "clinic-shell__typing";
    typing.setAttribute("aria-live", "polite");

    if (!state.started) {
      const w = document.createElement("p");
      w.className = "clinic-shell__welcome";
      w.textContent = config.welcomeText;
      feed.appendChild(w);
      const row = document.createElement("div");
      row.className = "clinic-shell__starters";
      for (const s of config.starterPrompts || []) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "clinic-chip";
        b.textContent = s.label;
        b.addEventListener("click", () => {
          state.started = true;
          input.value = s.q;
          void sendFromComposer();
        });
        row.appendChild(b);
      }
      feed.appendChild(row);
    }

    state.messages.forEach((m, idx) => {
      if (m.role === "user") {
        const row = document.createElement("div");
        row.className = "clinic-row clinic-row--user";
        const bubble = document.createElement("div");
        bubble.className = "clinic-msg clinic-msg--user";
        bubble.textContent = m.text;
        row.appendChild(bubble);
        feed.appendChild(row);
        return;
      }

      const wrap = document.createElement("div");
      wrap.className = "clinic-turn";

      const row = document.createElement("div");
      row.className = "clinic-row clinic-row--bot";
      const av = document.createElement("img");
      av.className = "clinic-row__avatar";
      av.src = resolvedAvatarUrl;
      av.alt = "";
      av.width = 32;
      av.height = 32;
      const bubble = document.createElement("div");
      bubble.className = "clinic-msg clinic-msg--bot";
      const body = document.createElement("div");
      body.className = "clinic-msg__body";
      body.textContent = m.text;
      bubble.appendChild(body);
      renderInlineLinks(bubble, m, idx);
      row.appendChild(av);
      row.appendChild(bubble);
      wrap.appendChild(row);
      renderTrail(wrap, m, idx);
      feed.appendChild(wrap);
    });

    const typingWrap = document.createElement("div");
    typingWrap.className = "clinic-shell__typing-wrap";
    const typingAv = document.createElement("img");
    typingAv.className = "clinic-row__avatar";
    typingAv.src = resolvedAvatarUrl;
    typingAv.alt = "";
    typingAv.width = 32;
    typingAv.height = 32;
    typing.textContent = "Бот печатает…";
    typingWrap.appendChild(typingAv);
    typingWrap.appendChild(typing);
    typingWrap.classList.toggle("is-visible", state.pending);
    feed.appendChild(typingWrap);

    feed.scrollTop = feed.scrollHeight;
    if (btnResetSession) btnResetSession.disabled = state.pending;
    syncComposerLeadUi();
    syncSendState();
  }

  async function sendAsk(extra = {}) {
    const userEcho =
      typeof extra.userEcho === "string" ? extra.userEcho.trim() : "";
    const linkOnly = Boolean(extra._linkOnly);
    const apiFields = { ...extra };
    delete apiFields.userEcho;
    delete apiFields._linkOnly;

    if (userEcho) {
      state.started = true;
      if (linkOnly) {
        dismissTrailingsAll(state.messages);
      } else {
        dismissTrailingsAll(state.messages);
        dismissLinksAll(state.messages);
      }
      state.messages.push({ role: "user", text: userEcho });
    }

    const sid = getSid();
    const body = {
      client_id: clientId,
      sid,
      q: "",
      ...apiFields,
    };
    if (body.q === undefined) body.q = "";

    setError("");
    state.pending = true;
    renderFeed();

    let liveBubble = null;
    let fullText = "";
    let uiData = null;

    await streamAsk(apiBase, body, {
      onDelta(delta) {
        fullText += delta;
        if (!liveBubble) liveBubble = _createLiveBubble(feed, resolvedAvatarUrl);
        _updateLiveBubble(liveBubble, fullText, feed);
      },
      onUi(data) {
        uiData = data;
      },
      onDone() {
        if (uiData) {
          if (uiData.meta && uiData.meta.sid) setSid(uiData.meta.sid);
          const turn = botTurnFromPayload(uiData);
          if (turn && turn.text) state.messages.push(turn);
          state.lastPayload = uiData;
          if (!state.isOpen) state.unread = true;
        }
        state.pending = false;
        if (state.unread && !state.isOpen) unreadDot.classList.add("is-visible");
        renderFeed();
      },
      onError(msg) {
        setError(msg);
        state.pending = false;
        renderFeed();
      },
    });
  }

  async function sendFromComposer() {
    if (state.pending) return;

    let q = input.value.trim();
    let userBubbleText = q;

    if (isLeadPhoneStep()) {
      const backend = ruPhoneToBackendE164(input.value);
      if (backend.length !== 12) return;
      q = backend;
      userBubbleText = formatRuMobileDisplay(extractNational10Digits(input.value));
    } else if (!q) {
      return;
    }

    dismissTrailingsAll(state.messages);
    dismissLinksAll(state.messages);

    state.started = true;
    state.messages.push({ role: "user", text: userBubbleText });
    input.value = "";
    sendBtn.disabled = true;
    setError("");

    const sid = getSid();
    state.pending = true;
    renderFeed();

    let liveBubble = null;
    let fullText = "";
    let uiData = null;

    await streamAsk(apiBase, { client_id: clientId, sid, q }, {
      onDelta(delta) {
        fullText += delta;
        if (!liveBubble) liveBubble = _createLiveBubble(feed, resolvedAvatarUrl);
        _updateLiveBubble(liveBubble, fullText, feed);
      },
      onUi(data) {
        uiData = data;
      },
      onDone() {
        if (uiData) {
          if (uiData.meta && uiData.meta.sid) setSid(uiData.meta.sid);
          const turn = botTurnFromPayload(uiData);
          if (turn && turn.text) state.messages.push(turn);
          state.lastPayload = uiData;
          if (!state.isOpen) state.unread = true;
        }
        state.pending = false;
        if (state.unread && !state.isOpen) unreadDot.classList.add("is-visible");
        renderFeed();
        syncSendState();
      },
      onError(msg) {
        setError(msg);
        state.pending = false;
        renderFeed();
        syncSendState();
      },
    });
  }

  function isLeadPhoneStep() {
    const m = state.lastPayload?.meta;
    return leadMetaPhoneStep(m);
  }

  function syncComposerLeadUi() {
    const phone = isLeadPhoneStep();
    input.inputMode = phone ? "numeric" : "text";
    input.classList.toggle("clinic-shell__textarea--phone", phone);
    input.placeholder = phone ? "+7(900) 000-00-00" : "Напишите вопрос…";
  }

  function onComposerInput() {
    if (isLeadPhoneStep()) {
      const nat = extractNational10Digits(input.value);
      const next = formatRuMobileDisplay(nat);
      if (next !== input.value) {
        input.value = next;
        input.selectionStart = input.selectionEnd = next.length;
      }
    }
    syncSendState();
  }

  function syncSendState() {
    if (state.pending) {
      sendBtn.disabled = true;
      return;
    }
    if (isLeadPhoneStep()) {
      sendBtn.disabled = extractNational10Digits(input.value).length !== 10;
      return;
    }
    sendBtn.disabled = !input.value.trim();
  }

  launcher.addEventListener("click", () => {
    setOpen(!state.isOpen);
    renderFeed();
  });

  btnClose.addEventListener("click", () => {
    setOpen(false);
  });

  btnExpand.addEventListener("click", () => {
    setExpanded(!state.isExpanded);
  });

  input.addEventListener("input", onComposerInput);

  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      void sendFromComposer();
    }
  });

  sendBtn.addEventListener("click", () => {
    void sendFromComposer();
  });

  btnResetSession.addEventListener("click", () => {
    resetSession();
  });

  videoClose.addEventListener("click", () => {
    videoOverlay.hidden = true;
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.isOpen) {
      if (!videoOverlay.hidden) {
        videoOverlay.hidden = true;
      } else {
        setOpen(false);
      }
    }
  });

  renderFeed();
}
