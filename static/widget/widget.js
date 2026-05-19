import { postAsk, streamAsk } from "./api.js";

const STORAGE_SID = "clinic_widget_sid";
const DEFAULT_AVATAR_URL = "/static/avatar.png";

const WELCOME_CHAR_MS = 32;
const WELCOME_STREAM_START_MS = 280;
const WELCOME_LEAVE_MS = 240;
const TEXTAREA_MAX_HEIGHT = 112;
const SCROLL_NEAR_BOTTOM_PX = 80;
const TURN_SCROLL_TOP_GAP_PX = 12;

const SEND_BTN_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><path d="M22 2L11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

const LINK_CHEVRON_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><path d="M9 18l6-6-6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

const CTA_CHAT_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><path d="M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/></svg>`;

const CTA_CALENDAR_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="5" width="18" height="16" rx="2" stroke="currentColor" stroke-width="1.75"/><path d="M8 3v4M16 3v4M3 10h18" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/></svg>`;

const WELCOME_LOGO_SVG = `<svg viewBox="0 0 370 371" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_clinic_welcome_logo)">
<path d="M185.5 22.8296C153.376 22.8296 121.974 32.3553 95.2641 50.2023C68.5543 68.0492 47.7365 93.4158 35.4433 123.094C23.1501 152.773 19.9337 185.43 26.2007 216.936C32.4677 248.443 47.9367 277.383 70.6516 300.098C93.3664 322.813 122.307 338.282 153.813 344.549C185.32 350.816 217.977 347.599 247.655 335.306C277.334 323.013 302.7 302.195 320.547 275.485C338.394 248.775 347.92 217.373 347.92 185.25C347.92 163.92 343.719 142.8 335.556 123.094C327.394 103.388 315.43 85.4834 300.348 70.4013C285.266 55.3192 267.361 43.3554 247.655 35.1931C227.95 27.0307 206.829 22.8296 185.5 22.8296ZM289.11 267.75C307.22 180.15 271.61 118.48 229.67 122.75C186.67 127.08 190.84 172.26 196.79 190.87C182.65 180.45 142.2 157.12 121.85 184.87C84.8998 235.35 178.68 305.48 233.77 315.41C170.73 306.97 94.0498 264.75 61.9998 202.5C30.9998 142.16 83.6298 96.7496 131 92.5596C120.53 111.96 112.88 122.67 102.46 133.26C133.76 118.12 147.42 81.1396 162.3 67.7496C177.18 54.3596 211.18 31.0196 256.85 74.1996C305.08 119.81 308.71 226.57 289.11 267.75ZM185.5 370.75C142.706 370.721 101.238 355.901 68.1186 328.801C34.9995 301.7 12.266 263.985 3.76869 222.043C-4.72861 180.101 1.53269 136.512 21.492 98.6577C41.4514 60.8036 73.8817 31.0122 113.29 14.3296C158.568 -4.53136 209.472 -4.69727 254.871 13.8681C300.271 32.4335 336.473 68.2197 355.562 113.402C374.65 158.584 375.072 209.487 356.736 254.98C338.399 300.472 302.796 336.854 257.71 356.17C234.865 365.835 210.305 370.794 185.5 370.75ZM185.5 5.74958C86.4998 5.74958 5.99984 86.2496 5.99984 185.25C5.99984 284.25 86.4998 364.75 185.5 364.75C284.5 364.75 365 284.25 365 185.25C365 86.2496 284.5 5.74958 185.5 5.74958Z" fill="#23BFCF"/>
<path d="M229.67 122.75C186.67 127.08 190.84 172.26 196.79 190.87C182.65 180.45 142.2 157.12 121.85 184.87C102.99 210.62 118.18 241.49 144.9 266.87C198.49 237.96 240.79 198.87 272.72 145.03C261.14 129.3 245.88 121.08 229.67 122.75Z" fill="#68CBD8"/>
<path d="M5.99984 185.25C5.99984 86.2498 86.4998 5.7498 185.5 5.7498C232.679 5.70854 277.965 24.3045 311.5 57.4898C312.19 55.3598 312.86 53.1998 313.5 51.0298C279.021 18.2101 233.245 -0.0990757 185.644 -0.109371C138.042 -0.119665 92.2585 18.1697 57.7657 50.9745C23.2729 83.7793 2.71117 128.589 0.334976 176.131C-2.04122 223.673 13.95 270.309 44.9998 306.39L51.4098 304.52C22.1185 271.702 5.95139 229.239 5.99984 185.25Z" fill="#68CBD8"/>
<path d="M185.5 22.8297C153.525 22.8356 122.262 32.2799 95.6324 49.9789C69.0023 67.6778 48.1892 92.8437 35.802 122.322C23.4148 151.8 20.0048 184.279 25.9994 215.688C31.9939 247.096 47.1263 276.036 69.4997 298.88C90.0093 292.174 110.053 284.12 129.5 274.77C101.29 255.89 76.8697 231.36 61.9997 202.5C30.9997 142.16 83.6297 96.7497 131 92.5597C120.53 111.96 112.88 122.67 102.46 133.26C133.76 118.12 147.42 81.1397 162.3 67.7497C177.18 54.3597 211.18 31.0197 256.85 74.1997C269.53 86.1997 279.12 102.42 286.03 120.56C293.27 105.929 299.686 90.906 305.25 75.5597C290.053 58.9266 271.554 45.6462 250.935 36.5668C230.316 27.4874 208.029 22.8088 185.5 22.8297Z" fill="#68CBD8"/>
</g>
<defs>
<clipPath id="clip0_clinic_welcome_logo">
<rect width="370" height="371" fill="white"/>
</clipPath>
</defs>
</svg>`;

const WELCOME_STAR_SVG = `<svg viewBox="0 0 244 244" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
<path d="M243.838 121.919C168.164 147.46 147.459 168.164 121.919 243.838C96.3781 168.164 75.674 147.459 0 121.919C75.674 96.378 96.3791 75.673 121.919 0C147.459 75.674 168.164 96.379 243.838 121.919Z" fill="#23BFCF"/>
</svg>`;

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
 * @param {string} avatarUrl
 * @returns {HTMLElement}
 */
function createBotAvatarEl(avatarUrl) {
  const wrap = document.createElement("div");
  wrap.className = "clinic-row__avatar-wrap";
  const av = document.createElement("img");
  av.className = "clinic-row__avatar";
  av.src = avatarUrl;
  av.alt = "";
  av.width = 38;
  av.height = 38;
  const dot = document.createElement("span");
  dot.className = "clinic-row__avatar-online";
  dot.setAttribute("aria-hidden", "true");
  wrap.appendChild(av);
  wrap.appendChild(dot);
  return wrap;
}

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  const next = Math.min(textarea.scrollHeight, TEXTAREA_MAX_HEIGHT);
  textarea.style.height = `${next}px`;
  textarea.style.overflowY =
    textarea.scrollHeight > TEXTAREA_MAX_HEIGHT ? "auto" : "hidden";
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
  const bubble = document.createElement("div");
  bubble.className = "clinic-msg clinic-msg--bot clinic-msg--bot--streaming";
  const body = document.createElement("div");
  body.className = "clinic-msg__body";
  bubble.appendChild(body);
  row.appendChild(createBotAvatarEl(resolvedAvatarUrl));
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
/**
 * @param {HTMLElement} feedEl
 * @param {{ force?: boolean }} [opts]
 */
function scrollChatPaneToEnd(feedEl, opts = {}) {
  const scroller = feedEl.closest(".clinic-shell__main");
  if (!scroller) return;
  if (!opts.force) {
    const dist =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    if (dist >= SCROLL_NEAR_BOTTOM_PX) return;
  }
  scroller.scrollTop = scroller.scrollHeight;
}

/**
 * Скроллит так, чтобы начало последнего bot-turn'а было видно сверху,
 * учитывая высоту липкой шапки внутри scroller'а.
 * @param {HTMLElement} feedEl
 */
function scrollToLastTurnStart(feedEl) {
  const scroller = feedEl.closest(".clinic-shell__main");
  if (!scroller) return;
  const turns = feedEl.querySelectorAll(".clinic-turn");
  const last = turns[turns.length - 1];
  if (!last) {
    scroller.scrollTop = scroller.scrollHeight;
    return;
  }
  const header = scroller.querySelector(".clinic-shell__header--glass");
  const headerH = header ? header.getBoundingClientRect().height : 0;
  const turnRect = last.getBoundingClientRect();
  const scrollerRect = scroller.getBoundingClientRect();
  const effectiveViewH = scroller.clientHeight - headerH;

  if (turnRect.height + TURN_SCROLL_TOP_GAP_PX > effectiveViewH) {
    const target =
      scroller.scrollTop +
      (turnRect.top - scrollerRect.top) -
      headerH -
      TURN_SCROLL_TOP_GAP_PX;
    scroller.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
  } else {
    scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
  }
}

function _updateLiveBubble(row, text, feed) {
  const body = row.querySelector(".clinic-msg__body");
  if (body) body.textContent = text;
  scrollChatPaneToEnd(feed);
}

function isDevHost() {
  const host = location.hostname;
  if (host === "localhost" || host === "127.0.0.1" || host === "[::1]") return true;
  if (new URLSearchParams(location.search).get("dev") === "1") return true;
  return false;
}

/**
 * Временно: кнопка сброса sid слева сверху (только dev-хост).
 * @param {() => void} onReset
 */
function attachDevResetControl(onReset) {
  if (!isDevHost()) return;
  if (document.querySelector("[data-clinic-dev-reset]")) return;

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "clinic-dev-reset";
  btn.setAttribute("data-clinic-dev-reset", "");
  btn.textContent = "DEV · сброс sid";
  btn.title = "Очистить sid и историю. Ctrl+Alt+R";
  btn.addEventListener("click", onReset);
  document.body.appendChild(btn);

  if (!window.__clinicDevResetKeyBound) {
    window.__clinicDevResetKeyBound = true;
    document.addEventListener("keydown", (ev) => {
      if (!isDevHost()) return;
      if (ev.ctrlKey && ev.altKey && ev.key.toLowerCase() === "r") {
        ev.preventDefault();
        onReset();
      }
    });
  }
}

/**
 * @param {HTMLElement} root
 * @param {WidgetConfig} config
 * @returns {{ resetSession: () => void }}
 */
export function mountWidget(root, config) {
  const apiBase = config.apiBase ?? "";
  const clientId = config.clientId || "default";
  const resolvedAvatarUrl = (config.avatarUrl || "").trim() || DEFAULT_AVATAR_URL;

  const state = {
    isOpen: false,
    messages: [],
    lastPayload: null,
    pending: false,
    unread: false,
    started: false,
    errorLine: "",
    welcomeAnimActive: false,
    welcomeStreamDone: false,
  };

  let welcomeStreamTimer = 0;

  function prefersReducedMotion() {
    return (
      typeof matchMedia !== "undefined" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function clearWelcomeStream() {
    if (welcomeStreamTimer) {
      clearTimeout(welcomeStreamTimer);
      welcomeStreamTimer = 0;
    }
  }

  /**
   * @param {HTMLParagraphElement} textP
   * @param {HTMLElement} textBody
   * @param {HTMLElement} cursor
   * @param {HTMLElement} card
   */
  function finishWelcomeStream(textP, card) {
    clearWelcomeStream();
    state.welcomeAnimActive = false;
    state.welcomeStreamDone = true;
    textP.classList.remove("is-typing");
    textP.classList.add("is-done");
    card.classList.remove("is-streaming");
  }

  /**
   * @param {HTMLParagraphElement} textP
   * @param {HTMLElement} textBody
   * @param {HTMLElement} card
   * @param {number} [startAt]
   */
  function startWelcomeTextStream(textP, textBody, card, startAt = 0) {
    const full = String(config.welcomeText || "").trim();
    clearWelcomeStream();
    state.welcomeAnimActive = true;
    textP.classList.remove("is-done");
    textP.classList.add("is-typing");
    card.classList.add("is-streaming");

    if (prefersReducedMotion() || !full) {
      textBody.textContent = full;
      finishWelcomeStream(textP, card);
      return;
    }

    let i = Math.min(Math.max(0, startAt), full.length);
    textBody.textContent = full.slice(0, i);

    if (i >= full.length) {
      finishWelcomeStream(textP, card);
      return;
    }

    const step = () => {
      welcomeStreamTimer = 0;
      if (state.started || !state.isOpen) return;
      if (i < full.length) {
        textBody.textContent = full.slice(0, i + 1);
        i += 1;
        welcomeStreamTimer = window.setTimeout(step, WELCOME_CHAR_MS);
      } else {
        finishWelcomeStream(textP, card);
      }
    };
    welcomeStreamTimer = window.setTimeout(step, WELCOME_STREAM_START_MS);
  }

  function maybeStartWelcomeStream() {
    if (!state.isOpen || state.started || state.welcomeStreamDone || state.welcomeAnimActive) {
      return;
    }
    const textP = feed.querySelector(".clinic-shell__welcome-text");
    const textBody = feed.querySelector(".clinic-shell__welcome-text-body");
    const card = feed.querySelector(".clinic-shell__welcome-card");
    if (!textP || !textBody || !card) return;

    const full = String(config.welcomeText || "").trim();
    const startAt = (textBody.textContent || "").length;
    if (startAt >= full.length && full.length > 0) {
      finishWelcomeStream(textP, card);
      return;
    }
    startWelcomeTextStream(textP, textBody, card, startAt);
  }

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
        <div class="clinic-shell__frame">
          <div class="clinic-shell__surface">
            <main class="clinic-shell__main" aria-label="Сообщения">
              <header class="clinic-shell__header clinic-shell__header--glass">
              <div class="clinic-shell__header-main">
                <div class="clinic-shell__header-avatar">
                  <span class="clinic-shell__avatar-fallback clinic-shell__avatar-fallback--header" data-clinic-header-fb>
                    <img class="clinic-shell__avatar-fallback-img" alt="" width="48" height="48" data-clinic-header-avatar />
                  </span>
                  <span class="clinic-shell__header-online-dot" aria-hidden="true"></span>
                </div>
                <div class="clinic-shell__header-text">
                  <span class="clinic-shell__header-name" data-clinic-header-name></span>
                  <span class="clinic-shell__header-status" data-clinic-header-online></span>
                </div>
              </div>
              <div class="clinic-shell__header-actions">
                <button type="button" class="clinic-shell__header-close clinic-btn-icon clinic-btn-ghost" data-clinic-close title="Свернуть" aria-label="Свернуть чат">✕</button>
              </div>
            </header>
              <div class="clinic-shell__feed" data-clinic-feed></div>
            </main>
            <form class="clinic-shell__composer" data-clinic-composer-form>
              <div class="clinic-shell__error" data-clinic-err hidden></div>
              <div class="clinic-shell__composer-inner">
                <textarea class="clinic-shell__textarea" rows="1" data-clinic-input placeholder="Введите сообщение" aria-label="Введите сообщение"></textarea>
                <button type="submit" class="clinic-btn-send" data-clinic-send disabled aria-label="Отправить сообщение">${SEND_BTN_SVG}</button>
              </div>
            </form>
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
  const composerForm = root.querySelector("[data-clinic-composer-form]");
  const errBox = root.querySelector("[data-clinic-err]");
  const unreadDot = root.querySelector("[data-clinic-unread]");
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
  autoResizeTextarea(input);

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
    clearWelcomeStream();
    clearStoredSid();
    state.messages = [];
    state.lastPayload = null;
    state.started = false;
    state.welcomeAnimActive = false;
    state.welcomeStreamDone = false;
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
      if (state.welcomeAnimActive) {
        clearWelcomeStream();
        state.welcomeAnimActive = false;
      }
      launcher.focus();
    }
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
      const lab = document.createElement("span");
      lab.className = "clinic-msg__link-text";
      lab.textContent = it.label || it.ref;
      const chev = document.createElement("span");
      chev.className = "clinic-msg__link-chevron";
      chev.setAttribute("aria-hidden", "true");
      chev.innerHTML = LINK_CHEVRON_SVG;
      btn.appendChild(lab);
      btn.appendChild(chev);
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
      sb.className = "clinic-turn__btn clinic-turn__btn--cta-secondary";
      sb.innerHTML = `<span class="clinic-turn__btn-icon">${CTA_CHAT_SVG}</span><span class="clinic-turn__btn-label">Рассказать о ситуации</span><span class="clinic-turn__btn-spacer" aria-hidden="true"></span>`;
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
      c.className = "clinic-turn__btn clinic-turn__btn--cta-primary";
      const ctaLabel = (m.cta.text || "Записаться на консультацию").trim();
      c.innerHTML = `<span class="clinic-turn__btn-icon">${CTA_CALENDAR_SVG}</span><span class="clinic-turn__btn-label"></span><span class="clinic-turn__btn-spacer" aria-hidden="true"></span>`;
      c.querySelector(".clinic-turn__btn-label").textContent = ctaLabel;
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
    const prevWelcome = feed.querySelector(".clinic-shell__welcome-screen");
    const keepWelcome = prevWelcome && !state.started && !state.welcomeStreamDone;

    if (keepWelcome && prevWelcome) {
      prevWelcome.remove();
    } else if (!state.started) {
      clearWelcomeStream();
      state.welcomeAnimActive = false;
      state.welcomeStreamDone = false;
    }

    feed.textContent = "";
    const typing = document.createElement("div");
    typing.className = "clinic-shell__typing";
    typing.setAttribute("aria-live", "polite");

    if (keepWelcome && prevWelcome) {
      feed.appendChild(prevWelcome);
    } else if (!state.started) {
      const screen = document.createElement("section");
      screen.className = "clinic-shell__welcome-screen";
      screen.setAttribute("aria-label", "Приветствие");

      const card = document.createElement("section");
      card.className = "clinic-shell__welcome-card";

      const logoWrap = document.createElement("div");
      logoWrap.className = "clinic-shell__welcome-logo-wrap";
      logoWrap.setAttribute("aria-hidden", "true");

      const logo = document.createElement("div");
      logo.className = "clinic-shell__welcome-logo";
      logo.innerHTML = WELCOME_LOGO_SVG;

      for (const key of /** @type {const} */ (["one", "two", "three"])) {
        const star = document.createElement("span");
        star.className = `clinic-shell__welcome-star clinic-shell__welcome-star--${key}`;
        star.innerHTML = WELCOME_STAR_SVG;
        logoWrap.appendChild(star);
      }
      logoWrap.insertBefore(logo, logoWrap.firstChild);

      const lead = document.createElement("div");
      lead.className = "clinic-shell__welcome-lead";
      const textP = document.createElement("p");
      textP.className = "clinic-shell__welcome-text";
      const textBody = document.createElement("span");
      textBody.className = "clinic-shell__welcome-text-body";
      const cursor = document.createElement("span");
      cursor.className = "clinic-shell__stream-cursor";
      cursor.setAttribute("aria-hidden", "true");
      textP.appendChild(textBody);
      textP.appendChild(cursor);
      lead.appendChild(textP);

      const wave = document.createElement("div");
      wave.className = "clinic-shell__welcome-wave";
      wave.setAttribute("aria-hidden", "true");

      card.appendChild(logoWrap);
      card.appendChild(lead);
      card.appendChild(wave);

      const actions = document.createElement("div");
      actions.className = "clinic-shell__welcome-actions";
      for (const s of config.starterPrompts || []) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "clinic-shell__welcome-action";
        b.textContent = s.label;
        b.addEventListener("click", () => {
          transitionFromWelcome(() => {
            input.value = s.q;
            void sendFromComposer();
          });
        });
        actions.appendChild(b);
      }

      screen.appendChild(card);
      screen.appendChild(actions);
      feed.appendChild(screen);

      if (state.welcomeStreamDone) {
        textBody.textContent = String(config.welcomeText || "").trim();
        textP.classList.add("is-done");
      }
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
      const bubble = document.createElement("div");
      bubble.className = "clinic-msg clinic-msg--bot";
      const body = document.createElement("div");
      body.className = "clinic-msg__body";
      body.textContent = m.text;
      bubble.appendChild(body);
      renderInlineLinks(bubble, m, idx);
      row.appendChild(createBotAvatarEl(resolvedAvatarUrl));
      row.appendChild(bubble);
      wrap.appendChild(row);
      renderTrail(wrap, m, idx);
      feed.appendChild(wrap);
    });

    const typingWrap = document.createElement("div");
    typingWrap.className = "clinic-shell__typing-wrap";
    typing.textContent = "Бот печатает…";
    typingWrap.appendChild(createBotAvatarEl(resolvedAvatarUrl));
    typingWrap.appendChild(typing);
    typingWrap.classList.toggle("is-visible", state.pending);
    feed.appendChild(typingWrap);

    const lastMsg = state.messages.length
      ? state.messages[state.messages.length - 1]
      : null;
    if (lastMsg && lastMsg.role === "bot") {
      requestAnimationFrame(() => scrollToLastTurnStart(feed));
    } else {
      scrollChatPaneToEnd(feed, { force: state.messages.length > 0 });
    }
    syncComposerLeadUi();
    syncSendState();
  }

  /**
   * @param {() => void} done
   */
  function transitionFromWelcome(done) {
    const welcome = feed.querySelector(".clinic-shell__welcome-screen");
    if (!welcome || state.started) {
      done();
      return;
    }
    welcome.classList.add("is-leaving");
    window.setTimeout(() => {
      state.started = true;
      clearWelcomeStream();
      done();
    }, WELCOME_LEAVE_MS);
  }

  async function sendAsk(extra = {}) {
    const userEcho =
      typeof extra.userEcho === "string" ? extra.userEcho.trim() : "";
    const linkOnly = Boolean(extra._linkOnly);
    const apiFields = { ...extra };
    delete apiFields.userEcho;
    delete apiFields._linkOnly;

    if (userEcho) {
      const applyUserEcho = () => {
        if (linkOnly) {
          dismissTrailingsAll(state.messages);
        } else {
          dismissTrailingsAll(state.messages);
          dismissLinksAll(state.messages);
        }
        state.messages.push({ role: "user", text: userEcho });
      };
      if (!state.started && feed.querySelector(".clinic-shell__welcome-screen")) {
        await new Promise((resolve) => {
          transitionFromWelcome(() => {
            applyUserEcho();
            resolve();
          });
        });
      } else {
        if (!state.started) {
          state.started = true;
          clearWelcomeStream();
        }
        applyUserEcho();
      }
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

    const runSend = async () => {
      dismissTrailingsAll(state.messages);
      dismissLinksAll(state.messages);
      state.messages.push({ role: "user", text: userBubbleText });
      input.value = "";
      autoResizeTextarea(input);
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
    };

    if (!state.started && feed.querySelector(".clinic-shell__welcome-screen")) {
      transitionFromWelcome(() => {
        void runSend();
      });
      return;
    }
    if (!state.started) {
      state.started = true;
      clearWelcomeStream();
    }
    await runSend();
  }

  function isLeadPhoneStep() {
    const m = state.lastPayload?.meta;
    return leadMetaPhoneStep(m);
  }

  function syncComposerLeadUi() {
    const phone = isLeadPhoneStep();
    input.inputMode = phone ? "numeric" : "text";
    input.classList.toggle("clinic-shell__textarea--phone", phone);
    input.placeholder = phone ? "+7(900) 000-00-00" : "Введите сообщение";
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
    autoResizeTextarea(input);
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
    if (state.isOpen) maybeStartWelcomeStream();
  });

  btnClose.addEventListener("click", () => {
    setOpen(false);
  });

  input.addEventListener("input", onComposerInput);

  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      void sendFromComposer();
    }
  });

  composerForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    void sendFromComposer();
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

  attachDevResetControl(resetSession);

  return { resetSession };
}
