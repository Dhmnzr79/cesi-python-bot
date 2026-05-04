import os
import re
import sys
import time
import inspect
import json
import threading
from collections import deque
import numpy as np

from flask import Flask, jsonify, request, send_from_directory
import session as session_mod

from config import (
    ALIAS_STRONG_THRESHOLD,
    ANTI_SPAM_NO_INTENT_TURNS,
    ANTI_SPAM_BURST_MESSAGES,
    ANTI_SPAM_BURST_WINDOW_SEC,
    CONTACTS_RE,
    DEBUG_TOKEN,
    DEFAULT_CLIENT_ID,
    INPUT_MAX_CHARS,
    PORT,
    PRICE_CONCERN_RE,
    PRICE_LOOKUP_RE,
    RATE_LIMIT_MAX_PER_IP,
    RATE_LIMIT_WINDOW_SEC,
    resolve_client_id,
)
from lead_service import handle_lead
from logging_setup import get_logger, make_request_context, log_json
from chunk_responder import respond_from_chunk, respond_from_chunk_stream
from flow_handlers import handle_flows
from llm import classify_handoff_filter, classify_intent
from query_selector import select_catalog_content_route
from query_selector import select_chunk_for_question
from query_selector import select_price_service_route
from policy import (
    apply_response_policy,
    pick_contacts_chunk,
)
from retriever import (
    alias_hit_score_for_chunk,
    best_alias_hit_in_corpus,
    chunk_info,
    get_chunk_by_ref,
    normalize_retrieval_query,
    retrieve,
)
from session import (
    get_topic_state,
    mem_add_bot,
    mem_add_user,
    mem_get,
    mem_reset,
    parse_yes,
    record_last_bot_payload,
    set_last_catalog_service,
    set_anti_spam_redirect_shown,
    sid_from_body,
)
from ux_builder import (
    _format_price_value,
    build_price_clarify_payload,
    build_price_concern_payload,
    build_price_lookup_payload,
    build_service_facts_card_payload,
    empty_question_response,
    internal_error_response,
    low_score_response,
    no_candidates_response,
    offtopic_response,
    reset_session_response,
)

app = Flask(__name__, static_folder="static")
logger = get_logger("bot")
APP_ENV = (os.getenv("APP_ENV") or "local").strip().lower()
_APPLY_POLICY_PARAMS = inspect.signature(apply_response_policy).parameters
TXT = {
    "lead_name_prompt": "Отлично. Как к вам можно обращаться?",
    "lead_name_retry": "Как к вам можно обращаться? Напишите, пожалуйста, имя.",
    "lead_name_hard": "Напишите просто имя — например, Мария или Андрей.",
    "lead_name_invalid": "Не совсем поняла — напишите просто имя, например Мария.",
    "lead_name_confirm_tpl": "Вас зовут {name}, правильно?",
    "lead_name_reenter": "Хорошо. Как к вам можно обращаться?",
    "lead_phone_prompt_tpl": (
        "{name}, оставьте номер — администратор свяжется и ответит на все вопросы."
    ),
    "lead_phone_retry": "Не получилось распознать номер. Напишите в формате +7XXXXXXXXXX.",
    "lead_submit_ok": "Готово. Администратор свяжется с вами в ближайшее время.",
    "lead_submit_error": "Что-то пошло не так. Проверьте номер и попробуйте ещё раз.",
    "situation_prompt": (
        "Опишите коротко ситуацию — что болит, что беспокоит, или просто какой вопрос. "
        "Врач заранее будет в курсе и это поможет при консультации"
    ),
    "situation_retry_short": (
        "Напишите чуть подробнее — буквально 1–2 фразы. "
        "Чем точнее, тем лучше врач подготовится."
    ),
    "situation_to_lead_name": "Спасибо. Как к вам можно обращаться?",
    "situation_back_fallback": "Хорошо, продолжим. Задайте вопрос или выберите тему.",
    "followup_choose_topic": "Могу рассказать про этапы или про сроки — что выбрать?",
}
_IP_RATE_LOCK = threading.RLock()
_IP_RATE_BUCKETS: dict[str, deque] = {}
_OBVIOUS_NOISE_RE = re.compile(r"^[^А-Яа-яЁёA-Za-z]{4,}$", re.U)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}", re.U)


def _get_last_content_ui_payload_compat(sid: str) -> dict | None:
    fn = getattr(session_mod, "get_last_content_ui_payload", None)
    if callable(fn):
        return fn(sid)
    return None


def _apply_response_policy_compat(
    payload: dict,
    session_state: dict,
    q: str,
    *,
    topic_state: dict,
    doc_meta: dict,
    pre_doc_turn_count: int | None,
    session_id: str | None = None,
    client_id: str | None = None,
) -> dict:
    kw: dict = {
        "payload": payload,
        "session_state": session_state,
        "q": q,
        "topic_state": topic_state,
        "doc_meta": doc_meta,
    }
    if "pre_doc_turn_count" in _APPLY_POLICY_PARAMS:
        kw["pre_doc_turn_count"] = pre_doc_turn_count
    if "session_id" in _APPLY_POLICY_PARAMS:
        kw["session_id"] = session_id
    if "client_id" in _APPLY_POLICY_PARAMS:
        kw["client_id"] = client_id
    return apply_response_policy(**kw)


def _service_reply(
    payload: dict,
    sid: str,
    q: str,
    *,
    doc_id: str | None = None,
    track_user: bool = True,
):
    if track_user and q:
        mem_add_user(sid, q)
    answer = (payload.get("answer") or "").strip()
    out = finalize_ask(payload, sid, q, doc_id=doc_id)
    if answer:
        mem_add_bot(sid, answer)
    return safe_jsonify(out)


def _service_payload(
    answer: str,
    sid: str,
    client_id: str | None,
    *,
    lead_flow: bool = False,
    lead_step: str | None = None,
    situation_mode: str = "normal",
    situation_collect: bool = False,
    booking_intent_flag: bool = False,
    situation_back: bool = False,
    lead_error: str | None = None,
    quick_replies: list | None = None,
    cta: dict | None = None,
) -> dict:
    meta = {"sid": sid, "client_id": client_id}
    if lead_flow:
        meta["lead_flow"] = True
    if lead_step:
        meta["lead_step"] = lead_step
    if situation_collect:
        meta["situation_collect"] = True
    if booking_intent_flag:
        meta["booking_intent"] = True
    if situation_back:
        meta["situation_back"] = True
    if lead_error:
        meta["lead_error"] = lead_error
    return {
        "answer": answer,
        "quick_replies": list(quick_replies or []),
        "cta": cta,
        "video": None,
        "situation": {"show": situation_mode == "pending", "mode": situation_mode},
        "offer": None,
        "meta": meta,
    }


def _to_plain(o):
    import numpy as _np

    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)
    return o


def _sanitize(x):
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize(v) for v in x]
    return _to_plain(x)


def _is_short_contextual(q: str, st: dict) -> bool:
    """True если запрос короткий и без явного интента — нет смысла гнать в retrieval."""
    tokens = q.split()
    if len(tokens) > 3:
        return False
    if PRICE_LOOKUP_RE.search(q) or PRICE_CONCERN_RE.search(q) or CONTACTS_RE.search(q):
        return False
    # parse_yes уже обработан в handle_flows для known состояний.
    # Здесь ловим его только если last_bot_action == "none" (нет pending действия).
    last_action = st.get("last_bot_action") or "none"
    if parse_yes(q) and last_action == "none":
        return True
    # Короткие нейтральные реплики: "понятно", "спасибо", "хм", "ясно" и т.п.
    _NEUTRAL_RX = re.compile(
        r"^(понятно|спасибо|хм+|ясно|окей|ок|ok|интересно|угу|ага|ладно|"
        r"хорошо|понял|поняла|ничего|неплохо|круто|отлично|супер)\W*$",
        re.I,
    )
    if _NEUTRAL_RX.search(q):
        return True
    return False


def _normalize_question_text(text: str) -> tuple[str, bool]:
    q = (text or "").strip()
    if len(q) <= INPUT_MAX_CHARS:
        return q, False
    return q[:INPUT_MAX_CHARS], True


def _resolve_request_ip() -> str:
    xff = (request.headers.get("X-Forwarded-For") or "").strip()
    if xff:
        return xff.split(",", 1)[0].strip() or (request.remote_addr or "unknown")
    return request.remote_addr or "unknown"


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    with _IP_RATE_LOCK:
        q = _IP_RATE_BUCKETS.get(ip)
        if q is None:
            q = deque()
            _IP_RATE_BUCKETS[ip] = q
        threshold = now - float(RATE_LIMIT_WINDOW_SEC)
        while q and q[0] < threshold:
            q.popleft()
        if len(q) >= int(RATE_LIMIT_MAX_PER_IP):
            return False
        q.append(now)
        return True


def _rate_limited_response_payload() -> dict:
    return {
        "answer": "Слишком много запросов за короткое время. Подождите немного и попробуйте снова.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"error": "rate_limited"},
    }


def _handoff_filter_payload(
    sid: str,
    client_id: str | None,
    *,
    reason: str,
) -> dict:
    no_cta_reasons = {
        "spam",
        "flood",
        "trolling",
        "abuse",
        "offtopic",
        "prompt_injection",
        "acute_symptom",
        "medical_advice",
        "legal",
    }
    use_cta = (reason or "").strip().lower() not in no_cta_reasons
    return {
        "answer": (
            "Понимаю. Такой вопрос лучше передать администратору, чтобы вам ответили "
            "корректно и без лишних ожиданий. Оставьте, пожалуйста, контакт — "
            "администратор свяжется с вами. Если ситуация срочная, пожалуйста, не ждите "
            "ответа в чате — позвоните в клинику напрямую."
        ),
        "quick_replies": [],
        "cta": {"text": "Связаться с администратором", "action": "lead"} if use_cta else None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {
            "sid": sid,
            "client_id": client_id,
            "handoff_filter": True,
            "handoff_label": "handoff",
            "handoff_reason": (reason or "unspecified")[:64],
        },
    }


def _is_obvious_noise(q: str) -> bool:
    s = (q or "").strip()
    if not s:
        return False
    if len(s) <= 3 and not any(ch.isalpha() for ch in s):
        return True
    if _OBVIOUS_NOISE_RE.fullmatch(s):
        return True
    if _REPEATED_CHAR_RE.search(s):
        return True
    return False


def _obvious_noise_payload(sid: str, client_id: str | None) -> dict:
    return {
        "answer": "Похоже, сообщение не распознано. Я помогу по вопросам клиники, записи, услуг и стоимости.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"sid": sid, "client_id": client_id, "obvious_noise": True},
    }


def _norm_dup_text(q: str) -> str:
    x = (q or "").strip().lower().replace("ё", "е")
    x = re.sub(r"[^\w\s]", " ", x, flags=re.U)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _is_duplicate_question(st: dict, q: str) -> bool:
    qn = _norm_dup_text(q)
    if len(qn) < 5:
        return False
    hist = list((st or {}).get("hist") or [])
    last_users = [m.get("content", "") for m in hist if isinstance(m, dict) and m.get("role") == "user"]
    if not last_users:
        return False
    recent = last_users[-2:]
    return any(_norm_dup_text(x) == qn for x in recent)


def _duplicate_payload(sid: str, client_id: str | None, snap: dict | None) -> dict:
    quick = list((snap or {}).get("quick_replies") or [])
    cta = (snap or {}).get("cta")
    return {
        "answer": "Похоже, мы это уже обсудили чуть выше. Если хотите, могу продолжить по следующему шагу.",
        "quick_replies": quick[:2],
        "cta": cta if isinstance(cta, dict) else None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"sid": sid, "client_id": client_id, "duplicate_short_circuit": True},
    }


def _should_soft_redirect_no_intent(st: dict) -> bool:
    turns = int((st or {}).get("session_turn_count") or 0)
    booking_ever = bool((st or {}).get("booking_intent_ever"))
    shown = bool((st or {}).get("anti_spam_redirect_shown"))
    return turns >= ANTI_SPAM_NO_INTENT_TURNS and (not booking_ever) and (not shown)


def _is_message_burst(st: dict) -> bool:
    ts = list((st or {}).get("user_turn_timestamps") or [])
    if not ts:
        return False
    now = time.time()
    recent = [
        float(x)
        for x in ts
        if isinstance(x, (int, float)) and float(x) >= (now - ANTI_SPAM_BURST_WINDOW_SEC)
    ]
    return len(recent) >= ANTI_SPAM_BURST_MESSAGES


def _soft_redirect_payload(sid: str, client_id: str | None) -> dict:
    payload = _service_payload(
        "Я рассказал уже довольно много о разных темах. Возможно, удобнее обсудить вашу ситуацию напрямую — консультация бесплатна, врач ответит на всё сразу.",
        sid,
        client_id,
        lead_flow=False,
        lead_step=None,
        quick_replies=[],
        cta={"text": "Связаться с администратором", "action": "lead"},
    )
    meta = payload.setdefault("meta", {})
    meta["anti_spam_soft_redirect"] = True
    return payload


def _with_default_anchor(md_entry_ref: str) -> str:
    ref = (md_entry_ref or "").strip()
    if not ref:
        return ""
    return ref if "#" in ref else f"{ref}#korotko"


def _load_prices_for_client(client_id: str | None) -> dict:
    cid = (client_id or DEFAULT_CLIENT_ID).strip() or DEFAULT_CLIENT_ID
    p = os.path.join("clients", cid, "prices.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _service_price_line_for_content(service: dict, client_id: str | None) -> str | None:
    if not isinstance(service, dict):
        return None
    if str(service.get("price_display") or "").strip().lower() != "always":
        return None
    price_key = str(service.get("price_key") or "").strip()
    if not price_key:
        return None
    prices = _load_prices_for_client(client_id)
    price_item = prices.get(price_key) if isinstance(prices, dict) else None
    if not isinstance(price_item, dict):
        return None
    rendered = _format_price_value(price_item)
    if not rendered:
        return None
    note = str(price_item.get("note") or "").strip()
    line = f"Стоимость: {rendered}."
    if note:
        line += f" Важно: {note}."
    return line


def safe_jsonify(payload):
    return jsonify(_sanitize(payload))


def finalize_ask(payload: dict, sid: str, q: str, *, doc_id: str | None = None) -> dict:
    record_last_bot_payload(sid, payload)
    st = mem_get(sid)
    meta = payload.setdefault("meta", {})
    session_turn_count = int(st.get("session_turn_count") or 0)
    if doc_id:
        tstate = get_topic_state(sid, doc_id)
        meta["turn_count"] = int(tstate.get("doc_turn_count") or 0)
    else:
        meta["turn_count"] = session_turn_count
    meta["session_turn_count"] = session_turn_count
    return payload


def _log_selection(
    *,
    q: str,
    chosen_chunk: dict,
    chosen_score,
    original_top_score,
    rerank_applied: bool,
):
    log_json(
        logger,
        "selection",
        question=q[:200],
        original_top_score=(
            round(float(original_top_score), 4) if original_top_score is not None else None
        ),
        rerank_applied=bool(rerank_applied),
        chosen=chunk_info(chosen_chunk, chosen_score),
    )


log_json(logger, "app_start", env=os.getenv("APP_ENV"), version=os.getenv("APP_VERSION"))


def _startup_check() -> None:
    emb_path = os.path.join("data", "embeddings.npy")
    corpus_path = os.path.join("data", "corpus.jsonl")
    service_catalog_path = os.path.join("clients", DEFAULT_CLIENT_ID, "service_catalog.json")
    prices_path = os.path.join("clients", DEFAULT_CLIENT_ID, "prices.json")

    if not os.path.isfile(emb_path):
        logger.error("startup_check_failed: embeddings file is missing: %s", emb_path)
        sys.exit(1)
    try:
        arr = np.load(emb_path)
        if not isinstance(arr, np.ndarray):
            logger.error("startup_check_failed: embeddings file is not a numpy array: %s", emb_path)
            sys.exit(1)
    except Exception as e:
        logger.error("startup_check_failed: cannot read embeddings file %s: %s", emb_path, e)
        sys.exit(1)

    if not os.path.isfile(corpus_path):
        logger.error("startup_check_failed: corpus file is missing: %s", corpus_path)
        sys.exit(1)
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            chunks = sum(1 for line in f if line.strip())
    except Exception as e:
        logger.error("startup_check_failed: cannot read corpus file %s: %s", corpus_path, e)
        sys.exit(1)
    if chunks == 0:
        logger.error("startup_check_failed: corpus file is empty: %s", corpus_path)
        sys.exit(1)

    if not os.path.isfile(service_catalog_path):
        logger.error("startup_check_failed: service catalog file is missing: %s", service_catalog_path)
        sys.exit(1)
    try:
        with open(service_catalog_path, "r", encoding="utf-8") as f:
            service_catalog = json.load(f)
        if not isinstance(service_catalog, dict):
            logger.error("startup_check_failed: service catalog must be a JSON object: %s", service_catalog_path)
            sys.exit(1)
    except Exception as e:
        logger.error("startup_check_failed: invalid service catalog file %s: %s", service_catalog_path, e)
        sys.exit(1)

    if not os.path.isfile(prices_path):
        logger.error("startup_check_failed: prices file is missing: %s", prices_path)
        sys.exit(1)
    try:
        with open(prices_path, "r", encoding="utf-8") as f:
            prices = json.load(f)
        if not isinstance(prices, dict):
            logger.error("startup_check_failed: prices must be a JSON object: %s", prices_path)
            sys.exit(1)
    except Exception as e:
        logger.error("startup_check_failed: invalid prices file %s: %s", prices_path, e)
        sys.exit(1)

    log_json(logger, "startup_check_ok", chunks=chunks, services=len(service_catalog))


_startup_check()


@app.before_request
def _before():
    request.ctx = make_request_context(session_id=request.cookies.get("sid"))
    request.ctx["path"] = request.path
    request.ctx["method"] = request.method
    request.ctx["t0"] = time.time()


@app.after_request
def _after(resp):
    latency = int((time.time() - request.ctx["t0"]) * 1000)
    log_json(
        logger,
        "http_request",
        **{
            **request.ctx,
            "status": resp.status_code,
            "latency_ms": latency,
            "ip": request.remote_addr,
        },
    )
    return resp


@app.get("/_debug/ping")
def debug_ping():
    if APP_ENV == "prod":
        return jsonify({"error": "not_found"}), 404
    if request.headers.get("X-Debug-Token") != DEBUG_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"ok": True})


@app.post("/ask")
def ask():
    q = ""
    try:
        data = request.get_json(force=True) or {}
        client_id = resolve_client_id(data.get("client_id"))
        if client_id is None:
            return jsonify({"error": "unknown_client"}), 403

        q_raw = data.get("q") or ""
        q = (q_raw or "").strip()
        ref = (data.get("ref") or "").strip()
        sid = sid_from_body(data)

        if q and q.lower() in ("/reset", "/новая"):
            mem_reset(sid)
            return safe_jsonify(reset_session_response(sid))
        q, truncated = _normalize_question_text(q_raw)
        if truncated:
            log_json(logger, "input_truncated", sid=sid, client_id=client_id, original_len=len((q_raw or "").strip()), max_len=INPUT_MAX_CHARS)
        ip = _resolve_request_ip()
        if not _check_rate_limit(ip):
            log_json(logger, "rate_limited", sid=sid, client_id=client_id, ip=ip)
            return safe_jsonify(_rate_limited_response_payload()), 429
        if _is_obvious_noise(q):
            log_json(logger, "obvious_noise_short_circuit", sid=sid, client_id=client_id)
            return _service_reply(_obvious_noise_payload(sid, client_id), sid, q)
        if q:
            hf = classify_handoff_filter(q, client_id=client_id, sid=sid)
            label = str(hf.get("label") or "").lower()
            reason = str(hf.get("reason") or "unspecified").lower()
            confidence = float(hf.get("confidence") or 0.0)
            is_handoff = label == "handoff"
            log_json(
                logger,
                "handoff_filter_gate",
                sid=sid,
                client_id=client_id,
                label=label,
                reason=reason[:64],
                confidence=round(confidence, 4),
                is_handoff=is_handoff,
            )
            if is_handoff:
                return _service_reply(
                    _handoff_filter_payload(
                        sid=sid,
                        client_id=client_id,
                        reason=reason,
                    ),
                    sid,
                    q,
                )

        st = mem_get(sid)

        flow_result = handle_flows(
            data=data,
            st=st,
            sid=sid,
            q=q,
            client_id=client_id,
            txt=TXT,
            service_payload=_service_payload,
            get_last_content_ui_payload=_get_last_content_ui_payload_compat,
            get_topic_state=get_topic_state,
        )
        if flow_result is not None:
            redirect_ref = (flow_result.get("redirect_ref") or "").strip()
            if redirect_ref:
                ch = get_chunk_by_ref(redirect_ref, client_id=client_id)
                if ch:
                    return respond_from_chunk(
                        chunk=ch,
                        q=q,
                        sid=sid,
                        client_id=client_id,
                        finalize_ask=finalize_ask,
                        safe_jsonify=safe_jsonify,
                        logger=logger,
                        llm_question=q or f"Информация из {redirect_ref}",
                        log_event="Answer generated from flow redirect_ref",
                    )
            return _service_reply(
                flow_result["payload"],
                sid,
                q,
                doc_id=flow_result.get("doc_id"),
            )

        st = mem_get(sid)
        if _is_duplicate_question(st, q):
            snap = _get_last_content_ui_payload_compat(sid)
            log_json(logger, "duplicate_short_circuit", sid=sid, client_id=client_id)
            return _service_reply(_duplicate_payload(sid, client_id, snap), sid, q)
        if _is_message_burst(st):
            set_anti_spam_redirect_shown(sid, True)
            log_json(
                logger,
                "anti_spam_burst_redirect",
                sid=sid,
                client_id=client_id,
                burst_window_sec=ANTI_SPAM_BURST_WINDOW_SEC,
                burst_messages=ANTI_SPAM_BURST_MESSAGES,
            )
            return _service_reply(_soft_redirect_payload(sid, client_id), sid, q)
        if _should_soft_redirect_no_intent(st):
            set_anti_spam_redirect_shown(sid, True)
            log_json(
                logger,
                "anti_spam_soft_redirect",
                sid=sid,
                client_id=client_id,
                session_turn_count=int(st.get("session_turn_count") or 0),
            )
            return _service_reply(_soft_redirect_payload(sid, client_id), sid, q)

        if ref:
            ch = get_chunk_by_ref(ref, client_id=client_id)
            if ch:
                return respond_from_chunk(
                    chunk=ch,
                    q=q,
                    sid=sid,
                    client_id=client_id,
                    finalize_ask=finalize_ask,
                    safe_jsonify=safe_jsonify,
                    logger=logger,
                    llm_question=q or f"Информация из {ref}",
                    log_event="Answer generated from ref",
                )

        if not q:
            return _service_reply(empty_question_response(), sid, q, track_user=False)

        # Короткие реплики без явного интента — обрабатываем через контекст сессии.
        # Предотвращает падение "да", "понятно", "хорошо" в retrieval с низким score.
        if _is_short_contextual(q, st):
            current_doc_id = (st.get("current_doc_id") or "").strip()
            if current_doc_id:
                ch = get_chunk_by_ref(f"{current_doc_id}#korotko", client_id=client_id)
                if ch:
                    return respond_from_chunk(
                        chunk=ch,
                        q=q,
                        sid=sid,
                        client_id=client_id,
                        finalize_ask=finalize_ask,
                        safe_jsonify=safe_jsonify,
                        logger=logger,
                        llm_question=q,
                        log_event="Answer from short_contextual fallback",
                    )

        intent = classify_intent(q, client_id=client_id, sid=sid)

        if intent == "offtopic":
            return _service_reply(offtopic_response(), sid, q)

        if intent == "contacts":
            cands = retrieve(q, topk=4, client_id=client_id)
            picked = pick_contacts_chunk(cands)
            if picked:
                return respond_from_chunk(
                    chunk=picked,
                    q=q,
                    sid=sid,
                    client_id=client_id,
                    finalize_ask=finalize_ask,
                    safe_jsonify=safe_jsonify,
                    logger=logger,
                    llm_question=q,
                    log_event="Answer generated from contacts intent",
                )

        if intent in ("price_lookup", "price_concern"):
            price_route = select_price_service_route(
                q,
                client_id=client_id,
                sid=sid,
                intent_override=intent,
            )
            if price_route.get("mode") == "clarify":
                payload = build_price_clarify_payload(
                    sid=sid,
                    client_id=client_id,
                    intent=str(price_route.get("intent") or "other"),
                    fallback_reason=str(price_route.get("fallback_reason") or "service_not_found"),
                )
                log_json(logger, "price_route", **(payload.get("meta") or {}))
                return _service_reply(payload, sid, q)
            if price_route.get("mode") == "matched":
                intent = str(price_route.get("intent") or "other")
                service = price_route.get("service") or {}
                service_id = str(price_route.get("matched_service_id") or "")
                match_score = float(price_route.get("match_score") or 0.0)
                route_source = str(price_route.get("route_source") or "catalog")
                if service_id:
                    set_last_catalog_service(sid, service_id)
                if intent == "price_concern":
                    concern_ref = str(service.get("concern_ref") or "").strip()
                    if concern_ref:
                        ch = get_chunk_by_ref(concern_ref, client_id=client_id)
                        if ch:
                            log_json(
                                logger,
                                "price_route",
                                intent="price_concern",
                                matched_service_id=service_id,
                                match_score=round(match_score, 4),
                                route_source="concern_ref",
                                concern_ref=concern_ref,
                                fallback_reason=None,
                            )
                            return respond_from_chunk(
                                chunk=ch,
                                q=q,
                                sid=sid,
                                client_id=client_id,
                                finalize_ask=finalize_ask,
                                safe_jsonify=safe_jsonify,
                                logger=logger,
                                llm_question=q,
                                log_event="Answer generated from concern_ref",
                            )
                    payload = build_price_concern_payload(
                        sid=sid,
                        client_id=client_id,
                        service_id=service_id,
                        service=service,
                        match_score=match_score,
                    )
                    log_json(logger, "price_route", **(payload.get("meta") or {}))
                    return _service_reply(payload, sid, q)
                if route_source == "price_ref" and price_route.get("price_ref"):
                    ref = str(price_route.get("price_ref") or "").strip()
                    ch = get_chunk_by_ref(ref, client_id=client_id)
                    if ch:
                        log_json(
                            logger,
                            "price_route",
                            intent="price_lookup",
                            matched_service_id=service_id,
                            match_score=round(match_score, 4),
                            route_source="price_ref",
                            price_key=price_route.get("price_key"),
                            price_ref=ref,
                            fallback_reason=None,
                        )
                        return respond_from_chunk(
                            chunk=ch,
                            q=q,
                            sid=sid,
                            client_id=client_id,
                            finalize_ask=finalize_ask,
                            safe_jsonify=safe_jsonify,
                            logger=logger,
                            llm_question=q or f"Цена по {ref}",
                            log_event="Answer generated from price_ref",
                        )
                payload = build_price_lookup_payload(
                    sid=sid,
                    client_id=client_id,
                    service_id=service_id,
                    service=service,
                    match_score=match_score,
                    route_source=route_source,
                    price_key=price_route.get("price_key"),
                    price_ref=price_route.get("price_ref"),
                    price_item=price_route.get("price_item"),
                )
                log_json(logger, "price_route", **(payload.get("meta") or {}))
                return _service_reply(payload, sid, q)

        if intent == "content":
            cat = select_catalog_content_route(q, client_id=client_id)
            if cat.get("mode") == "md_first":
                sid_svc = str(cat.get("matched_service_id") or "")
                md_ref = _with_default_anchor(str(cat.get("md_entry_ref") or ""))
                service = cat.get("service") or {}
                price_line = _service_price_line_for_content(service, client_id)
                if md_ref:
                    ch = get_chunk_by_ref(md_ref, client_id=client_id)
                    if ch:
                        log_json(
                            logger,
                            "catalog_route",
                            route="md_first",
                            matched_service_id=sid_svc,
                            match_score=cat.get("match_score"),
                            md_entry_ref=md_ref,
                        )
                        if sid_svc:
                            set_last_catalog_service(sid, sid_svc)
                        llm_q = q or f"Информация из {md_ref}"
                        if price_line:
                            llm_q = f"{llm_q}\n\nВажно: если это уместно, явно укажи в ответе: {price_line}"
                        return respond_from_chunk(
                            chunk=ch,
                            q=q,
                            sid=sid,
                            client_id=client_id,
                            finalize_ask=finalize_ask,
                            safe_jsonify=safe_jsonify,
                            logger=logger,
                            llm_question=llm_q,
                            log_event="Answer generated from md_entry_ref",
                        )
            if cat.get("mode") == "facts":
                svc = cat.get("service") or {}
                sid_svc = str(cat.get("matched_service_id") or "")
                payload = build_service_facts_card_payload(
                    sid=sid,
                    client_id=client_id,
                    service_id=sid_svc,
                    service=svc,
                    match_score=float(cat.get("match_score") or 0.0),
                    user_question=q,
                )
                price_line = _service_price_line_for_content(svc, client_id)
                if price_line:
                    base = (payload.get("answer") or "").strip()
                    payload["answer"] = f"{base}\n\n{price_line}" if base else price_line
                    payload.setdefault("meta", {})["price_display_applied"] = "always"
                log_json(
                    logger,
                    "catalog_route",
                    route="facts",
                    matched_service_id=sid_svc,
                    match_score=cat.get("match_score"),
                )
                if sid_svc:
                    set_last_catalog_service(sid, sid_svc)
                return _service_reply(payload, sid, q, doc_id=None)

        log_json(logger, "Processing question", question=q[:100], question_length=len(q))
        selection = select_chunk_for_question(q, client_id=client_id, sid=sid)
        mode = selection.get("mode")
        dmeta = selection.get("debug_meta") or {}
        if mode == "no_candidates":
            log_json(logger, "No candidates found", question=q[:50])
            return _service_reply(no_candidates_response(), sid, q)
        if mode == "low_score":
            log_json(logger, "low_score_fallback", **dmeta)
            st_ls = mem_get(sid)
            pls = low_score_response(sid, client_id)
            pls = _apply_response_policy_compat(
                pls,
                st_ls,
                q,
                topic_state={},
                doc_meta={},
                pre_doc_turn_count=None,
                session_id=sid,
                client_id=client_id,
            )
            return _service_reply(pls, sid, q)
        if mode == "chunk":
            final_chunk = selection.get("chunk")
            if not isinstance(final_chunk, dict):
                log_json(logger, "selection_invalid_chunk", debug_meta=dmeta)
                return _service_reply(no_candidates_response(), sid, q)
            if dmeta.get("selected_by") == "alias":
                log_json(
                    logger,
                    "alias_hit_selected",
                    alias_score=dmeta.get("alias_score"),
                    file=final_chunk.get("file"),
                    h2_id=final_chunk.get("h2_id"),
                    h3_id=final_chunk.get("h3_id"),
                )
            _log_selection(
                q=q,
                chosen_chunk=final_chunk,
                chosen_score=final_chunk.get("_score"),
                original_top_score=dmeta.get("top_score"),
                rerank_applied=bool(selection.get("rerank_applied")),
            )
            return respond_from_chunk(
                chunk=final_chunk,
                q=q,
                sid=sid,
                client_id=client_id,
                finalize_ask=finalize_ask,
                safe_jsonify=safe_jsonify,
                logger=logger,
            )
        log_json(logger, "selection_unknown_mode", mode=mode, debug_meta=dmeta)
        return _service_reply(no_candidates_response(), sid, q)

    except Exception as e:
        logger.exception("ask_failed", extra={"q": q, "err": str(e)})
        return safe_jsonify(internal_error_response()), 200


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # отключает буферизацию в nginx
}


def _sse_service_reply(
    payload: dict,
    sid: str,
    q: str,
    *,
    doc_id: str | None = None,
    track_user: bool = True,
):
    """Обёртка _service_reply для SSE: один event ui + done."""
    if track_user and q:
        mem_add_user(sid, q)
    answer = (payload.get("answer") or "").strip()
    out = finalize_ask(payload, sid, q, doc_id=doc_id)
    if answer:
        mem_add_bot(sid, answer)

    def _gen():
        yield f"event: ui\ndata: {json.dumps(_sanitize(out), ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return app.response_class(_gen(), mimetype="text/event-stream", headers=_SSE_HEADERS)


def _sse_chunk_response(
    chunk: dict,
    q: str,
    sid: str,
    client_id: str | None,
    *,
    llm_question: str | None = None,
    log_event: str = "Answer generated",
):
    """Стриминговый ответ из чанка через SSE."""
    return app.response_class(
        respond_from_chunk_stream(
            chunk=chunk,
            q=q,
            sid=sid,
            client_id=client_id,
            finalize_ask=finalize_ask,
            logger=logger,
            llm_question=llm_question,
            log_event=log_event,
        ),
        mimetype="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/ask/stream")
def ask_stream():
    """Стриминговый вариант /ask. Протокол SSE:
      event: text_delta  data: {"delta": "..."}   — токены ответа
      event: ui          data: {полный payload}    — UI элементы после генерации
      event: done        data: {}                  — конец стрима
    Direct-ответы (цены, контакты, flow) отдают сразу один ui + done без text_delta.
    """
    q = ""
    try:
        data = request.get_json(force=True) or {}
        client_id = resolve_client_id(data.get("client_id"))
        if client_id is None:
            return jsonify({"error": "unknown_client"}), 403

        q_raw = data.get("q") or ""
        q = (q_raw or "").strip()
        ref = (data.get("ref") or "").strip()
        sid = sid_from_body(data)

        if q and q.lower() in ("/reset", "/новая"):
            mem_reset(sid)
            return safe_jsonify(reset_session_response(sid))
        q, truncated = _normalize_question_text(q_raw)
        if truncated:
            log_json(logger, "input_truncated", sid=sid, client_id=client_id, original_len=len((q_raw or "").strip()), max_len=INPUT_MAX_CHARS)
        ip = _resolve_request_ip()
        if not _check_rate_limit(ip):
            log_json(logger, "rate_limited", sid=sid, client_id=client_id, ip=ip)
            return safe_jsonify(_rate_limited_response_payload()), 429
        if _is_obvious_noise(q):
            log_json(logger, "obvious_noise_short_circuit", sid=sid, client_id=client_id)
            return _sse_service_reply(_obvious_noise_payload(sid, client_id), sid, q)
        if q:
            hf = classify_handoff_filter(q, client_id=client_id, sid=sid)
            label = str(hf.get("label") or "").lower()
            reason = str(hf.get("reason") or "unspecified").lower()
            confidence = float(hf.get("confidence") or 0.0)
            is_handoff = label == "handoff"
            log_json(
                logger,
                "handoff_filter_gate",
                sid=sid,
                client_id=client_id,
                label=label,
                reason=reason[:64],
                confidence=round(confidence, 4),
                is_handoff=is_handoff,
            )
            if is_handoff:
                return _sse_service_reply(
                    _handoff_filter_payload(
                        sid=sid,
                        client_id=client_id,
                        reason=reason,
                    ),
                    sid,
                    q,
                )

        st = mem_get(sid)

        flow_result = handle_flows(
            data=data,
            st=st,
            sid=sid,
            q=q,
            client_id=client_id,
            txt=TXT,
            service_payload=_service_payload,
            get_last_content_ui_payload=_get_last_content_ui_payload_compat,
            get_topic_state=get_topic_state,
        )
        if flow_result is not None:
            redirect_ref = (flow_result.get("redirect_ref") or "").strip()
            if redirect_ref:
                ch = get_chunk_by_ref(redirect_ref, client_id=client_id)
                if ch:
                    return _sse_chunk_response(
                        ch, q, sid, client_id,
                        llm_question=q or f"Информация из {redirect_ref}",
                        log_event="Answer generated from flow redirect_ref",
                    )
            return _sse_service_reply(
                flow_result["payload"], sid, q, doc_id=flow_result.get("doc_id")
            )

        st = mem_get(sid)
        if _is_duplicate_question(st, q):
            snap = _get_last_content_ui_payload_compat(sid)
            log_json(logger, "duplicate_short_circuit", sid=sid, client_id=client_id)
            return _sse_service_reply(_duplicate_payload(sid, client_id, snap), sid, q)
        if _is_message_burst(st):
            set_anti_spam_redirect_shown(sid, True)
            log_json(
                logger,
                "anti_spam_burst_redirect",
                sid=sid,
                client_id=client_id,
                burst_window_sec=ANTI_SPAM_BURST_WINDOW_SEC,
                burst_messages=ANTI_SPAM_BURST_MESSAGES,
            )
            return _sse_service_reply(_soft_redirect_payload(sid, client_id), sid, q)
        if _should_soft_redirect_no_intent(st):
            set_anti_spam_redirect_shown(sid, True)
            log_json(
                logger,
                "anti_spam_soft_redirect",
                sid=sid,
                client_id=client_id,
                session_turn_count=int(st.get("session_turn_count") or 0),
            )
            return _sse_service_reply(_soft_redirect_payload(sid, client_id), sid, q)

        if ref:
            ch = get_chunk_by_ref(ref, client_id=client_id)
            if ch:
                return _sse_chunk_response(
                    ch, q, sid, client_id,
                    llm_question=q or f"Информация из {ref}",
                    log_event="Answer generated from ref",
                )

        if not q:
            return _sse_service_reply(empty_question_response(), sid, q, track_user=False)

        if _is_short_contextual(q, st):
            current_doc_id = (st.get("current_doc_id") or "").strip()
            if current_doc_id:
                ch = get_chunk_by_ref(f"{current_doc_id}#korotko", client_id=client_id)
                if ch:
                    return _sse_chunk_response(
                        ch, q, sid, client_id,
                        log_event="Answer from short_contextual fallback",
                    )

        intent = classify_intent(q, client_id=client_id, sid=sid)

        if intent == "offtopic":
            return _sse_service_reply(offtopic_response(), sid, q)

        if intent == "contacts":
            cands = retrieve(q, topk=4, client_id=client_id)
            picked = pick_contacts_chunk(cands)
            if picked:
                return _sse_chunk_response(
                    picked, q, sid, client_id,
                    log_event="Answer generated from contacts intent",
                )

        if intent in ("price_lookup", "price_concern"):
            price_route = select_price_service_route(
                q, client_id=client_id, sid=sid, intent_override=intent,
            )
            if price_route.get("mode") == "clarify":
                payload = build_price_clarify_payload(
                    sid=sid,
                    client_id=client_id,
                    intent=str(price_route.get("intent") or "other"),
                    fallback_reason=str(price_route.get("fallback_reason") or "service_not_found"),
                )
                log_json(logger, "price_route", **(payload.get("meta") or {}))
                return _sse_service_reply(payload, sid, q)
            if price_route.get("mode") == "matched":
                intent = str(price_route.get("intent") or "other")
                service = price_route.get("service") or {}
                service_id = str(price_route.get("matched_service_id") or "")
                match_score = float(price_route.get("match_score") or 0.0)
                route_source = str(price_route.get("route_source") or "catalog")
                if service_id:
                    set_last_catalog_service(sid, service_id)
                if intent == "price_concern":
                    concern_ref = str(service.get("concern_ref") or "").strip()
                    if concern_ref:
                        ch = get_chunk_by_ref(concern_ref, client_id=client_id)
                        if ch:
                            log_json(
                                logger, "price_route",
                                intent="price_concern",
                                matched_service_id=service_id,
                                match_score=round(match_score, 4),
                                route_source="concern_ref",
                                concern_ref=concern_ref,
                                fallback_reason=None,
                            )
                            return _sse_chunk_response(
                                ch, q, sid, client_id,
                                log_event="Answer generated from concern_ref",
                            )
                    payload = build_price_concern_payload(
                        sid=sid, client_id=client_id,
                        service_id=service_id, service=service, match_score=match_score,
                    )
                    log_json(logger, "price_route", **(payload.get("meta") or {}))
                    return _sse_service_reply(payload, sid, q)
                if route_source == "price_ref" and price_route.get("price_ref"):
                    ref = str(price_route.get("price_ref") or "").strip()
                    ch = get_chunk_by_ref(ref, client_id=client_id)
                    if ch:
                        log_json(
                            logger, "price_route",
                            intent="price_lookup",
                            matched_service_id=service_id,
                            match_score=round(match_score, 4),
                            route_source="price_ref",
                            price_key=price_route.get("price_key"),
                            price_ref=ref,
                            fallback_reason=None,
                        )
                        return _sse_chunk_response(
                            ch, q, sid, client_id,
                            llm_question=q or f"Цена по {ref}",
                            log_event="Answer generated from price_ref",
                        )
                payload = build_price_lookup_payload(
                    sid=sid, client_id=client_id,
                    service_id=service_id, service=service, match_score=match_score,
                    route_source=route_source,
                    price_key=price_route.get("price_key"),
                    price_ref=price_route.get("price_ref"),
                    price_item=price_route.get("price_item"),
                )
                log_json(logger, "price_route", **(payload.get("meta") or {}))
                return _sse_service_reply(payload, sid, q)

        if intent == "content":
            cat = select_catalog_content_route(q, client_id=client_id)
            if cat.get("mode") == "md_first":
                sid_svc = str(cat.get("matched_service_id") or "")
                md_ref = _with_default_anchor(str(cat.get("md_entry_ref") or ""))
                service = cat.get("service") or {}
                price_line = _service_price_line_for_content(service, client_id)
                if md_ref:
                    ch = get_chunk_by_ref(md_ref, client_id=client_id)
                    if ch:
                        log_json(
                            logger,
                            "catalog_route",
                            route="md_first",
                            matched_service_id=sid_svc,
                            match_score=cat.get("match_score"),
                            md_entry_ref=md_ref,
                        )
                        if sid_svc:
                            set_last_catalog_service(sid, sid_svc)
                        llm_q = q or f"Информация из {md_ref}"
                        if price_line:
                            llm_q = f"{llm_q}\n\nВажно: если это уместно, явно укажи в ответе: {price_line}"
                        return _sse_chunk_response(
                            ch, q, sid, client_id,
                            llm_question=llm_q,
                            log_event="Answer generated from md_entry_ref",
                        )
            if cat.get("mode") == "facts":
                svc = cat.get("service") or {}
                sid_svc = str(cat.get("matched_service_id") or "")
                payload = build_service_facts_card_payload(
                    sid=sid, client_id=client_id,
                    service_id=sid_svc, service=svc,
                    match_score=float(cat.get("match_score") or 0.0),
                    user_question=q,
                )
                price_line = _service_price_line_for_content(svc, client_id)
                if price_line:
                    base = (payload.get("answer") or "").strip()
                    payload["answer"] = f"{base}\n\n{price_line}" if base else price_line
                    payload.setdefault("meta", {})["price_display_applied"] = "always"
                log_json(logger, "catalog_route", route="facts",
                         matched_service_id=sid_svc, match_score=cat.get("match_score"))
                if sid_svc:
                    set_last_catalog_service(sid, sid_svc)
                return _sse_service_reply(payload, sid, q, doc_id=None)

        log_json(logger, "Processing question", question=q[:100], question_length=len(q))
        selection = select_chunk_for_question(q, client_id=client_id, sid=sid)
        mode = selection.get("mode")
        dmeta = selection.get("debug_meta") or {}
        if mode == "no_candidates":
            log_json(logger, "No candidates found", question=q[:50])
            return _sse_service_reply(no_candidates_response(), sid, q)
        if mode == "low_score":
            log_json(logger, "low_score_fallback", **dmeta)
            st_ls = mem_get(sid)
            pls = low_score_response(sid, client_id)
            pls = _apply_response_policy_compat(
                pls, st_ls, q,
                topic_state={}, doc_meta={},
                pre_doc_turn_count=None,
                session_id=sid, client_id=client_id,
            )
            return _sse_service_reply(pls, sid, q)
        if mode == "chunk":
            final_chunk = selection.get("chunk")
            if not isinstance(final_chunk, dict):
                log_json(logger, "selection_invalid_chunk", debug_meta=dmeta)
                return _sse_service_reply(no_candidates_response(), sid, q)
            if dmeta.get("selected_by") == "alias":
                log_json(
                    logger, "alias_hit_selected",
                    alias_score=dmeta.get("alias_score"),
                    file=final_chunk.get("file"),
                    h2_id=final_chunk.get("h2_id"),
                    h3_id=final_chunk.get("h3_id"),
                )
            _log_selection(
                q=q,
                chosen_chunk=final_chunk,
                chosen_score=final_chunk.get("_score"),
                original_top_score=dmeta.get("top_score"),
                rerank_applied=bool(selection.get("rerank_applied")),
            )
            return _sse_chunk_response(final_chunk, q, sid, client_id)
        log_json(logger, "selection_unknown_mode", mode=mode, debug_meta=dmeta)
        return _sse_service_reply(no_candidates_response(), sid, q)

    except Exception as e:
        logger.exception("ask_stream_failed", extra={"q": q, "err": str(e)})
        return safe_jsonify(internal_error_response()), 200


@app.get("/__debug/retrieval")
def dbg():
    if APP_ENV == "prod":
        return jsonify({"error": "not_found"}), 404
    if request.headers.get("X-Debug-Token") != DEBUG_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    q = request.args.get("q", "")
    client_id = resolve_client_id(request.args.get("client_id"))
    if client_id is None:
        return jsonify({"error": "unknown_client"}), 403
    q_raw = (q or "").strip()
    q_use = normalize_retrieval_query(q_raw) or q_raw
    c = retrieve(q_raw, topk=5, client_id=client_id)
    alias_selected, alias_score = best_alias_hit_in_corpus(
        q_use,
        client_id=client_id,
        strong_threshold=ALIAS_STRONG_THRESHOLD,
    )
    for x in c:
        x["alias_score"] = alias_hit_score_for_chunk(q_use, x)
        x.pop("text", None)
    alias_summary = None
    if isinstance(alias_selected, dict):
        alias_summary = {
            "file": alias_selected.get("file"),
            "h2_id": alias_selected.get("h2_id"),
            "h3_id": alias_selected.get("h3_id"),
            "score": alias_selected.get("_score"),
        }
    return jsonify(
        {
            "q": q,
            "client_id": client_id,
            "alias_score": round(float(alias_score or 0.0), 4),
            "alias_selected": alias_summary,
            "candidates": c,
        }
    )


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.post("/lead")
def create_lead():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error_code": "bad_json", "delivery": None}), 400
    client_id = resolve_client_id(data.get("client_id"))
    if client_id is None:
        return jsonify({"ok": False, "error_code": "unknown_client", "delivery": None}), 403
    data["client_id"] = client_id
    payload, status = handle_lead(data)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
