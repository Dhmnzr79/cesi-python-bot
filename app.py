import os
import time

from flask import Flask, jsonify, request, send_from_directory
import session as session_mod

from config import DEBUG_TOKEN, LOW_SCORE_THRESHOLD, PORT, resolve_client_id
from lead_service import handle_lead
from llm import generate_answer_with_empathy
from logging_setup import get_logger, make_request_context, log_json
from meta_loader import get_doc_meta
from policy import (
    apply_response_policy,
    booking_intent,
    contacts_intent,
    pick_contacts_chunk,
    pick_prices_chunk,
    price_intent,
)
from retriever import (
    alias_hit_score_for_chunk,
    best_alias_hit_in_corpus,
    broad_query_detect,
    chunk_info,
    get_chunk_by_ref,
    is_point_literal_query,
    llm_rerank,
    prefer_overview_if_broad,
    retrieve,
)
from session import (
    defer_refs,
    extract_name,
    extract_phone,
    get_topic_state,
    increment_doc_turn_if_contentful,
    is_active_lead_flow,
    mark_h3_covered,
    mark_situation_offered,
    mark_video_pending,
    mark_video_shown,
    pop_deferred_ref,
    set_cta_shown,
    set_current_doc,
    mem_get,
    mem_reset,
    parse_yes,
    record_last_bot_payload,
    set_lead_intent,
    set_situation_note,
    set_situation_pending,
    sid_from_body,
    update_profile,
)
from ux_builder import (
    build_ask_response,
    empty_question_response,
    internal_error_response,
    low_score_response,
    no_candidates_response,
    normalize_policy_payload,
    reset_session_response,
)

app = Flask(__name__, static_folder="static")
logger = get_logger("bot")


def _get_last_content_ui_payload_compat(sid: str) -> dict | None:
    fn = getattr(session_mod, "get_last_content_ui_payload", None)
    if callable(fn):
        return fn(sid)
    return None


def _mark_suggest_ref_used_compat(sid: str, doc_id: str, used: bool = True) -> None:
    fn = getattr(session_mod, "mark_suggest_ref_used", None)
    if callable(fn):
        fn(sid, doc_id, used)


def _increment_doc_turn_with_pre(
    sid: str,
    doc_id: str | None,
    *,
    contentful: bool,
    is_low_score: bool,
    is_error: bool,
    lead_flow_active: bool,
) -> int | None:
    pre_turn = increment_doc_turn_if_contentful(
        sid,
        doc_id,
        contentful=contentful,
        is_low_score=is_low_score,
        is_error=is_error,
        lead_flow_active=lead_flow_active,
    )
    if pre_turn is not None or not doc_id:
        return pre_turn
    # Backward compatibility: old session.py increments but returns None.
    if contentful and not is_low_score and not is_error and not lead_flow_active:
        cur = int((get_topic_state(sid, doc_id) or {}).get("doc_turn_count") or 0)
        if cur > 0:
            return cur - 1
    return None


def _apply_response_policy_compat(
    payload: dict,
    session_state: dict,
    q: str,
    *,
    topic_state: dict,
    doc_meta: dict,
    pre_doc_turn_count: int | None,
) -> dict:
    try:
        return apply_response_policy(
            payload,
            session_state,
            q,
            topic_state=topic_state,
            doc_meta=doc_meta,
            pre_doc_turn_count=pre_doc_turn_count,
        )
    except TypeError:
        # Backward compatibility for older policy.py without pre_doc_turn_count.
        return apply_response_policy(
            payload,
            session_state,
            q,
            topic_state=topic_state,
            doc_meta=doc_meta,
        )


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


def safe_jsonify(payload):
    return jsonify(_sanitize(payload))


def ensure_answer(answer: str, chunk: dict) -> str:
    if isinstance(answer, str) and answer.strip():
        return answer
    fallback = (chunk.get("text") or "").strip()
    return (fallback[:800] + ("…" if len(fallback) > 800 else "")) or (
        "Пока не нашёл точный ответ. Можете уточнить вопрос?"
    )


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


def _meta_for_chunk(chunk: dict, client_id: str | None = None) -> dict:
    meta = get_doc_meta(
        os.path.basename(chunk.get("file", "") or ""),
        client_id=client_id or chunk.get("client_id"),
    ) or {}
    meta = dict(meta)
    if not meta.get("doc_id"):
        meta["doc_id"] = os.path.splitext(os.path.basename(chunk.get("file", "") or ""))[0]
    return meta


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


def _respond_from_chunk(
    *,
    chunk: dict,
    q: str,
    sid: str,
    client_id: str | None,
    llm_question: str | None = None,
    log_event: str = "Answer generated",
):
    meta = _meta_for_chunk(chunk, client_id=client_id)
    doc_id = meta.get("doc_id")
    if doc_id:
        set_current_doc(sid, doc_id)

    answer, profile = generate_answer_with_empathy(
        llm_question or q, chunk.get("text", ""), meta, sid
    )
    answer = ensure_answer(answer, chunk)

    st = mem_get(sid)
    lead_flow_active = is_active_lead_flow(st)
    pre_turn = _increment_doc_turn_with_pre(
        sid,
        doc_id,
        contentful=bool(answer.strip()),
        is_low_score=False,
        is_error=False,
        lead_flow_active=lead_flow_active,
    )
    tstate = get_topic_state(sid, doc_id) if doc_id else {}
    suggest_h3 = set(meta.get("suggest_h3") or [])
    h3_id = chunk.get("h3_id")
    if h3_id and h3_id in suggest_h3:
        mark_h3_covered(sid, doc_id, h3_id)
        tstate = get_topic_state(sid, doc_id)

    payload = build_ask_response(
        answer=answer,
        top=chunk,
        meta=meta,
        sid=sid,
        profile=profile,
        client_id=client_id,
        topic_state=tstate,
    )
    payload = _apply_response_policy_compat(
        payload,
        st,
        q,
        topic_state=tstate,
        doc_meta=meta,
        pre_doc_turn_count=pre_turn,
    )
    refs_before_ui = list(payload.get("quick_replies") or [])
    payload = normalize_policy_payload(payload)
    pdec = (payload.get("meta") or {}).get("policy_decision") or {}
    ui_dropped = set((payload.get("meta") or {}).get("ui_dropped") or [])
    if doc_id:
        if bool(pdec.get("show_video")):
            mark_video_shown(sid, doc_id)
        elif meta.get("video_key") and not bool(get_topic_state(sid, doc_id).get("video_shown")):
            mark_video_pending(sid, doc_id, pending=True)

        sit = payload.get("situation") or {}
        if sit.get("show") and sit.get("mode") == "normal":
            mark_situation_offered(sid, doc_id)

        if bool(pdec.get("defer_refs")):
            defer_refs(sid, doc_id, pdec.get("refs_to_defer") or [])
        elif "refs_with_two_followups_conflict" in ui_dropped and refs_before_ui:
            defer_refs(sid, doc_id, refs_before_ui[:1])
        elif payload.get("quick_replies"):
            _mark_suggest_ref_used_compat(sid, doc_id, True)
            # if we showed a deferred ref, consume one from per-doc queue
            tstate_after = get_topic_state(sid, doc_id)
            if tstate_after.get("refs_deferred"):
                pop_deferred_ref(sid, doc_id)

    if payload.get("cta") and doc_id:
        set_cta_shown(sid, doc_id, shown=True)

    log_json(
        logger,
        log_event,
        file=chunk.get("file"),
        score=round(float(chunk.get("_score", 0.0)), 3),
        answer_length=len(answer),
    )
    return safe_jsonify(finalize_ask(payload, sid, q, doc_id=doc_id))


def _lead_flow_reply(sid: str, q: str, client_id: str | None):
    st = mem_get(sid)
    intent = (st.get("lead_intent") or "none").strip()

    if intent == "collecting_name":
        name = extract_name(q)
        if not name:
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Как к вам можно обращаться? Напишите, пожалуйста, имя.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                    },
                    sid,
                    q,
                )
            )
        update_profile(sid, name=name)
        set_lead_intent(sid, "collecting_phone")
        return safe_jsonify(
            finalize_ask(
                {
                    "answer": f"Спасибо, {name}. Оставьте номер телефона для связи.",
                    "quick_replies": [],
                    "cta": None,
                    "video": None,
                    "situation": {"show": False, "mode": "normal"},
                    "offer": None,
                    "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                },
                sid,
                q,
            )
        )

    if intent == "collecting_phone":
        phone = extract_phone(q)
        if not phone:
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Не получилось распознать номер. Напишите телефон в формате +7XXXXXXXXXX.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                    },
                    sid,
                    q,
                )
            )
        update_profile(sid, phone=phone)
        prof = mem_get(sid).get("profile") or {}
        lead_payload, lead_status = handle_lead(
            {
                "name": (prof.get("name") or "").strip(),
                "phone": (prof.get("phone") or "").strip(),
                "intent": "lead",
                "sid": sid,
                "client_id": client_id,
                "situation_note": (st.get("situation_note") or "").strip(),
            }
        )
        if lead_status != 200:
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Не удалось сохранить заявку. Проверьте номер телефона и попробуйте ещё раз.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True, "lead_error": lead_payload.get("error")},
                    },
                    sid,
                    q,
                )
            )
        set_lead_intent(sid, "submitted")
        set_situation_pending(sid, False)
        set_situation_note(sid, "")
        return safe_jsonify(
            finalize_ask(
                {
                    "answer": "Принято, передали заявку администратору. С вами свяжутся.",
                    "quick_replies": [],
                    "cta": None,
                    "video": None,
                    "situation": {"show": False, "mode": "normal"},
                    "offer": None,
                    "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                },
                sid,
                q,
            )
        )
    return None


log_json(logger, "app_start", env=os.getenv("APP_ENV"), version=os.getenv("APP_VERSION"))


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
    if request.headers.get("X-Debug-Token") and request.headers.get("X-Debug-Token") != DEBUG_TOKEN:
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

        q = (data.get("q") or "").strip()
        ref = (data.get("ref") or "").strip()
        sid = sid_from_body(data)

        if data.get("q") and data.get("q").strip().lower() in ("/reset", "/новая"):
            mem_reset(sid)
            return safe_jsonify(reset_session_response(sid))

        st = mem_get(sid)

        if data.get("situation_action") == "back":
            set_situation_pending(sid, False)
            snap = _get_last_content_ui_payload_compat(sid)
            if isinstance(snap, dict) and snap.get("answer"):
                restored = {
                    "answer": snap.get("answer") or "",
                    "quick_replies": list(snap.get("quick_replies") or []),
                    "cta": snap.get("cta"),
                    "video": snap.get("video"),
                    "situation": snap.get("situation") or {"show": False, "mode": "normal"},
                    "offer": snap.get("offer"),
                    "meta": dict(snap.get("meta") or {}),
                }
                doc_id_back = st.get("current_doc_id") or (
                    (restored.get("meta") or {}).get("file") and os.path.splitext(
                        os.path.basename((restored.get("meta") or {}).get("file") or "")
                    )[0]
                )
                if doc_id_back and get_topic_state(sid, doc_id_back).get("situation_offered"):
                    restored["situation"] = {"show": False, "mode": "normal"}
                meta_r = restored.setdefault("meta", {})
                meta_r["situation_back"] = True
                meta_r.setdefault("sid", sid)
                meta_r.setdefault("client_id", client_id)
                return safe_jsonify(
                    finalize_ask(
                        restored,
                        sid,
                        q,
                        doc_id=st.get("current_doc_id"),
                    )
                )
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Хорошо, продолжим. Задайте вопрос или выберите тему.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "situation_back": True},
                    },
                    sid,
                    q,
                    doc_id=st.get("current_doc_id"),
                )
            )

        if q and booking_intent(q) and not is_active_lead_flow(st):
            set_lead_intent(sid, "collecting_name")
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Отлично. Как к вам можно обращаться?",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True, "booking_intent": True},
                    },
                    sid,
                    q,
                )
            )

        if is_active_lead_flow(st):
            flow_resp = _lead_flow_reply(sid, q, client_id)
            if flow_resp is not None:
                return flow_resp

        if st.get("situation_pending"):
            if not q or len(q.strip()) < 3:
                return safe_jsonify(
                    finalize_ask(
                        {
                            "answer": "Чтобы помочь точнее, напишите коротко вашу ситуацию в 1–2 фразах.",
                            "quick_replies": [],
                            "cta": None,
                            "video": None,
                            "situation": {"show": True, "mode": "pending"},
                            "offer": None,
                            "meta": {"sid": sid, "client_id": client_id, "situation_collect": True},
                        },
                        sid,
                        q,
                    )
                )
            set_situation_note(sid, q)
            set_situation_pending(sid, False)
            set_lead_intent(sid, "collecting_name")
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Спасибо. Как к вам можно обращаться?",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                    },
                    sid,
                    q,
                )
            )

        if data.get("cta_action") == "lead":
            set_lead_intent(sid, "collecting_name")
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Отлично. Как к вам можно обращаться?",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                    },
                    sid,
                    q,
                )
            )

        if data.get("situation_action") == "start" or data.get("action") == "situation":
            set_situation_pending(sid, True)
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Понимаю, что у каждого ситуация своя. Напишите коротко, что вас беспокоит или какой у вас вопрос, буквально в 1–2 фразах.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": True, "mode": "pending"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "situation_collect": True},
                    },
                    sid,
                    q,
                )
            )

        if ref:
            ch = get_chunk_by_ref(ref, client_id=client_id)
            if ch:
                return _respond_from_chunk(
                    chunk=ch,
                    q=q,
                    sid=sid,
                    client_id=client_id,
                    llm_question=q or f"Информация из {ref}",
                    log_event="Answer generated from ref",
                )

        if not q:
            return safe_jsonify(empty_question_response())

        if st.get("last_bot_action") == "offered_situation" and parse_yes(q):
            set_situation_pending(sid, True)
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Понимаю, что у каждого ситуация своя. Напишите коротко, что вас беспокоит или какой у вас вопрос, буквально в 1–2 фразах.",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": True, "mode": "pending"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "situation_collect": True},
                    },
                    sid,
                    q,
                )
            )

        if st.get("last_bot_action") == "offered_cta" and parse_yes(q):
            set_lead_intent(sid, "collecting_name")
            return safe_jsonify(
                finalize_ask(
                    {
                        "answer": "Отлично. Как к вам можно обращаться?",
                        "quick_replies": [],
                        "cta": None,
                        "video": None,
                        "situation": {"show": False, "mode": "normal"},
                        "offer": None,
                        "meta": {"sid": sid, "client_id": client_id, "lead_flow": True},
                    },
                    sid,
                    q,
                )
            )

        log_json(logger, "Processing question", question=q[:100], question_length=len(q))

        cands = retrieve(q, topk=8, client_id=client_id)
        cands = prefer_overview_if_broad(cands, broad_query_detect(q))

        if not cands:
            log_json(logger, "No candidates found", question=q[:50])
            return safe_jsonify(no_candidates_response())

        is_contacts = contacts_intent(q)
        is_price = price_intent(q)
        alias_chunk, alias_score = best_alias_hit_in_corpus(
            q,
            client_id=client_id,
            strong_threshold=0.82,
        )
        alias_strong = alias_chunk is not None

        allow_low = alias_strong or (is_contacts and pick_contacts_chunk(cands)) or (
            is_price and pick_prices_chunk(cands)
        )
        if float(cands[0].get("_score") or 0) < LOW_SCORE_THRESHOLD and not allow_low:
            log_json(
                logger,
                "low_score_fallback",
                score=round(float(cands[0].get("_score") or 0), 4),
                threshold=LOW_SCORE_THRESHOLD,
                alias_score=alias_score,
            )
            st_ls = mem_get(sid)
            pls = low_score_response(sid, client_id)
            pls = _apply_response_policy_compat(
                pls,
                st_ls,
                q,
                topic_state={},
                doc_meta={},
                pre_doc_turn_count=None,
            )
            return safe_jsonify(finalize_ask(pls, sid, q))

        if is_contacts:
            picked = pick_contacts_chunk(cands)
            if picked is not None:
                final_chunk = picked
                final_score = final_chunk.get("_score")
                top_score = cands[0].get("_score") if cands else None
                _log_selection(
                    q=q,
                    chosen_chunk=final_chunk,
                    chosen_score=final_score,
                    original_top_score=top_score,
                    rerank_applied=False,
                )
                return _respond_from_chunk(
                    chunk=final_chunk,
                    q=q,
                    sid=sid,
                    client_id=client_id,
                )

        if is_price:
            picked = pick_prices_chunk(cands)
            if picked is not None:
                final_chunk = picked
                final_score = final_chunk.get("_score")
                _log_selection(
                    q=q,
                    chosen_chunk=final_chunk,
                    chosen_score=final_score,
                    original_top_score=(cands[0].get("_score") if cands else None),
                    rerank_applied=False,
                )
                return _respond_from_chunk(
                    chunk=final_chunk,
                    q=q,
                    sid=sid,
                    client_id=client_id,
                )

        if alias_strong and alias_chunk is not None:
            final_chunk = alias_chunk
            final_score = final_chunk.get("_score")
            _log_selection(
                q=q,
                chosen_chunk=final_chunk,
                chosen_score=final_score,
                original_top_score=(cands[0].get("_score") if cands else None),
                rerank_applied=False,
            )
            log_json(
                logger,
                "alias_hit_selected",
                alias_score=alias_score,
                file=final_chunk.get("file"),
                h2_id=final_chunk.get("h2_id"),
                h3_id=final_chunk.get("h3_id"),
            )
            return _respond_from_chunk(
                chunk=final_chunk,
                q=q,
                sid=sid,
                client_id=client_id,
            )

        top = cands[0]
        best = float(top["_score"])
        score_gap = (
            abs(float(cands[0].get("_score") or 0.0) - float(cands[1].get("_score") or 0.0))
            if len(cands) >= 2
            else 1.0
        )
        use_rerank = (
            0.45 <= best <= 0.62
            and len(cands) >= 2
            and score_gap <= 0.05
            and not is_point_literal_query(q)
            and not alias_strong
        )
        if use_rerank:
            log_json(
                logger,
                "Applying rerank",
                original_score=best,
                candidates_count=len(cands),
                score_gap=round(score_gap, 4),
            )
            top = llm_rerank(q, cands[:3])

        _log_selection(
            q=q,
            chosen_chunk=top,
            chosen_score=(top.get("_score") if isinstance(top, dict) else None),
            original_top_score=best,
            rerank_applied=bool(use_rerank),
        )
        return _respond_from_chunk(chunk=top, q=q, sid=sid, client_id=client_id)

    except Exception as e:
        logger.exception("ask_failed", extra={"q": q, "err": str(e)})
        return safe_jsonify(internal_error_response()), 200


@app.get("/__debug/retrieval")
def dbg():
    q = request.args.get("q", "")
    client_id = resolve_client_id(request.args.get("client_id"))
    if client_id is None:
        return jsonify({"error": "unknown_client"}), 403
    c = retrieve(q, topk=5, client_id=client_id)
    alias_selected, alias_score = best_alias_hit_in_corpus(
        q,
        client_id=client_id,
        strong_threshold=0.82,
    )
    for x in c:
        x["alias_score"] = alias_hit_score_for_chunk(q, x)
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
        return jsonify({"ok": False, "error": "bad_json"}), 400
    payload, status = handle_lead(data)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
