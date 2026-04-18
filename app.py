import os
import time
import inspect

from flask import Flask, jsonify, request, send_from_directory
import session as session_mod

from config import ALIAS_STRONG_THRESHOLD, DEBUG_TOKEN, PORT, resolve_client_id
from lead_service import handle_lead
from logging_setup import get_logger, make_request_context, log_json
from chunk_responder import respond_from_chunk
from flow_handlers import handle_flows
from query_selector import select_chunk_for_question
from policy import (
    apply_response_policy,
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
    record_last_bot_payload,
    sid_from_body,
)
from ux_builder import (
    empty_question_response,
    internal_error_response,
    low_score_response,
    no_candidates_response,
    reset_session_response,
)

app = Flask(__name__, static_folder="static")
logger = get_logger("bot")
_POLICY_SUPPORTS_PRE_DOC_TURN = "pre_doc_turn_count" in inspect.signature(
    apply_response_policy
).parameters
TXT = {
    "lead_name_prompt": "Отлично. Как к вам можно обращаться?",
    "lead_name_retry": "Как к вам можно обращаться? Напишите, пожалуйста, имя.",
    "lead_phone_prompt_tpl": "Спасибо, {name}. Оставьте номер телефона для связи.",
    "lead_phone_retry": "Не получилось распознать номер. Напишите телефон в формате +7XXXXXXXXXX.",
    "lead_submit_ok": "Принято, передали заявку администратору. С вами свяжутся.",
    "lead_submit_error": "Не удалось сохранить заявку. Проверьте номер телефона и попробуйте ещё раз.",
    "situation_prompt": (
        "Понимаю, что у каждого ситуация своя. Напишите коротко, что вас беспокоит "
        "или какой у вас вопрос, буквально в 1–2 фразах."
    ),
    "situation_retry_short": "Чтобы помочь точнее, напишите коротко вашу ситуацию в 1–2 фразах.",
    "situation_to_lead_name": "Спасибо. Как к вам можно обращаться?",
    "situation_back_fallback": "Хорошо, продолжим. Задайте вопрос или выберите тему.",
}


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
) -> dict:
    if _POLICY_SUPPORTS_PRE_DOC_TURN:
        return apply_response_policy(
            payload,
            session_state,
            q,
            topic_state=topic_state,
            doc_meta=doc_meta,
            pre_doc_turn_count=pre_doc_turn_count,
        )
    # Backward compatibility for older policy.py without pre_doc_turn_count.
    return apply_response_policy(
        payload,
        session_state,
        q,
        topic_state=topic_state,
        doc_meta=doc_meta,
    )


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
            return _service_reply(
                flow_result["payload"],
                sid,
                q,
                doc_id=flow_result.get("doc_id"),
            )

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


@app.get("/__debug/retrieval")
def dbg():
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
        return jsonify({"ok": False, "error": "bad_json"}), 400
    payload, status = handle_lead(data)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
