import os
import time

from flask import Flask, jsonify, request, send_from_directory

from config import DEBUG_TOKEN, LOW_SCORE_THRESHOLD, PORT, resolve_client_id
from lead_service import handle_lead
from llm import generate_answer_with_empathy
from logging_setup import get_logger, make_request_context, log_json
from meta_loader import get_doc_meta
from policy import (
    apply_response_policy,
    contacts_intent,
    pick_contacts_chunk,
    pick_prices_chunk,
    price_intent,
)
from retriever import (
    broad_query_detect,
    chunk_info,
    get_chunk_by_ref,
    llm_rerank,
    prefer_overview_if_broad,
    retrieve,
)
from session import mem_get, mem_reset, record_last_bot_payload, sid_from_body
from ux_builder import (
    build_ask_response,
    empty_question_response,
    internal_error_response,
    low_score_response,
    no_candidates_response,
    reset_session_response,
)

app = Flask(__name__, static_folder="static")
logger = get_logger("bot")


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


def finalize_ask(payload: dict, sid: str, q: str) -> dict:
    apply_response_policy(payload, mem_get(sid), q or "")
    record_last_bot_payload(sid, payload)
    st = mem_get(sid)
    payload.setdefault("meta", {})["turn_count"] = int(st.get("turn_count") or 0)
    return payload


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

        if ref:
            ch = get_chunk_by_ref(ref)
            if ch:
                top = ch
                md_file = top.get("file")
                meta_doc = get_doc_meta(os.path.basename(md_file or "")) or {}
                answer, profile = generate_answer_with_empathy(
                    q or f"Информация из {ref}",
                    top.get("text", ""),
                    meta_doc,
                    sid,
                )
                answer = ensure_answer(answer, top)
                log_json(
                    logger,
                    "Answer generated from ref",
                    file=top.get("file"),
                    score=round(float(top.get("_score", 0.0)), 3),
                    answer_length=len(answer),
                )
                return safe_jsonify(
                    finalize_ask(
                        build_ask_response(
                            answer=answer,
                            top=top,
                            meta=meta_doc,
                            sid=sid,
                            profile=profile,
                            client_id=client_id,
                        ),
                        sid,
                        q,
                    )
                )

        if not q:
            return safe_jsonify(empty_question_response())

        log_json(logger, "Processing question", question=q[:100], question_length=len(q))

        cands = retrieve(q, topk=3)
        cands = prefer_overview_if_broad(cands, broad_query_detect(q))

        if not cands:
            log_json(logger, "No candidates found", question=q[:50])
            return safe_jsonify(no_candidates_response())

        allow_low = (contacts_intent(q) and pick_contacts_chunk(cands)) or (
            price_intent(q) and pick_prices_chunk(cands)
        )
        if float(cands[0].get("_score") or 0) < LOW_SCORE_THRESHOLD and not allow_low:
            log_json(
                logger,
                "low_score_fallback",
                score=round(float(cands[0].get("_score") or 0), 4),
                threshold=LOW_SCORE_THRESHOLD,
            )
            return safe_jsonify(finalize_ask(low_score_response(sid, client_id), sid, q))

        if contacts_intent(q):
            picked = pick_contacts_chunk(cands)
            if picked is not None:
                final_chunk = picked
                final_score = final_chunk.get("_score")
                top_score = cands[0].get("_score") if cands else None
                log_json(
                    logger,
                    "selection",
                    question=q[:200],
                    original_top_score=(
                        round(float(top_score), 4) if top_score is not None else None
                    ),
                    rerank_applied=False,
                    chosen=chunk_info(final_chunk, final_score),
                )
                meta = get_doc_meta(os.path.basename(final_chunk.get("file", ""))) or {}
                answer, profile = generate_answer_with_empathy(
                    q, final_chunk.get("text", ""), meta, sid
                )
                answer = ensure_answer(answer, final_chunk)
                log_json(
                    logger,
                    "Answer generated",
                    file=final_chunk["file"],
                    score=round(float(final_score), 3),
                    answer_length=len(answer),
                )
                return safe_jsonify(
                    finalize_ask(
                        build_ask_response(
                            answer=answer,
                            top=final_chunk,
                            meta=meta,
                            sid=sid,
                            profile=profile,
                            client_id=client_id,
                        ),
                        sid,
                        q,
                    )
                )

        if price_intent(q):
            picked = pick_prices_chunk(cands)
            if picked is not None:
                final_chunk = picked
                final_score = final_chunk.get("_score")
                log_json(
                    logger,
                    "selection",
                    question=q[:200],
                    original_top_score=(
                        round(float(cands[0].get("_score")), 4) if cands else None
                    ),
                    rerank_applied=False,
                    chosen=chunk_info(final_chunk, final_score),
                )
                meta = get_doc_meta(os.path.basename(final_chunk.get("file", ""))) or {}
                answer, profile = generate_answer_with_empathy(
                    q, final_chunk.get("text", ""), meta, sid
                )
                answer = ensure_answer(answer, final_chunk)
                log_json(
                    logger,
                    "Answer generated",
                    file=final_chunk["file"],
                    score=round(float(final_score), 3),
                    answer_length=len(answer),
                )
                return safe_jsonify(
                    finalize_ask(
                        build_ask_response(
                            answer=answer,
                            top=final_chunk,
                            meta=meta,
                            sid=sid,
                            profile=profile,
                            client_id=client_id,
                        ),
                        sid,
                        q,
                    )
                )

        top = cands[0]
        best = float(top["_score"])
        use_rerank = 0.45 <= best <= 0.62 and len(cands) >= 2
        if use_rerank:
            log_json(logger, "Applying rerank", original_score=best, candidates_count=len(cands))
            top = llm_rerank(q, cands[:3])

        log_json(
            logger,
            "selection",
            question=q[:200],
            original_top_score=round(float(best), 4) if best is not None else None,
            rerank_applied=bool(use_rerank),
            chosen=chunk_info(top, top.get("_score") if isinstance(top, dict) else None),
        )

        meta = get_doc_meta(os.path.basename(top.get("file", ""))) or {}
        answer, profile = generate_answer_with_empathy(q, top.get("text", ""), meta, sid)
        answer = ensure_answer(answer, top)
        log_json(
            logger,
            "Answer generated",
            file=top["file"],
            score=round(float(top.get("_score", 0.0)), 3),
            answer_length=len(answer),
        )
        return safe_jsonify(
            finalize_ask(
                build_ask_response(
                    answer=answer,
                    top=top,
                    meta=meta,
                    sid=sid,
                    profile=profile,
                    client_id=client_id,
                ),
                sid,
                q,
            )
        )

    except Exception as e:
        logger.exception("ask_failed", extra={"q": q, "err": str(e)})
        return safe_jsonify(internal_error_response()), 200


@app.get("/__debug/retrieval")
def dbg():
    q = request.args.get("q", "")
    c = retrieve(q, topk=5)
    for x in c:
        x.pop("text", None)
    return jsonify({"q": q, "candidates": c})


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
