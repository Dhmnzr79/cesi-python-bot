"""Orchestration: chunk → LLM answer → policy → session side-effects → HTTP payload."""
from __future__ import annotations

import inspect
import json as _json
import os
from typing import Any, Callable

import session as session_mod
from llm import LLM_FALLBACK_ANSWER, generate_answer_stream, generate_answer_with_empathy
from logging_setup import log_json
from meta_loader import get_doc_meta
from policy import apply_response_policy
from session import (
    defer_refs,
    get_topic_state,
    increment_doc_turn_if_contentful,
    is_active_lead_flow,
    mark_h3_covered,
    mark_situation_offered,
    mark_video_pending,
    mark_video_shown,
    mem_add_bot,
    mem_add_user,
    mem_get,
    pop_deferred_ref,
    set_cta_shown,
    set_current_doc,
)
from ux_builder import build_ask_response, normalize_policy_payload

_APPLY_POLICY_PARAMS = inspect.signature(apply_response_policy).parameters


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


def chunk_context_md_for_llm(chunk: dict) -> str:
    """Контент для генерации: H2 + H3 + тело чанка (имя врача и т.п. часто только в h2)."""
    parts: list[str] = []
    h2 = (chunk.get("h2") or "").strip()
    h3 = (chunk.get("h3") or "").strip()
    body = (chunk.get("text") or "").strip()
    if h2:
        parts.append(h2)
    if h3:
        parts.append(h3)
    if body:
        parts.append(body)
    return "\n\n".join(parts) if parts else ""


def ensure_answer(answer: str, chunk: dict) -> str:
    if isinstance(answer, str) and answer.strip():
        return answer
    fallback = chunk_context_md_for_llm(chunk).strip() or (chunk.get("text") or "").strip()
    return (fallback[:800] + ("…" if len(fallback) > 800 else "")) or (
        "Пока не нашёл точный ответ. Можете уточнить вопрос?"
    )


def meta_for_chunk(chunk: dict, client_id: str | None = None) -> dict:
    meta = get_doc_meta(
        os.path.basename(chunk.get("file", "") or ""),
        client_id=client_id or chunk.get("client_id"),
    ) or {}
    meta = dict(meta)
    if not meta.get("doc_id"):
        meta["doc_id"] = os.path.splitext(os.path.basename(chunk.get("file", "") or ""))[0]
    return meta


def respond_from_chunk(
    *,
    chunk: dict,
    q: str,
    sid: str,
    client_id: str | None,
    finalize_ask: Callable[..., dict],
    safe_jsonify: Callable[[dict], Any],
    logger,
    llm_question: str | None = None,
    log_event: str = "Answer generated",
    route: str = "retrieval_chunk",
):
    if (q or "").strip():
        mem_add_user(sid, q)
    meta = meta_for_chunk(chunk, client_id=client_id)
    doc_id = meta.get("doc_id")
    if doc_id:
        set_current_doc(sid, doc_id)

    answer, profile = generate_answer_with_empathy(
        llm_question or q, chunk_context_md_for_llm(chunk), meta, sid
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
        session_id=sid,
        client_id=client_id,
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
    qs = (q or "").strip()
    turn_meta = (
        {"interaction": "user_message", "question_len": len(qs), "preview": qs[:120]}
        if qs
        else None
    )
    out = finalize_ask(payload, sid, q, doc_id=doc_id, turn_meta=turn_meta, route=route)
    if answer.strip():
        mem_add_bot(sid, answer)
    return safe_jsonify(out)


def respond_from_chunk_stream(
    *,
    chunk: dict,
    q: str,
    sid: str,
    client_id: str | None,
    finalize_ask: Callable[..., dict],
    logger,
    llm_question: str | None = None,
    log_event: str = "Answer generated",
    route: str = "retrieval_chunk",
):
    """Generator yielding SSE strings: text_delta → ui → done.

    Используй с Flask: Response(respond_from_chunk_stream(...), mimetype='text/event-stream')
    Полностью зеркалит respond_from_chunk, но стримит токены ответа.
    """
    if (q or "").strip():
        mem_add_user(sid, q)
    meta = meta_for_chunk(chunk, client_id=client_id)
    doc_id = meta.get("doc_id")
    if doc_id:
        set_current_doc(sid, doc_id)

    full_text = ""
    profile: dict = {}

    try:
        for event_type, value in generate_answer_stream(
            llm_question or q, chunk_context_md_for_llm(chunk), meta, sid
        ):
            if event_type == "delta":
                full_text += value
                yield f"event: text_delta\ndata: {_json.dumps({'delta': value}, ensure_ascii=False)}\n\n"
            elif event_type == "done":
                full_text, profile = value
    except Exception as e:
        log_json(logger, "stream_chunk_failed", sid=sid, err=str(e)[:300])
        if not full_text.strip():
            full_text = LLM_FALLBACK_ANSWER

    answer = ensure_answer(full_text, chunk)

    # Все session side-effects — идентично respond_from_chunk
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
        session_id=sid,
        client_id=client_id,
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
    qs = (q or "").strip()
    turn_meta = (
        {"interaction": "user_message", "question_len": len(qs), "preview": qs[:120]}
        if qs
        else None
    )
    final = finalize_ask(payload, sid, q, doc_id=doc_id, turn_meta=turn_meta, route=route)
    if answer.strip():
        mem_add_bot(sid, answer)
    yield f"event: ui\ndata: {_json.dumps(final, ensure_ascii=False, default=_sse_default)}\n\n"
    yield "event: done\ndata: {}\n\n"


def _sse_default(obj):
    """JSON default для SSE — обрабатывает numpy типы из retrieval."""
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
