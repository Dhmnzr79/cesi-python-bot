from __future__ import annotations

import json
import os
import threading
from typing import Any, Literal

from pydantic import ValidationError

from contracts.decision_frame import DecisionFrame
from llm import client
from logging_setup import get_logger, log_json, log_llm_error, log_llm_usage, emit_bot_event


logger = get_logger("bot")

_MODEL = (os.getenv("MODEL_RESOLVER") or "").strip() or "gpt-5.4-nano"
_ON = (os.getenv("V5_RESOLVER_SHADOW_ON") or "1").strip().lower() in ("1", "true", "yes")
_TIMEOUT_SEC = float(os.getenv("V5_RESOLVER_TIMEOUT_SEC", "8"))

_QueryModeEval = Literal["v5_resolver", "v5_resolver_shadow"]


RESOLVER_SYSTEM_PROMPT = (
    "Ты — Resolver слоя v5. Твоя задача: классифицировать запрос и вернуть DecisionFrame.\n"
    "Верни только JSON (без markdown) со СТРОГО этими ключами:\n"
    "route_intent, service_topic, service_id, query_mode, confidence, needs_clarification\n"
    "\n"
    "ВАЖНО: route_intent и query_mode — разные поля.\n"
    "- route_intent: content | price_lookup | price_concern | unknown\n"
    "- query_mode:  overview | specific | comparison | process\n"
    "Никогда не пиши 'comparison' или 'process' в route_intent — это ТОЛЬКО query_mode.\n"
    "\n"
    "Определения query_mode:\n"
    "- overview: общий обзор услуги/темы/объекта (\"Расскажите про имплантацию\", \"Какие у вас врачи\", \"Что такое All-on-4\").\n"
    "- specific: конкретный аспект услуги/клиники (боль, длительность, гарантия, материалы, бренды,\n"
    "  противопоказания, методы оплаты, наличие услуги/специалиста). Сюда же:\n"
    "  \"есть ли у вас X\", \"какие X\", \"какая X\", \"можно ли оплатить картой\", \"больно ли\".\n"
    "  Eligibility под условия пациента (диабет, возраст, состояние) — это тоже specific.\n"
    "- comparison: сравнение двух или более вариантов (\"X или Y\", \"что лучше\").\n"
    "- process: как проходит лечение, этапы.\n"
    "\n"
    "Few-shot examples (формат JSON, это примеры классификации):\n"
    "1) {\"Q\": \"Можно ли оплатить картой?\", \"query_mode\": \"specific\"}\n"
    "2) {\"Q\": \"Можно ли мне с диабетом?\", \"query_mode\": \"specific\"}\n"
    "3) {\"Q\": \"Больно ли ставить имплант?\", \"query_mode\": \"specific\"}\n"
    "4) {\"Q\": \"Подходит ли мне имплантация?\", \"query_mode\": \"specific\"}\n"
    "5) {\"Q\": \"Есть ли у вас имплантолог?\", \"query_mode\": \"specific\"}\n"
    "6) {\"Q\": \"Какие импланты вы ставите?\", \"query_mode\": \"specific\"}\n"
    "7) {\"Q\": \"Расскажите про доктора Иванова\", \"query_mode\": \"overview\"}\n"
    "8) {\"Q\": \"Какая гарантия?\", \"query_mode\": \"specific\"}\n"
    "\n"
    "service_topic: implantation | prosthetics | clinic | doctors | unknown\n"
    "service_id: строка или null (если не уверен).\n"
    "confidence: объект с числами 0..1: {intent, topic, service, query_mode}.\n"
    "needs_clarification: true/false.\n"
    "\n"
    "Пример формата (не копируй буквально):\n"
    "{\n"
    "  \"route_intent\": \"content\",\n"
    "  \"service_topic\": \"implantation\",\n"
    "  \"service_id\": null,\n"
    "  \"query_mode\": \"process\",\n"
    "  \"confidence\": {\"intent\": 0.8, \"topic\": 0.7, \"service\": 0.0, \"query_mode\": 0.6},\n"
    "  \"needs_clarification\": false\n"
    "}\n"
    "\n"
    "Если не уверен — route_intent=unknown, service_topic=unknown, needs_clarification=true и confidence=0.\n"
)


def _resolver_user_content(*, question: str, history: list[dict[str, Any]] | None) -> str:
    q = (question or "").strip()
    hist = list(history or [])[-6:]
    hist_text = "\n".join(
        f"{str(m.get('role') or '')}: {str(m.get('content') or '')}".strip()
        for m in hist
        if isinstance(m, dict)
    ).strip()
    user = f"Вопрос:\n{q}\n"
    if hist_text:
        user += f"\nКонтекст (последние сообщения):\n{hist_text}\n"
    return user


def _fallback_unknown() -> DecisionFrame:
    return DecisionFrame.model_validate(
        {
            "route_intent": "unknown",
            "service_topic": "unknown",
            "service_id": None,
            "query_mode": "specific",
            "confidence": {"intent": 0.0, "topic": 0.0, "service": 0.0, "query_mode": 0.0},
            "needs_clarification": True,
        }
    )


def _call_resolver_llm(
    *,
    question: str,
    history: list[dict[str, Any]] | None,
    log_call_type: _QueryModeEval,
) -> DecisionFrame:
    """Shared OpenAI JSON path; single system prompt (PR #1.2.6)."""
    user = _resolver_user_content(question=question, history=history)
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            max_completion_tokens=250,
            response_format={"type": "json_object"},
            timeout=_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        log_llm_usage(logger, resp, call_type=log_call_type, model=_MODEL)
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        return DecisionFrame.model_validate(obj)
    except ValidationError as e:
        try:
            logger.warning(
                "resolver_validation_failed",
                extra={
                    "extra_data": {
                        "call_type": log_call_type,
                        "model": _MODEL,
                        "raw_output": (raw or "")[:2000],
                        "error": str(e)[:2000],
                    }
                },
            )
        except Exception:
            pass
        return _fallback_unknown()
    except Exception as e:
        log_llm_error(logger, call_type=log_call_type, err=str(e), model=_MODEL)
        return _fallback_unknown()


def resolve_decision_frame_shadow(*, question: str, history: list[dict[str, Any]] | None) -> DecisionFrame:
    """Compute DecisionFrame for logs only (shadow); same prompt + model path as routing Resolver."""
    return _call_resolver_llm(question=question, history=history, log_call_type="v5_resolver_shadow")


def resolve_decision_frame(*, question: str, history: list[dict[str, Any]] | None) -> DecisionFrame:
    """Compute DecisionFrame for routing (/ask pipeline)."""
    return _call_resolver_llm(question=question, history=history, log_call_type="v5_resolver")


def map_classify_intent_to_route_intent(old_intent: str) -> str:
    """Map legacy classify_intent label to DecisionFrame.route_intent (PR #1.2 safety-net)."""
    x = (old_intent or "").strip().lower()
    if x == "price_lookup":
        return "price_lookup"
    if x == "price_concern":
        return "price_concern"
    if x == "content":
        return "content"
    if x == "contacts":
        logger.warning(
            "resolver_safety_net_contacts_unexpected",
            extra={
                "extra_data": {"old_intent": old_intent, "hint": "should be handled by gates (A1), not Resolver"},
            },
        )
        return "content"
    if x == "offtopic":
        return "unknown"
    return "unknown"


def resolve_with_fallback(
    *,
    question: str,
    history: list[dict[str, Any]] | None,
    client_id: str,
    sid: str,
    session_state: dict[str, Any] | None = None,
) -> tuple[DecisionFrame, list[str]]:
    """
    Run Resolver LLM decision, apply safety-net against THRESHOLDS via classify_intent fallback.
    session_state reserved for future (context); not used in PR #1.2.

    Returns (DecisionFrame, safety_net_used) where safety_net_used is a subset of
    ["intent", "topic", "query_mode"].
    """
    _ = session_state  # intentional no-op until multi-client/context wiring
    from core.routing_loader import THRESHOLDS

    decision = resolve_decision_frame(question=question, history=history)
    flags: list[str] = []

    thresh = THRESHOLDS.resolver.min_confidence
    ci = float(decision.confidence.intent or 0.0)
    if ci < float(thresh.intent):
        from llm import classify_intent

        old_intent = classify_intent(question, client_id=client_id, sid=sid)
        decision.route_intent = map_classify_intent_to_route_intent(old_intent)
        flags.append("intent")
        log_json(
            logger,
            "safety_net_intent_used",
            sid=sid,
            client_id=client_id,
            old=old_intent,
            conf=round(ci, 4),
            route_intent=decision.route_intent,
        )
    else:
        log_json(logger, "resolver_used_intent", sid=sid, client_id=client_id, conf=round(ci, 4))

    ct = float(decision.confidence.topic or 0.0)
    if ct < float(thresh.topic):
        decision.service_topic = "unknown"
        flags.append("topic")
        log_json(logger, "safety_net_topic_used", sid=sid, client_id=client_id, conf=round(ct, 4))
    else:
        log_json(logger, "resolver_used_topic", sid=sid, client_id=client_id, conf=round(ct, 4))

    cm = float(decision.confidence.query_mode or 0.0)
    if cm < float(thresh.query_mode):
        decision.query_mode = "specific"
        flags.append("query_mode")
        log_json(logger, "safety_net_query_mode_used", sid=sid, client_id=client_id, conf=round(cm, 4))

    return decision, flags


def maybe_start_shadow_resolver(*, question: str, sid: str, client_id: str) -> None:
    """Fire-and-forget shadow resolver. Never blocks the request."""
    if not _ON:
        return
    q = (question or "").strip()
    if not q:
        return

    def _run():
        try:
            # Import here to avoid any import-time coupling in Phase 0/early startup.
            from session import mem_get

            st = mem_get(sid)
            hist = list((st or {}).get("hist") or [])
        except Exception:
            hist = []
        df = resolve_decision_frame_shadow(question=q, history=hist)
        emit_bot_event(
            logger,
            "v5_decision_frame_shadow",
            status="ok",
            sid=sid,
            client_id=client_id,
            details={"decision_frame": df.model_dump()},
        )

    threading.Thread(target=_run, name="v5-resolver-shadow", daemon=True).start()
