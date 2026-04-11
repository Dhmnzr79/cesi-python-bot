"""Детерминированные правила до/после LLM (без вызова модели)."""
from config import BOOKING_INTENT_RE, CONTACTS_RE, PRICES_RE
from retriever import chunk_doc_type


def contacts_intent(q: str) -> bool:
    return bool(CONTACTS_RE.search(q or ""))


def price_intent(q: str) -> bool:
    return bool(PRICES_RE.search(q or ""))


def booking_intent(q: str) -> bool:
    return bool(BOOKING_INTENT_RE.search(q or ""))


def pick_contacts_chunk(cands: list) -> dict | None:
    for ch in cands:
        dt = (chunk_doc_type(ch) or "").strip().lower()
        if dt == "contacts":
            return ch
    return None


def pick_prices_chunk(cands: list) -> dict | None:
    for ch in cands:
        dt = (chunk_doc_type(ch) or "").strip().lower()
        if dt == "prices":
            return ch
    return None


def apply_response_policy(payload: dict, session_state: dict, q: str) -> dict:
    """
    Порядок (упрощённо под этап 2):
    1) lead_intent — воронка; пока не подключена, поле зарезервировано.
    2) turn_count < 2 — убрать CTA, кроме явного намерения записаться.
    Остальные 8 правил — позже (ситуация, handoff, смена темы).
    """
    if (payload.get("meta") or {}).get("low_score"):
        return payload

    if session_state.get("lead_intent"):
        return payload

    turn = int(session_state.get("turn_count") or 0)
    if turn < 2 and not booking_intent(q) and payload.get("cta"):
        payload["cta"] = None

    return payload
