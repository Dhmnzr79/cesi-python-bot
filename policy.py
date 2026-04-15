"""Детерминированные правила до/после LLM (без вызова модели)."""
from config import BOOKING_INTENT_RE, CONTACTS_RE, PRICES_RE
from retriever import chunk_doc_type
from session import is_active_lead_flow


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
        file_name = (ch.get("file") or "").strip().lower() if isinstance(ch, dict) else ""
        if dt in {"prices", "pricing"} or "__pricing__" in file_name:
            return ch
    return None


def _is_topic_exhausted(doc_meta: dict, topic_state: dict) -> bool:
    suggest_h3 = list(doc_meta.get("suggest_h3") or [])
    covered = set(topic_state.get("covered_h3_ids") or [])
    if not suggest_h3:
        return int(topic_state.get("doc_turn_count") or 0) >= 1
    return covered.issuperset(set(suggest_h3))


def build_policy_decision(
    *,
    payload: dict,
    session_state: dict,
    topic_state: dict,
    doc_meta: dict,
    q: str,
) -> dict:
    meta = payload.get("meta") or {}
    low_score = bool(meta.get("low_score"))
    lead_flow_active = is_active_lead_flow(session_state)
    booking = booking_intent(q)
    exhausted = _is_topic_exhausted(doc_meta, topic_state)
    doc_turn = int(topic_state.get("doc_turn_count") or 0)

    covered_h3 = {str(x).strip().lower() for x in (topic_state.get("covered_h3_ids") or []) if x}
    followups = []
    for f in list(meta.get("followups") or []):
        if not isinstance(f, dict):
            continue
        ref = str(f.get("ref") or "")
        anchor = ref.split("#", 1)[1].strip().lower() if "#" in ref else ""
        if anchor and anchor in covered_h3:
            continue
        followups.append(f)
    followups = followups[:2]
    refs = list(payload.get("quick_replies") or [])
    deferred = list(topic_state.get("refs_deferred") or [])
    if deferred:
        refs = deferred + refs
    refs = refs[:1]

    has_video = bool(doc_meta.get("video_key"))
    video_shown = bool(topic_state.get("video_shown"))
    video_pending = bool(topic_state.get("video_pending")) or (has_video and not video_shown)
    situation_allowed = bool(doc_meta.get("situation_allowed"))
    situation_offered = bool(topic_state.get("situation_offered"))
    can_offer_situation = (
        situation_allowed
        and doc_turn >= 3
        and not situation_offered
        and not lead_flow_active
        and len(followups) <= 1
    )
    can_offer_video = (
        video_pending
        and not lead_flow_active
        and not bool(session_state.get("situation_pending"))
        and len(followups) <= 1
    )

    show_video = False
    show_situation = False
    suggest_h3 = list(doc_meta.get("suggest_h3") or [])
    is_one_screen_topic = not suggest_h3
    show_refs = bool(refs) and (bool(exhausted) or is_one_screen_topic)

    if not low_score:
        if not video_shown:
            if can_offer_video:
                show_video = True
                show_refs = False
            elif can_offer_situation:
                show_situation = True
                show_refs = False
        else:
            if can_offer_situation:
                show_situation = True
                show_refs = False

    cta = payload.get("cta")
    show_cta = bool(cta) and not lead_flow_active
    if show_cta and not (booking or exhausted or doc_turn >= 2):
        show_cta = False

    defer_refs = bool(refs) and not show_refs and (show_video or show_situation)
    dropped = []
    if not show_refs and refs:
        dropped.append("suggest_refs")
    if payload.get("cta") and not show_cta:
        dropped.append("cta")

    return {
        "low_score": low_score,
        "topic_exhausted": exhausted,
        "lead_flow_active": lead_flow_active,
        "show_cta": show_cta,
        "show_video": show_video,
        "show_situation": show_situation,
        "show_refs": show_refs,
        "followups": followups,
        "refs": refs,
        "defer_refs": defer_refs,
        "dropped": dropped,
    }


def apply_response_policy(
    payload: dict,
    session_state: dict,
    q: str,
    *,
    topic_state: dict | None = None,
    doc_meta: dict | None = None,
) -> dict:
    topic_state = topic_state or {}
    doc_meta = doc_meta or {}
    decision = build_policy_decision(
        payload=payload,
        session_state=session_state,
        topic_state=topic_state,
        doc_meta=doc_meta,
        q=q,
    )

    payload["quick_replies"] = decision["refs"] if decision["show_refs"] else []
    payload["cta"] = payload.get("cta") if decision["show_cta"] else None
    payload["video"] = (
        {"key": doc_meta.get("video_key")} if decision["show_video"] and doc_meta.get("video_key") else None
    )
    payload["situation"] = {"show": bool(decision["show_situation"])}

    meta = payload.setdefault("meta", {})
    meta["followups"] = decision["followups"]
    meta["topic_exhausted"] = bool(decision["topic_exhausted"])
    meta["policy_decision"] = {
        "show_video": bool(decision["show_video"]),
        "show_situation": bool(decision["show_situation"]),
        "show_refs": bool(decision["show_refs"]),
        "show_cta": bool(decision["show_cta"]),
        "defer_refs": bool(decision["defer_refs"]),
        "refs_to_defer": (decision["refs"] if decision["defer_refs"] else []),
        "lead_flow_active": bool(decision["lead_flow_active"]),
        "refs_candidate_count": len(decision["refs"]),
        "dropped": decision["dropped"],
    }
    return payload
