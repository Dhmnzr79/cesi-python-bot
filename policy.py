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
    pre_doc_turn_count: int | None = None,
) -> dict:
    meta = payload.get("meta") or {}
    low_score = bool(meta.get("low_score"))
    lead_flow_active = is_active_lead_flow(session_state)
    booking = booking_intent(q)
    exhausted = _is_topic_exhausted(doc_meta, topic_state)
    doc_turn_after = int(topic_state.get("doc_turn_count") or 0)
    doc_turn_before = (
        int(pre_doc_turn_count)
        if pre_doc_turn_count is not None
        else max(doc_turn_after - 1, 0)
    )

    covered_h3 = {str(x).strip().lower() for x in (topic_state.get("covered_h3_ids") or []) if x}
    followups_all = []
    for f in list(meta.get("followups") or []):
        if not isinstance(f, dict):
            continue
        ref = str(f.get("ref") or "")
        anchor = ref.split("#", 1)[1].strip().lower() if "#" in ref else ""
        if anchor and anchor in covered_h3:
            continue
        followups_all.append(f)

    refs_all = list(payload.get("quick_replies") or [])
    deferred = list(topic_state.get("refs_deferred") or [])
    if deferred:
        refs_all = deferred + refs_all

    has_video = bool(doc_meta.get("video_key"))
    video_shown = bool(topic_state.get("video_shown"))
    situation_allowed = bool(doc_meta.get("situation_allowed"))
    situation_offered = bool(topic_state.get("situation_offered"))
    ref_used = bool(topic_state.get("suggest_ref_used"))
    cta_from_turn = max(0, int(doc_meta.get("cta_from_turn", 0) or 0))

    is_first_content_turn = doc_turn_before == 0
    max_follow_slots = 1 if has_video else 2

    followups_out: list = []
    show_video = False
    show_situation = False
    show_ref = False

    slots = max_follow_slots
    fu_queue = list(followups_all)

    if (
        not low_score
        and has_video
        and not video_shown
        and is_first_content_turn
        and fu_queue
    ):
        show_video = True
        followups_out.append(fu_queue.pop(0))
        slots = 0
    elif (
        not low_score
        and has_video
        and not video_shown
        and is_first_content_turn
        and not fu_queue
    ):
        show_video = True
        slots = max_follow_slots - 1
        if (
            slots > 0
            and situation_allowed
            and not situation_offered
            and not lead_flow_active
            and not bool(session_state.get("situation_pending"))
        ):
            show_situation = True
            slots -= 1
        if slots > 0 and refs_all and not ref_used and exhausted:
            show_ref = True
            slots -= 1
        slots = 0
    else:
        while slots > 0 and fu_queue:
            followups_out.append(fu_queue.pop(0))
            slots -= 1
        if slots > 0 and not low_score and has_video and not video_shown:
            show_video = True
            slots -= 1

        situation_blocked = (
            is_first_content_turn
            and has_video
            and not video_shown
            and bool(followups_all)
        )
        if (
            slots > 0
            and not low_score
            and situation_allowed
            and not situation_offered
            and not lead_flow_active
            and not bool(session_state.get("situation_pending"))
            and not situation_blocked
        ):
            show_situation = True
            slots -= 1

        if (
            slots > 0
            and not low_score
            and refs_all
            and not ref_used
            and exhausted
        ):
            show_ref = True
            slots -= 1

    refs_out = refs_all[:1] if show_ref else []

    cta = payload.get("cta")
    show_cta = bool(cta) and not lead_flow_active and not bool(
        session_state.get("situation_pending")
    )
    if show_cta and doc_turn_before < cta_from_turn:
        show_cta = False
    if show_cta and booking:
        show_cta = False

    defer_refs = bool(refs_all) and not show_ref and (show_video or show_situation)
    dropped = []
    if not show_ref and refs_all:
        dropped.append("suggest_refs")
    if payload.get("cta") and not show_cta:
        dropped.append("cta")

    return {
        "low_score": low_score,
        "topic_exhausted": exhausted,
        "lead_flow_active": lead_flow_active,
        "booking": booking,
        "show_cta": show_cta,
        "show_video": show_video,
        "show_situation": show_situation,
        "show_refs": show_ref,
        "followups": followups_out,
        "refs": refs_out,
        "defer_refs": defer_refs,
        "dropped": dropped,
        "doc_turn_before": doc_turn_before,
        "doc_turn_after": doc_turn_after,
        "cta_from_turn": cta_from_turn,
    }


def apply_response_policy(
    payload: dict,
    session_state: dict,
    q: str,
    *,
    topic_state: dict | None = None,
    doc_meta: dict | None = None,
    pre_doc_turn_count: int | None = None,
) -> dict:
    topic_state = topic_state or {}
    doc_meta = doc_meta or {}
    decision = build_policy_decision(
        payload=payload,
        session_state=session_state,
        topic_state=topic_state,
        doc_meta=doc_meta,
        q=q,
        pre_doc_turn_count=pre_doc_turn_count,
    )

    payload["quick_replies"] = decision["refs"] if decision["show_refs"] else []
    payload["cta"] = payload.get("cta") if decision["show_cta"] else None
    payload["video"] = (
        {"key": doc_meta.get("video_key")}
        if decision["show_video"] and doc_meta.get("video_key")
        else None
    )

    sit_show = bool(decision["show_situation"])
    payload["situation"] = {"show": sit_show, "mode": "normal"}

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
        "booking": bool(decision["booking"]),
        "refs_candidate_count": len(decision["refs"]),
        "dropped": decision["dropped"],
        "doc_turn_before": decision["doc_turn_before"],
        "doc_turn_after": decision["doc_turn_after"],
        "cta_from_turn": decision["cta_from_turn"],
    }
    return payload
