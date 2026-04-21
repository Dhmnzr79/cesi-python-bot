"""Flow orchestration for non-retrieval branches in /ask."""

import os

from lead_service import handle_lead
from policy import booking_intent
from session import (
    extract_name,
    extract_phone,
    is_active_lead_flow,
    parse_yes,
    set_lead_intent,
    set_situation_note,
    set_situation_pending,
    update_profile,
    mem_get,
)


def _lead_flow_payload(
    sid: str,
    q: str,
    client_id: str | None,
    *,
    txt: dict,
    service_payload,
) -> dict | None:
    st = mem_get(sid)
    intent = (st.get("lead_intent") or "none").strip()

    if intent == "collecting_name":
        name = extract_name(q)
        if not name:
            return service_payload(
                txt["lead_name_retry"],
                sid,
                client_id,
                lead_flow=True,
            )
        update_profile(sid, name=name)
        set_lead_intent(sid, "collecting_phone")
        return service_payload(
            txt["lead_phone_prompt_tpl"].format(name=name),
            sid,
            client_id,
            lead_flow=True,
        )

    if intent == "collecting_phone":
        phone = extract_phone(q)
        if not phone:
            return service_payload(
                txt["lead_phone_retry"],
                sid,
                client_id,
                lead_flow=True,
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
            return service_payload(
                txt["lead_submit_error"],
                sid,
                client_id,
                lead_flow=True,
                lead_error=lead_payload.get("error"),
            )
        set_lead_intent(sid, "submitted")
        set_situation_pending(sid, False)
        set_situation_note(sid, "")
        return service_payload(
            txt["lead_submit_ok"],
            sid,
            client_id,
            lead_flow=True,
        )
    return None


def handle_flows(
    *,
    data: dict,
    st: dict,
    sid: str,
    q: str,
    client_id: str | None,
    txt: dict,
    service_payload,
    get_last_content_ui_payload,
    get_topic_state,
) -> dict | None:
    """Return {'payload': dict, 'doc_id': str|None} when flow handled.

    May also return {'redirect_ref': str} for followup redirect.
    """
    if data.get("situation_action") == "back":
        set_situation_pending(sid, False)
        snap = get_last_content_ui_payload(sid)
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
                (restored.get("meta") or {}).get("file")
                and os.path.splitext(
                    os.path.basename((restored.get("meta") or {}).get("file") or "")
                )[0]
            )
            if doc_id_back and get_topic_state(sid, doc_id_back).get("situation_offered"):
                restored["situation"] = {"show": False, "mode": "normal"}
            meta_r = restored.setdefault("meta", {})
            meta_r["situation_back"] = True
            meta_r.setdefault("sid", sid)
            meta_r.setdefault("client_id", client_id)
            return {"payload": restored, "doc_id": st.get("current_doc_id")}
        return {
            "payload": service_payload(
                txt["situation_back_fallback"],
                sid,
                client_id,
                situation_back=True,
            ),
            "doc_id": st.get("current_doc_id"),
        }

    if q and booking_intent(q) and not is_active_lead_flow(st):
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
                booking_intent_flag=True,
            ),
            "doc_id": None,
        }

    # Practical rule:
    # - "yes" can auto-redirect only when a single subtopic button is present
    # - with 2+ buttons, ask user to choose explicitly and repeat options
    if (
        st.get("last_bot_action") == "offered_subtopic"
        and q
        and len(q.strip().split()) <= 3
        and parse_yes(q)
    ):
        buttons = [b for b in (st.get("last_presented_buttons") or []) if b.get("ref")]
        if len(buttons) == 1:
            return {"payload": None, "doc_id": None, "redirect_ref": buttons[0]["ref"]}
        if len(buttons) >= 2:
            quick_replies = [
                {"label": (b.get("label") or "").strip(), "ref": b.get("ref")}
                for b in buttons[:2]
                if (b.get("label") or "").strip() and b.get("ref")
            ]
            return {
                "payload": service_payload(
                    txt["followup_choose_topic"],
                    sid,
                    client_id,
                    quick_replies=quick_replies,
                ),
                "doc_id": st.get("current_doc_id"),
            }

    if is_active_lead_flow(st):
        payload = _lead_flow_payload(
            sid,
            q,
            client_id,
            txt=txt,
            service_payload=service_payload,
        )
        if payload is not None:
            return {"payload": payload, "doc_id": None}

    if st.get("situation_pending"):
        if not q or len(q.strip()) < 3:
            return {
                "payload": service_payload(
                    txt["situation_retry_short"],
                    sid,
                    client_id,
                    situation_mode="pending",
                    situation_collect=True,
                ),
                "doc_id": None,
            }
        set_situation_note(sid, q)
        set_situation_pending(sid, False)
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["situation_to_lead_name"],
                sid,
                client_id,
                lead_flow=True,
            ),
            "doc_id": None,
        }

    if data.get("cta_action") == "lead":
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
            ),
            "doc_id": None,
        }

    if data.get("situation_action") == "start" or data.get("action") == "situation":
        set_situation_pending(sid, True)
        return {
            "payload": service_payload(
                txt["situation_prompt"],
                sid,
                client_id,
                situation_mode="pending",
                situation_collect=True,
            ),
            "doc_id": None,
        }

    if st.get("last_bot_action") == "offered_situation" and parse_yes(q):
        set_situation_pending(sid, True)
        return {
            "payload": service_payload(
                txt["situation_prompt"],
                sid,
                client_id,
                situation_mode="pending",
                situation_collect=True,
            ),
            "doc_id": None,
        }

    if st.get("last_bot_action") == "offered_cta" and parse_yes(q):
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
            ),
            "doc_id": None,
        }

    return None
