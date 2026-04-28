"""Flow orchestration for non-retrieval branches in /ask."""

import os

from lead_service import handle_lead
from llm import classify_lead_name_shape
from name_gate import hard_reject_lead_name
from policy import booking_intent
from session import (
    extract_name,
    extract_phone,
    get_lead_pending_name,
    is_active_lead_flow,
    mark_booking_intent_ever,
    mem_get,
    parse_no,
    parse_yes,
    set_lead_intent,
    set_lead_pending_name,
    set_situation_note,
    set_situation_pending,
    update_profile,
)

_LEAD_NAME_CONFIRM_YES = "lead:name_confirm:yes"
_LEAD_NAME_CONFIRM_NO = "lead:name_confirm:no"


def _name_confirm_quick_replies() -> list[dict]:
    return [
        {"label": "Да", "ref": _LEAD_NAME_CONFIRM_YES},
        {"label": "Нет, введу по-другому", "ref": _LEAD_NAME_CONFIRM_NO},
    ]


def _collecting_name_reply(
    sid: str,
    q: str,
    client_id: str | None,
    *,
    txt: dict,
    service_payload,
) -> dict | None:
    if hard_reject_lead_name(q):
        return service_payload(
            txt["lead_name_hard"],
            sid,
            client_id,
            lead_flow=True,
            lead_step="name",
        )
    name = extract_name(q)
    if not name:
        return service_payload(
            txt["lead_name_retry"],
            sid,
            client_id,
            lead_flow=True,
            lead_step="name",
        )
    label = classify_lead_name_shape(name, q, client_id=client_id, sid=sid)
    if label == "invalid_name":
        return service_payload(
            txt["lead_name_invalid"],
            sid,
            client_id,
            lead_flow=True,
            lead_step="name",
        )
    if label == "unsure":
        set_lead_pending_name(sid, name)
        set_lead_intent(sid, "confirming_name")
        return service_payload(
            txt["lead_name_confirm_tpl"].format(name=name),
            sid,
            client_id,
            lead_flow=True,
            lead_step="confirm_name",
            quick_replies=_name_confirm_quick_replies(),
        )
    update_profile(sid, name=name)
    set_lead_intent(sid, "collecting_phone")
    return service_payload(
        txt["lead_phone_prompt_tpl"].format(name=name),
        sid,
        client_id,
        lead_flow=True,
        lead_step="phone",
    )


def _handle_lead_name_confirm(
    *,
    data: dict,
    sid: str,
    q: str,
    client_id: str | None,
    txt: dict,
    service_payload,
) -> dict | None:
    ref = (data.get("ref") or "").strip()
    pending = get_lead_pending_name(sid)
    yes = ref == _LEAD_NAME_CONFIRM_YES or parse_yes(q)
    no = ref == _LEAD_NAME_CONFIRM_NO or parse_no(q)

    if yes and pending:
        update_profile(sid, name=pending)
        set_lead_pending_name(sid, None)
        set_lead_intent(sid, "collecting_phone")
        return {
            "payload": service_payload(
                txt["lead_phone_prompt_tpl"].format(name=pending),
                sid,
                client_id,
                lead_flow=True,
                lead_step="phone",
            ),
            "doc_id": None,
        }

    if no:
        set_lead_pending_name(sid, None)
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_reenter"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="name",
            ),
            "doc_id": None,
        }

    if q.strip() and len(q.strip()) > 1 and not yes:
        set_lead_pending_name(sid, None)
        set_lead_intent(sid, "collecting_name")
        payload = _collecting_name_reply(
            sid, q, client_id, txt=txt, service_payload=service_payload
        )
        if payload is not None:
            return {"payload": payload, "doc_id": None}

    if pending:
        return {
            "payload": service_payload(
                txt["lead_name_confirm_tpl"].format(name=pending),
                sid,
                client_id,
                lead_flow=True,
                lead_step="confirm_name",
                quick_replies=_name_confirm_quick_replies(),
            ),
            "doc_id": None,
        }

    set_lead_intent(sid, "collecting_name")
    return {
        "payload": service_payload(
            txt["lead_name_prompt"],
            sid,
            client_id,
            lead_flow=True,
            lead_step="name",
        ),
        "doc_id": None,
    }


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
        return _collecting_name_reply(sid, q, client_id, txt=txt, service_payload=service_payload)

    if intent == "collecting_phone":
        phone = extract_phone(q)
        if not phone:
            return service_payload(
                txt["lead_phone_retry"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="phone",
            )
        update_profile(sid, phone=phone)
        st2 = mem_get(sid)
        prof = st2.get("profile") or {}
        lead_payload, lead_status = handle_lead(
            {
                "name": (prof.get("name") or "").strip(),
                "phone": (prof.get("phone") or "").strip(),
                "intent": "lead",
                "sid": sid,
                "client_id": client_id,
                "situation_note": (st2.get("situation_note") or "").strip(),
            }
        )
        if lead_status != 200:
            return service_payload(
                txt["lead_submit_error"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="phone",
                lead_error=lead_payload.get("error_code") or lead_payload.get("error"),
            )
        set_lead_intent(sid, "submitted")
        set_situation_pending(sid, False)
        set_situation_note(sid, "")
        return service_payload(
            txt["lead_submit_ok"],
            sid,
            client_id,
            lead_flow=True,
            lead_step="done",
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

    if st.get("lead_intent") == "confirming_name":
        return _handle_lead_name_confirm(
            data=data,
            sid=sid,
            q=q,
            client_id=client_id,
            txt=txt,
            service_payload=service_payload,
        )

    if q and booking_intent(q, sid=sid, client_id=client_id) and not is_active_lead_flow(st):
        mark_booking_intent_ever(sid)
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="name",
                booking_intent_flag=True,
            ),
            "doc_id": None,
        }

    if (
        st.get("last_bot_action") == "offered_subtopic"
        and q
        and len(q.strip().split()) <= 3
        and parse_yes(q)
        and st.get("lead_intent") != "confirming_name"
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
                lead_step="name",
            ),
            "doc_id": None,
        }

    if data.get("cta_action") == "lead":
        mark_booking_intent_ever(sid)
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="name",
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
        mark_booking_intent_ever(sid)
        set_lead_intent(sid, "collecting_name")
        return {
            "payload": service_payload(
                txt["lead_name_prompt"],
                sid,
                client_id,
                lead_flow=True,
                lead_step="name",
            ),
            "doc_id": None,
        }

    return None
