"""Сборка JSON ответа /ask (одна точка сборки)."""
import os
import re

from config import default_cta_dict
from llm import generate_facts_card_answer
from meta_loader import get_doc_path


def get_chunk_ids(chunk: dict) -> tuple:
    if not isinstance(chunk, dict):
        return (None, None)
    return (chunk.get("h2_id"), chunk.get("h3_id"))


def is_overview_by_ids(h2_id, h3_id) -> bool:
    h2 = (h2_id or "").strip().lower()
    h3 = (h3_id or "").strip().lower()
    return (not h2 and not h3) or (h2 in {"overview", "korotko"}) or (h3 in {"overview", "korotko"})


def heading_label(md_file: str, sect_id: str, client_id: str | None = None) -> str:
    if not md_file or not sect_id:
        return (sect_id or "").replace("-", " ").capitalize()
    try:
        path = get_doc_path(os.path.basename(md_file), client_id=client_id) or md_file
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        rx3 = re.compile(
            rf"^###\s+(.*?)\s*\{{#{re.escape(sect_id)}\}}\s*$", re.M | re.I
        )
        rx2 = re.compile(
            rf"^##\s+(.*?)\s*\{{#{re.escape(sect_id)}\}}\s*$", re.M | re.I
        )
        m = rx3.search(txt) or rx2.search(txt)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return (sect_id or "").replace("-", " ").capitalize()


def build_quick_refs(meta: dict, md_file: str, current_h2_id: str, current_h3_id: str) -> list:
    out = []
    cur_anchor = current_h3_id or current_h2_id or "overview"
    cur_ref = (
        f"{os.path.basename(md_file or '')}#{cur_anchor}".lower() if md_file else None
    )
    for r in meta.get("suggest_refs") or []:
        if isinstance(r, str):
            ref = r if "#" in r else None
            label = r.split("#", 1)[0] if ref else None
        else:
            ref = r.get("ref")
            label = r.get("label") or (ref.split("#", 1)[0] if ref else None)
        if not (label and ref):
            continue
        if cur_ref and ref.lower() == cur_ref:
            continue
        out.append({"label": label, "ref": ref})
    return out


def build_followups(
    meta: dict,
    md_file: str,
    current_h2_id: str,
    current_h3_id: str,
    covered_h3_ids: list[str] | None = None,
    client_id: str | None = None,
) -> list:
    out = []
    covered = {str(x).strip().lower() for x in (covered_h3_ids or []) if x}
    for s in meta.get("suggest_h3") or []:
        h_id = s if isinstance(s, str) else (s.get("h3_id") or s.get("id"))
        if not h_id:
            continue
        if str(h_id).lower() in covered:
            continue
        if str(h_id).lower() in {
            str(current_h2_id or "").lower(),
            str(current_h3_id or "").lower(),
        }:
            continue
        label = heading_label(md_file, h_id, client_id=client_id)
        out.append({"label": label, "ref": f"{os.path.basename(md_file)}#{h_id}"})
    return out


def build_cta(meta: dict):
    if meta.get("cta_text") and meta.get("cta_action"):
        return {"text": meta["cta_text"], "action": meta["cta_action"]}
    return None


def pick_relevant_offer(meta: dict):
    return None


def dedup_refs_vs_cta(quick_refs: list, cta_btn: dict | None) -> list:
    if not cta_btn or not quick_refs:
        return quick_refs
    cta_label = (cta_btn.get("text") or "").strip().lower()
    out = []
    seen = set()
    for r in quick_refs:
        lbl = (r.get("label") or "").strip().lower()
        if not lbl:
            continue
        if lbl == cta_label:
            continue
        if lbl not in seen:
            out.append(r)
            seen.add(lbl)
    return out


def meta_tags(meta: dict):
    t = meta.get("tags")
    if isinstance(t, set):
        return list(t)
    return t or []


def build_ask_response(
    *,
    answer: str,
    top: dict,
    meta: dict,
    sid: str,
    profile: dict,
    client_id: str | None = None,
    topic_state: dict | None = None,
) -> dict:
    """Единая структура успешного ответа /ask."""
    md_file = top.get("file")
    h2_id, h3_id = get_chunk_ids(top)
    h2_val = top.get("h2") or top.get("h2_id")
    h3_val = top.get("h3") or top.get("h3_id")
    is_overview = is_overview_by_ids(h2_id, h3_id)

    quick_refs = build_quick_refs(meta, md_file, h2_id, h3_id)
    fups_full = build_followups(
        meta,
        md_file,
        h2_id,
        h3_id,
        covered_h3_ids=(topic_state or {}).get("covered_h3_ids") or [],
        client_id=client_id,
    )
    followups = fups_full

    cta_btn = build_cta(meta)
    quick_refs = dedup_refs_vs_cta(quick_refs, cta_btn)

    score = float(round(float(top.get("_score", 0.0)), 3))

    meta_out = {
        "file": md_file,
        "h2": h2_val,
        "h3": h3_val,
        "h2_id": h2_id,
        "h3_id": h3_id,
        "score": score,
        "followups": followups,
        "is_overview": bool(is_overview),
        "cta_mode": meta.get("cta_mode"),
        "tags": meta_tags(meta),
        "sid": sid,
        "facts": {
            "name": profile.get("name"),
            "phone": profile.get("phone"),
        },
    }
    if client_id is not None:
        meta_out["client_id"] = client_id

    return {
        "answer": answer,
        "quick_replies": quick_refs,
        "cta": cta_btn,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": pick_relevant_offer(meta),
        "meta": meta_out,
    }


def normalize_policy_payload(payload: dict) -> dict:
    """UI-level limiter: enforce screen limits; do not invent business logic."""
    dropped = []
    if not isinstance(payload, dict):
        return payload

    meta = payload.setdefault("meta", {})
    followups = list(meta.get("followups") or [])
    if len(followups) > 2:
        dropped.append("followups_over_limit")
        meta["followups"] = followups[:2]

    refs = list(payload.get("quick_replies") or [])
    if len(refs) > 1:
        dropped.append("suggest_refs_over_limit")
        payload["quick_replies"] = refs[:1]

    if dropped:
        meta["ui_dropped"] = dropped
    return payload


def empty_question_response() -> dict:
    return {
        "answer": "Уточните вопрос.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"error": "empty_question"},
    }


def no_candidates_response() -> dict:
    return {
        "answer": "Пока не нашёл подходящий материал в базе. Сформулируйте вопрос иначе или выберите один из вариантов ниже.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"file": None},
    }


def reset_session_response(sid: str) -> dict:
    return {
        "answer": "Начнём заново. Чем помочь?",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"sid": sid},
    }


def internal_error_response() -> dict:
    return {
        "answer": "Извините, не получилось ответить. Попробуйте переформулировать вопрос.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {"error": "internal"},
    }


def low_score_response(sid: str, client_id: str | None = None) -> dict:
    """Fallback при top similarity < порога; CTA из конфига (policy не снимает low_score)."""
    meta_out: dict = {
        "low_score": True,
        "sid": sid,
        "score": None,
        "followups": [],
        "file": None,
    }
    if client_id is not None:
        meta_out["client_id"] = client_id
    return {
        "answer": "Не нашёл точного ответа на этот вопрос. Уточните или выберите тему.",
        "quick_replies": [],
        "cta": default_cta_dict(),
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": meta_out,
    }


def _suggest_refs_at_most_one(service: dict | None) -> list:
    """Не более одной кнопки suggest_ref; пустой список если данных нет."""
    refs = list((service or {}).get("suggest_refs") or [])
    if not refs:
        return []
    r0 = refs[0]
    if isinstance(r0, dict):
        label = (r0.get("label") or "").strip()
        ref = (r0.get("ref") or "").strip()
        if label and ref:
            return [{"label": label, "ref": ref}]
        return []
    if isinstance(r0, str) and "#" in r0:
        return [{"label": r0.split("#", 1)[0].strip() or "Подробнее", "ref": r0}]
    return []


def _format_price_value(price: dict) -> str | None:
    if not isinstance(price, dict):
        return None
    ptype = str(price.get("price_type") or "").strip().lower()
    cur = str(price.get("currency") or "RUB").strip().upper()
    symbol = "₽" if cur == "RUB" else cur
    if ptype == "fixed" and price.get("value") is not None:
        return f"{int(price['value'])} {symbol}"
    if ptype == "from" and price.get("value") is not None:
        return f"от {int(price['value'])} {symbol}"
    if ptype == "range" and price.get("value_min") is not None and price.get("value_max") is not None:
        return f"{int(price['value_min'])}–{int(price['value_max'])} {symbol}"
    return None


def build_service_facts_card_payload(
    *,
    sid: str,
    client_id: str | None,
    service_id: str,
    service: dict,
    match_score: float,
    user_question: str = "",
) -> dict:
    """Короткий ответ только из facts каталога (md_entry_ref = null)."""
    title = str((service or {}).get("title") or service_id).strip()
    facts = [
        str(x).strip()
        for x in ((service or {}).get("facts") or [])
        if str(x).strip()
    ]
    llm_answer = generate_facts_card_answer(
        title, facts, sid=sid, client_id=client_id, user_question=user_question
    )
    if llm_answer:
        answer = llm_answer
    else:
        lines = "\n".join(f"• {t}" for t in facts)
        answer = f"{title}\n\n{lines}" if lines else title
    quick = _suggest_refs_at_most_one(service)
    return {
        "answer": answer,
        "quick_replies": quick,
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {
            "sid": sid,
            "client_id": client_id,
            "intent": "catalog_facts",
            "matched_service_id": service_id,
            "match_score": round(float(match_score or 0.0), 4),
            "route_source": "catalog",
            "followups": [],
        },
    }


def build_price_lookup_payload(
    *,
    sid: str,
    client_id: str | None,
    service_id: str,
    service: dict,
    match_score: float,
    route_source: str,
    price_key: str | None,
    price_ref: str | None,
    price_item: dict | None,
) -> dict:
    title = str((service or {}).get("title") or service_id).strip()
    rendered = _format_price_value(price_item or {})
    note = (price_item or {}).get("note")
    if rendered:
        answer = f"{title}: {rendered}."
        if isinstance(note, str) and note.strip():
            answer += f" Важно: {note.strip()}."
    else:
        answer = (
            f"По услуге «{title}» сейчас не вижу точной цены в ценовом слое. "
            "Могу уточнить формат услуги или передать запрос администратору."
        )
    # price_lookup: ответ «закрыт», без кнопок и без CTA (сценарий «человек думает»).
    return {
        "answer": answer,
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {
            "sid": sid,
            "client_id": client_id,
            "intent": "price_lookup",
            "matched_service_id": service_id,
            "match_score": round(float(match_score or 0.0), 4),
            "route_source": route_source,
            "price_key": price_key,
            "price_ref": price_ref,
            "fallback_reason": None if rendered else "price_not_found",
            "followups": [],
        },
    }


def build_price_concern_payload(
    *,
    sid: str,
    client_id: str | None,
    service_id: str,
    service: dict,
    match_score: float,
) -> dict:
    title = str((service or {}).get("title") or service_id).strip()
    quick = _suggest_refs_at_most_one(service)
    return {
        "answer": (
            f"Понимаю сомнение по стоимости «{title}». "
            "Цена зависит от клинической ситуации, объема работ и выбранного протокола. "
            "На консультации можно подобрать вариант под ваш бюджет без лишних этапов."
        ),
        "quick_replies": quick,
        "cta": default_cta_dict(),
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {
            "sid": sid,
            "client_id": client_id,
            "intent": "price_concern",
            "matched_service_id": service_id,
            "match_score": round(float(match_score or 0.0), 4),
            "route_source": "catalog",
            "price_key": (service or {}).get("price_key"),
            "price_ref": (service or {}).get("price_ref"),
            "fallback_reason": None,
            "followups": [],
        },
    }


def build_price_clarify_payload(
    *,
    sid: str,
    client_id: str | None,
    intent: str,
    fallback_reason: str,
) -> dict:
    return {
        "answer": "Уточните, пожалуйста, какую именно услугу вы имеете в виду, и я отвечу точнее по цене.",
        "quick_replies": [],
        "cta": None,
        "video": None,
        "situation": {"show": False, "mode": "normal"},
        "offer": None,
        "meta": {
            "sid": sid,
            "client_id": client_id,
            "intent": intent,
            "matched_service_id": None,
            "match_score": 0.0,
            "route_source": "catalog",
            "price_key": None,
            "price_ref": None,
            "fallback_reason": fallback_reason,
            "followups": [],
        },
    }
