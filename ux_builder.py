"""Сборка JSON ответа /ask (одна точка сборки)."""
import os
import re

from config import default_cta_dict
from meta_loader import get_doc_path


def get_chunk_ids(chunk: dict) -> tuple:
    if not isinstance(chunk, dict):
        return (None, None)
    return (chunk.get("h2_id"), chunk.get("h3_id"))


def is_overview_by_ids(h2_id, h3_id) -> bool:
    h2 = (h2_id or "").strip().lower()
    h3 = (h3_id or "").strip().lower()
    return (not h2 and not h3) or (h2 == "overview") or (h3 == "overview")


def heading_label(md_file: str, sect_id: str) -> str:
    if not md_file or not sect_id:
        return (sect_id or "").replace("-", " ").capitalize()
    try:
        path = get_doc_path(os.path.basename(md_file)) or md_file
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


def build_followups(meta: dict, md_file: str, current_h2_id: str, current_h3_id: str) -> list:
    out = []
    for s in meta.get("suggest_h3") or []:
        h_id = s if isinstance(s, str) else (s.get("h3_id") or s.get("id"))
        if not h_id:
            continue
        if str(h_id).lower() in {
            str(current_h2_id or "").lower(),
            str(current_h3_id or "").lower(),
        }:
            continue
        label = heading_label(md_file, h_id)
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
) -> dict:
    """Единая структура успешного ответа /ask (как раньше: quick_replies + meta.followups)."""
    md_file = top.get("file")
    h2_id, h3_id = get_chunk_ids(top)
    h2_val = top.get("h2") or top.get("h2_id")
    h3_val = top.get("h3") or top.get("h3_id")
    is_overview = is_overview_by_ids(h2_id, h3_id)

    quick_refs = build_quick_refs(meta, md_file, h2_id, h3_id)
    fups_full = build_followups(meta, md_file, h2_id, h3_id)
    followups = fups_full[:1] if is_overview else []

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
        "offer": pick_relevant_offer(meta),
        "meta": meta_out,
    }


def empty_question_response() -> dict:
    return {
        "answer": "Уточните вопрос.",
        "quick_replies": [],
        "cta": None,
        "offer": None,
        "meta": {"error": "empty_question"},
    }


def no_candidates_response() -> dict:
    return {
        "answer": "Пока не нашёл подходящий материал в базе. Сформулируйте вопрос иначе или выберите один из вариантов ниже.",
        "quick_replies": [],
        "cta": None,
        "offer": None,
        "meta": {"file": None},
    }


def reset_session_response(sid: str) -> dict:
    return {
        "answer": "Начнём заново. Чем помочь?",
        "quick_replies": [],
        "cta": None,
        "offer": None,
        "meta": {"sid": sid},
    }


def internal_error_response() -> dict:
    return {
        "answer": "Извините, не получилось ответить. Попробуйте переформулировать вопрос.",
        "quick_replies": [],
        "cta": None,
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
        "offer": None,
        "meta": meta_out,
    }
