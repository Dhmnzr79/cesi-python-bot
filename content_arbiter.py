"""Content candidate collection for A5 (v5).

Runtime route choice among candidates: `arbiter.decide_content_route` (LLM Arbiter when 2+ distinct refs).

Retrieval runs at most once per collect (via ``select_chunk_for_question`` and its rerank path).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from core.routing_loader import THRESHOLDS
from query_selector import select_catalog_content_route, select_chunk_for_question
from retriever import broad_query_detect, corpus_alias_leader, normalize_retrieval_query
from session import mem_get


def _slim_chunk_for_log(chunk: dict | None) -> dict | None:
    if not isinstance(chunk, dict):
        return None
    meta = chunk.get("meta") or {}
    file = chunk.get("file")
    return {
        "file": os.path.basename(str(file or "")) or None,
        "doc_id": os.path.splitext(os.path.basename(str(file or "")))[0] if file else None,
        "h2_id": chunk.get("h2_id") or meta.get("h2_id"),
        "h3_id": chunk.get("h3_id") or meta.get("h3_id"),
        "score": float(chunk.get("_score")) if chunk.get("_score") is not None else None,
        "doc_type": meta.get("doc_type") or chunk.get("doc_type"),
        "subtype": meta.get("subtype") or chunk.get("subtype"),
        "snippet": (str(chunk.get("text") or "").strip()[:180] or None),
    }


def _doc_id_from_ref(ref: str) -> str | None:
    """`implantation__service__classic.md#korotko` -> `implantation__service__classic`."""
    if not ref:
        return None
    fname = ref.split("#", 1)[0].strip()
    base = os.path.basename(fname)
    if base.lower().endswith(".md"):
        base = base[:-3]
    return base.strip() or None


def _anchor_from_ref(ref: str) -> str:
    if not ref or "#" not in ref:
        return ""
    return (ref.split("#", 1)[1] or "").strip().lower()


def _is_overview_anchor(anchor: str) -> bool:
    a = (anchor or "").strip().lower()
    return a in {"", "overview", "korotko"}


def _classify_doc_kind(doc_id: str | None) -> str | None:
    """Classify by doc_id naming convention from map.md contract."""
    if not doc_id:
        return None
    d = doc_id.strip().lower()
    if "__faq__" in d:
        return "faq_specific"
    if "__info__" in d:
        return "info_specific"
    if "__pricing__" in d:
        return "pricing_specific"
    if d.startswith("doctors__doctor__") or "__doctor__" in d:
        return "doctor_specific"
    if "__service__" in d:
        return "service"
    return None


def _classify_chunk_kind(chunk: dict) -> str | None:
    if not isinstance(chunk, dict):
        return None
    meta = chunk.get("meta") or {}
    doc_type = str(meta.get("doc_type") or chunk.get("doc_type") or "").strip().lower()
    subtype = str(meta.get("subtype") or chunk.get("subtype") or "").strip().lower()

    # Prefer explicit doc_type/subtype over filename conventions.
    if doc_type == "faq":
        return "faq_specific"
    if doc_type == "info":
        return "info_specific"
    if doc_type == "pricing":
        return "pricing_specific"
    if doc_type == "doctor":
        return "doctor_specific"
    if doc_type == "service":
        h3_id = (chunk.get("h3_id") or meta.get("h3_id") or "").strip().lower()
        return "service_overview" if _is_overview_anchor(h3_id) else "service_section"

    # subtype should promote to "specific" when doc_type isn't set.
    if subtype in {
        "faq",
        "clinical_faq",
        "specific",
        "contraindications",
        "aftercare",
        "pain",
        "safety",
        "timing",
    }:
        return "faq_specific"

    file = os.path.basename(str(chunk.get("file") or ""))
    doc_id = os.path.splitext(file)[0] if file else None
    base_kind = _classify_doc_kind(doc_id)
    if base_kind != "service":
        return base_kind
    h3_id = (chunk.get("h3_id") or meta.get("h3_id") or "").strip().lower()
    return "service_overview" if _is_overview_anchor(h3_id) else "service_section"


def _alias_specificity(alias_text: str | None) -> str | None:
    """Broad vs specific by token count (no hand-crafted phrase exceptions)."""
    s = (alias_text or "").strip()
    if not s:
        return None
    toks = [t for t in s.replace("ё", "е").split() if t.strip()]
    if len(toks) <= 1:
        return "broad_service"
    if 2 <= len(toks) <= 4:
        return "specific"
    return "specific"


def _token_count(s: str) -> int:
    return len([t for t in (s or "").split() if t.strip()])


def _has_specific_modifier(s: str) -> bool:
    """Generic signal that the question is about a specific method/thing.

    Avoids phrase-specific exceptions.
    """
    x = (s or "").strip()
    if not x:
        return False
    if re.search(r"[0-9]", x):
        return True
    if re.search(r"[a-z]", x, flags=re.I):
        return True
    if "-" in x or "—" in x:
        return True
    return False


@dataclass(frozen=True)
class ContentCandidates:
    retrieval: dict
    catalog: dict
    alias: dict
    session: dict
    debug_meta: dict


@dataclass(frozen=True)
class ContentRouteResult:
    kind: str  # chunk | service | guided | fallback
    selected_route: str
    selected_doc_id: str | None
    selected_chunk: dict | None
    reason: str
    debug_meta: dict
    candidates: dict
    rejected_candidates: list[dict]


def collect_content_candidates(
    *,
    q: str,
    sid: str,
    client_id: str | None,
    scope_topic: str | None = None,
    catalog_md_priority_ref: str | None = None,
    catalog_md_priority_service_id: str | None = None,
    catalog_md_priority_match_score: float | None = None,
) -> ContentCandidates:
    q_user = (q or "").strip()
    q_norm = normalize_retrieval_query(q_user) or q_user

    selection = select_chunk_for_question(
        q_user, client_id=client_id, sid=sid, scope_topic=scope_topic
    )
    retrieval_mode = str(selection.get("mode") or "")
    retrieval_chunk = selection.get("chunk") if retrieval_mode == "chunk" else None
    retrieval_kind = _classify_chunk_kind(retrieval_chunk) if isinstance(retrieval_chunk, dict) else None
    retrieval_doc_id = None
    if isinstance(retrieval_chunk, dict):
        file = os.path.basename(str(retrieval_chunk.get("file") or ""))
        retrieval_doc_id = os.path.splitext(file)[0] if file else None

    cat = select_catalog_content_route(q_user, client_id=client_id)
    cat_mode = str(cat.get("mode") or "none")
    md_ref = str(cat.get("md_entry_ref") or "").strip() if cat_mode == "md_first" else ""
    cat_doc_id = _doc_id_from_ref(md_ref) if md_ref else None
    cat_anchor = _anchor_from_ref(md_ref) if md_ref else ""
    cat_is_overview = bool(md_ref and _is_overview_anchor(cat_anchor))

    alias_leader, alias_score, alias_diag = corpus_alias_leader(q_norm, client_id=client_id)
    # `corpus_alias_leader` returns (chunk, score, diag). PR #1.10: diag carries alias_* telemetry.
    # For P0 we approximate "specificity" from the normalized user query token count.
    # This stays deterministic and avoids any per-phrase keyword exceptions.
    alias_text = q_norm
    alias_spec = _alias_specificity(alias_text)
    alias_kind = _classify_chunk_kind(alias_leader) if isinstance(alias_leader, dict) else None
    alias_doc_id = None
    if isinstance(alias_leader, dict):
        file_a = os.path.basename(str(alias_leader.get("file") or ""))
        alias_doc_id = os.path.splitext(file_a)[0] if file_a else None

    st = mem_get(sid) if sid else {}
    session_ctx = {
        "current_doc_id": (st.get("current_doc_id") or None),
        "last_catalog_service_id": (st.get("last_catalog_service_id") or None),
    }

    retrieval_candidate = {
        "mode": retrieval_mode,
        "doc_id": retrieval_doc_id,
        "chunk_kind": retrieval_kind,
        # Full chunk is kept in-memory for immediate execution when selected.
        # Never log this dict as-is: it may contain large `text`.
        "chunk": retrieval_chunk,
        "chunk_slim": _slim_chunk_for_log(retrieval_chunk),
        "debug_meta": selection.get("debug_meta") or {},
        "rerank_applied": bool(selection.get("rerank_applied")),
    }
    catalog_candidate = {
        "mode": cat_mode,
        "matched_service_id": cat.get("matched_service_id"),
        "match_score": cat.get("match_score"),
        "md_entry_ref": md_ref,
        "doc_id": cat_doc_id,
        "is_overview": cat_is_overview,
        "service": cat.get("service") if isinstance(cat.get("service"), dict) else {},
    }
    prref = (catalog_md_priority_ref or "").strip()
    if prref:
        pri_score = catalog_md_priority_match_score
        if pri_score is None:
            pri_score = float(THRESHOLDS.catalog_match.containment_min)
        cat_doc_id2 = _doc_id_from_ref(prref)
        cat_anchor2 = _anchor_from_ref(prref)
        catalog_candidate = {
            "mode": "md_first",
            "matched_service_id": catalog_md_priority_service_id or cat.get("matched_service_id"),
            "match_score": pri_score,
            "md_entry_ref": prref,
            "doc_id": cat_doc_id2,
            "is_overview": bool(prref and _is_overview_anchor(cat_anchor2)),
            "service": cat.get("service") if isinstance(cat.get("service"), dict) else {},
        }
    alias_tel = {
        k: v
        for k, v in (alias_diag or {}).items()
        if k.startswith("alias_") or k.startswith("old_")
    }
    alias_candidate = {
        "leader": _slim_chunk_for_log(alias_leader) if isinstance(alias_leader, dict) else None,
        # Full chunk kept for immediate execution if selected; never log this.
        "leader_chunk": alias_leader if isinstance(alias_leader, dict) else None,
        "leader_kind": alias_kind,
        "leader_doc_id": alias_doc_id,
        "alias_text": alias_text[:120] if isinstance(alias_text, str) else None,
        "alias_score": float(alias_score or 0.0) if alias_score is not None else None,
        "specificity": alias_spec,
        **alias_tel,
    }

    debug_meta = {
        "q_norm": q_norm[:200],
        "broad_query": bool(broad_query_detect(q_norm)),
        "token_count": _token_count(q_user),
        "has_specific_modifier": bool(_has_specific_modifier(q_user)),
    }

    return ContentCandidates(
        retrieval=retrieval_candidate,
        catalog=catalog_candidate,
        alias=alias_candidate,
        session=session_ctx,
        debug_meta=debug_meta,
    )

