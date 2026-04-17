"""Chunk selection orchestration for /ask retrieval path."""

from config import LOW_SCORE_THRESHOLD
from policy import contacts_intent, pick_contacts_chunk, pick_prices_chunk, price_intent
from retriever import (
    best_alias_hit_in_corpus,
    broad_query_detect,
    is_point_literal_query,
    llm_rerank,
    prefer_overview_if_broad,
    retrieve,
)


def select_chunk_for_question(q: str, *, client_id: str | None) -> dict:
    """Return selection result for /ask.

    mode:
      - no_candidates
      - low_score
      - chunk
    """
    cands = retrieve(q, topk=8, client_id=client_id)
    cands = prefer_overview_if_broad(cands, broad_query_detect(q))
    if not cands:
        return {"mode": "no_candidates", "debug_meta": {"top_score": None}}

    is_contacts = contacts_intent(q)
    is_price = price_intent(q)
    alias_chunk, alias_score = best_alias_hit_in_corpus(
        q,
        client_id=client_id,
        strong_threshold=0.82,
    )
    alias_strong = alias_chunk is not None
    top_score = float(cands[0].get("_score") or 0.0)
    allow_low = alias_strong or (is_contacts and pick_contacts_chunk(cands)) or (
        is_price and pick_prices_chunk(cands)
    )
    if top_score < LOW_SCORE_THRESHOLD and not allow_low:
        return {
            "mode": "low_score",
            "debug_meta": {
                "top_score": round(top_score, 4),
                "threshold": LOW_SCORE_THRESHOLD,
                "alias_score": round(float(alias_score or 0.0), 4),
                "is_contacts": bool(is_contacts),
                "is_price": bool(is_price),
            },
        }

    if is_contacts:
        picked = pick_contacts_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": {
                    "selected_by": "contacts",
                    "top_score": round(top_score, 4),
                    "alias_score": round(float(alias_score or 0.0), 4),
                },
            }

    if is_price:
        picked = pick_prices_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": {
                    "selected_by": "price",
                    "top_score": round(top_score, 4),
                    "alias_score": round(float(alias_score or 0.0), 4),
                },
            }

    if alias_strong and alias_chunk is not None:
        return {
            "mode": "chunk",
            "chunk": alias_chunk,
            "rerank_applied": False,
            "debug_meta": {
                "selected_by": "alias",
                "alias_score": round(float(alias_score or 0.0), 4),
                "top_score": round(top_score, 4),
            },
        }

    top = cands[0]
    score_gap = (
        abs(float(cands[0].get("_score") or 0.0) - float(cands[1].get("_score") or 0.0))
        if len(cands) >= 2
        else 1.0
    )
    use_rerank = (
        0.45 <= top_score <= 0.62
        and len(cands) >= 2
        and score_gap <= 0.05
        and not is_point_literal_query(q)
        and not alias_strong
    )
    if use_rerank:
        top = llm_rerank(q, cands[:3])

    return {
        "mode": "chunk",
        "chunk": top,
        "rerank_applied": bool(use_rerank),
        "debug_meta": {
            "selected_by": "semantic",
            "top_score": round(top_score, 4),
            "score_gap": round(float(score_gap), 4),
            "alias_score": round(float(alias_score or 0.0), 4),
        },
    }
