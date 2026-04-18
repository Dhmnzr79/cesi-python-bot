"""Chunk selection orchestration for /ask retrieval path."""

from config import (
    ALIAS_SOFT_THRESHOLD,
    ALIAS_STRONG_THRESHOLD,
    LOW_SCORE_THRESHOLD,
    QUERY_REWRITE_ON,
    RERANK_GAP_MAX,
    RERANK_NEAR_LOW_GAP_MAX,
    RERANK_NEAR_LOW_TOP_MAX,
    RERANK_TOP_MAX,
    RERANK_TOP_MIN,
)
from llm import rewrite_query_for_retrieval
from policy import contacts_intent, pick_contacts_chunk, pick_prices_chunk, price_intent
from retriever import (
    broad_query_detect,
    corpus_alias_leader,
    is_point_literal_query,
    llm_rerank,
    normalize_retrieval_query,
    prefer_overview_if_broad,
    retrieve,
)


def select_chunk_for_question(
    q: str, *, client_id: str | None, sid: str | None = None
) -> dict:
    """Return selection result for /ask.

    mode:
      - no_candidates
      - low_score
      - chunk
    """
    q_user = (q or "").strip()
    if sid and QUERY_REWRITE_ON:
        q_raw = rewrite_query_for_retrieval(sid, q_user, client_id=client_id)
    else:
        q_raw = q_user
    q_use = normalize_retrieval_query(q_raw) or q_raw
    base_meta = {
        "query_user_raw": q_user[:200],
        "query_raw": q_raw[:200],
        "query_normalized": q_use[:200],
        "rewrite_applied": bool(q_raw != q_user),
    }

    def _dm(extra: dict) -> dict:
        return {**base_meta, **extra}

    cands = retrieve(q_raw, topk=8, client_id=client_id)
    cands = prefer_overview_if_broad(cands, broad_query_detect(q_use))
    if not cands:
        return {
            "mode": "no_candidates",
            "debug_meta": _dm({"top_score": None}),
        }

    is_contacts = contacts_intent(q_use)
    is_price = price_intent(q_use)
    alias_leader, alias_score = corpus_alias_leader(q_use, client_id=client_id)
    alias_strong = bool(alias_leader and alias_score >= ALIAS_STRONG_THRESHOLD)

    top_score = float(cands[0].get("_score") or 0.0)
    allow_low = alias_strong or (is_contacts and pick_contacts_chunk(cands)) or (
        is_price and pick_prices_chunk(cands)
    )
    if top_score < LOW_SCORE_THRESHOLD and not allow_low:
        if alias_leader and alias_score >= ALIAS_SOFT_THRESHOLD:
            soft = dict(alias_leader)
            soft["_alias_score"] = round(alias_score, 4)
            soft["_score"] = round(float(alias_score), 4)
            return {
                "mode": "chunk",
                "chunk": soft,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "soft_alias_assist",
                        "top_score": round(top_score, 4),
                        "threshold": LOW_SCORE_THRESHOLD,
                        "alias_score": round(float(alias_score or 0.0), 4),
                        "is_contacts": bool(is_contacts),
                        "is_price": bool(is_price),
                    }
                ),
            }
        return {
            "mode": "low_score",
            "debug_meta": _dm(
                {
                    "top_score": round(top_score, 4),
                    "threshold": LOW_SCORE_THRESHOLD,
                    "alias_score": round(float(alias_score or 0.0), 4),
                    "is_contacts": bool(is_contacts),
                    "is_price": bool(is_price),
                }
            ),
        }

    if is_contacts:
        picked = pick_contacts_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "contacts",
                        "top_score": round(top_score, 4),
                        "alias_score": round(float(alias_score or 0.0), 4),
                    }
                ),
            }

    if is_price:
        picked = pick_prices_chunk(cands)
        if picked is not None:
            return {
                "mode": "chunk",
                "chunk": picked,
                "rerank_applied": False,
                "debug_meta": _dm(
                    {
                        "selected_by": "price",
                        "top_score": round(top_score, 4),
                        "alias_score": round(float(alias_score or 0.0), 4),
                    }
                ),
            }

    if alias_strong and alias_leader is not None:
        strong = dict(alias_leader)
        strong["_alias_score"] = round(alias_score, 4)
        strong["_score"] = round(float(alias_score), 4)
        return {
            "mode": "chunk",
            "chunk": strong,
            "rerank_applied": False,
            "debug_meta": _dm(
                {
                    "selected_by": "alias",
                    "alias_score": round(float(alias_score or 0.0), 4),
                    "top_score": round(top_score, 4),
                }
            ),
        }

    top = cands[0]
    score_gap = (
        abs(float(cands[0].get("_score") or 0.0) - float(cands[1].get("_score") or 0.0))
        if len(cands) >= 2
        else 1.0
    )
    narrow_rerank = (
        RERANK_TOP_MIN <= top_score <= RERANK_TOP_MAX
        and len(cands) >= 2
        and score_gap <= RERANK_GAP_MAX
        and not is_point_literal_query(q_use)
        and not alias_strong
    )
    near_low_rerank = (
        len(cands) >= 2
        and RERANK_TOP_MIN <= top_score <= RERANK_NEAR_LOW_TOP_MAX
        and top_score < LOW_SCORE_THRESHOLD + 0.02
        and score_gap <= RERANK_NEAR_LOW_GAP_MAX
        and not is_point_literal_query(q_use)
        and not alias_strong
    )
    use_rerank = narrow_rerank or near_low_rerank
    if use_rerank:
        top = llm_rerank(q_raw, cands[:3])

    return {
        "mode": "chunk",
        "chunk": top,
        "rerank_applied": bool(use_rerank),
        "debug_meta": _dm(
            {
                "selected_by": "semantic",
                "top_score": round(top_score, 4),
                "score_gap": round(float(score_gap), 4),
                "alias_score": round(float(alias_score or 0.0), 4),
                "rerank_near_low": bool(near_low_rerank and not narrow_rerank),
            }
        ),
    }
