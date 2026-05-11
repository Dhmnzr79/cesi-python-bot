"""A5 Arbiter — выбор лучшего content-источника (structured ArbiterDecision).

PR #1.7: `decide_content_route` — единственный runtime route-decider для content (без legacy if-rules).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import ValidationError

from contracts.arbiter_decision import ArbiterDecision
from contracts.decision_frame import DecisionFrame
from content_arbiter import ContentCandidates, ContentRouteResult
from config import CHAT_MODEL
from core.routing_loader import THRESHOLDS
from llm import client
from retriever import get_chunk_by_ref
from logging_setup import get_logger, log_llm_error, log_llm_usage

logger = get_logger("bot")

_MODEL = (os.getenv("MODEL_ARBITER") or "").strip() or CHAT_MODEL
_TIMEOUT_SEC = float(os.getenv("V5_ARBITER_TIMEOUT_SEC", "12"))

ArbiterRunStatus = Literal["ok", "skipped", "error", "fallback"]
ArbiterCallType = Literal["v5_arbiter"]


def with_default_anchor(md_entry_ref: str) -> str:
    ref = (md_entry_ref or "").strip()
    if not ref:
        return ""
    return ref if "#" in ref else f"{ref}#korotko"


def canonical_ref(ref: str) -> str:
    """Normalize ref for dedup / agreement checks."""
    r = (ref or "").strip().lower().replace("\\", "/")
    if "#" not in r and r:
        r = f"{r}#korotko"
    left, _, right = r.partition("#")
    base = os.path.basename(left.strip())
    if base and not base.endswith(".md"):
        base = f"{base}.md"
    return f"{base}#{right.strip().lower()}"


def ref_from_chunk(ch: dict) -> str | None:
    if not isinstance(ch, dict):
        return None
    meta = ch.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    file = str(ch.get("file") or "")
    base = os.path.basename(file)
    if not base:
        return None
    if not base.lower().endswith(".md"):
        base = f"{base}.md"
    h3 = str(ch.get("h3_id") or meta.get("h3_id") or "").strip()
    h2 = str(ch.get("h2_id") or meta.get("h2_id") or "").strip()
    anchor = (h3 or h2 or "korotko").strip().lower() or "korotko"
    return f"{base}#{anchor}"


def _score_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _source_priority(source_kind: str) -> int:
    order = ("retrieval", "catalog", "alias", "session", "eval_golden")
    sk = (source_kind or "").strip().lower()
    try:
        return order.index(sk)
    except ValueError:
        return len(order)


def build_compact_content_candidates(
    cands: ContentCandidates,
    *,
    client_id: str | None = None,
) -> list[dict[str, Any]]:
    """Компактные кандидаты для Arbiter (без полного markdown). Дедуп по ref."""
    out_map: dict[str, dict[str, Any]] = {}

    def put(
        *,
        ref: str,
        source_kind: str,
        score: float | None,
        doc_type: str | None,
        subtype: str | None,
        topic: str | None,
        service_id: str | None,
        snippet: str | None,
        why: str | None,
    ) -> None:
        r = (ref or "").strip()
        if not r or "#" not in r:
            return
        key = canonical_ref(r)
        prev = out_map.get(key)
        sc = float(score) if score is not None else 0.0
        if prev is None:
            out_map[key] = {
                "ref": r,
                "source_kind": source_kind,
                "score": score,
                "doc_type": doc_type,
                "subtype": subtype,
                "topic": topic,
                "service_id": service_id,
                "snippet": (snippet or "")[:220] or None,
                "why": why,
            }
            return
        prev_sc = _score_float(prev.get("score"))
        prev_sc_f = float(prev_sc) if prev_sc is not None else 0.0
        if sc > prev_sc_f or (
            sc == prev_sc_f and _source_priority(source_kind) < _source_priority(str(prev.get("source_kind") or ""))
        ):
            out_map[key] = {
                "ref": r,
                "source_kind": source_kind,
                "score": score,
                "doc_type": doc_type,
                "subtype": subtype,
                "topic": topic,
                "service_id": service_id,
                "snippet": (snippet or "")[:220] or None,
                "why": why,
            }

    ret = cands.retrieval or {}
    if str(ret.get("mode") or "") == "chunk":
        ch = ret.get("chunk") if isinstance(ret.get("chunk"), dict) else None
        if isinstance(ch, dict):
            rr = ref_from_chunk(ch)
            meta = ch.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            if rr:
                slim = ret.get("chunk_slim") if isinstance(ret.get("chunk_slim"), dict) else {}
                snip = str((slim or {}).get("snippet") or ch.get("text") or "")[:220] or None
                rdbg = ret.get("debug_meta") if isinstance(ret.get("debug_meta"), dict) else {}
                why = None
                if isinstance(rdbg, dict) and rdbg.get("selected_by"):
                    why = f"retrieval:{rdbg.get('selected_by')}"
                put(
                    ref=rr,
                    source_kind="retrieval",
                    score=_score_float(ch.get("_score")),
                    doc_type=str(meta.get("doc_type") or ch.get("doc_type") or "") or None,
                    subtype=str(meta.get("subtype") or ch.get("subtype") or "") or None,
                    topic=str(meta.get("topic") or meta.get("service_topic") or "") or None,
                    service_id=None,
                    snippet=snip,
                    why=why,
                )

    cat = cands.catalog or {}
    cat_mode = str(cat.get("mode") or "none")
    if cat_mode == "md_first":
        md_ref = with_default_anchor(str(cat.get("md_entry_ref") or ""))
        if md_ref:
            svc = cat.get("service") if isinstance(cat.get("service"), dict) else {}
            title = str((svc or {}).get("title") or (svc or {}).get("name") or "")[:120] or None
            put(
                ref=md_ref,
                source_kind="catalog",
                score=_score_float(cat.get("match_score")),
                doc_type="catalog_md",
                subtype=None,
                topic=str(cat.get("doc_id") or "").split("__")[0] if cat.get("doc_id") else None,
                service_id=str(cat.get("matched_service_id") or "") or None,
                snippet=title,
                why="catalog_md_first",
            )

    alias = cands.alias or {}
    ach = alias.get("leader_chunk") if isinstance(alias.get("leader_chunk"), dict) else None
    if isinstance(ach, dict):
        rr = ref_from_chunk(ach)
        meta = ach.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        if rr:
            slim_a = alias.get("leader") if isinstance(alias.get("leader"), dict) else {}
            snip = str((slim_a or {}).get("snippet") or "")[:220] or None
            put(
                ref=rr,
                source_kind="alias",
                score=_score_float(alias.get("alias_score")),
                doc_type=str(meta.get("doc_type") or ach.get("doc_type") or "") or None,
                subtype=str(meta.get("subtype") or ach.get("subtype") or "") or None,
                topic=None,
                service_id=None,
                snippet=snip,
                why="corpus_alias_leader",
            )

    sess = cands.session or {}
    cur = str(sess.get("current_doc_id") or "").strip()
    if cur:
        doc = cur.removesuffix(".md")
        sref = with_default_anchor(f"{doc}.md")
        if sref and get_chunk_by_ref(sref, client_id=client_id) is not None:
            put(
                ref=sref,
                source_kind="session",
                score=0.25,
                doc_type="session",
                subtype=None,
                topic=None,
                service_id=None,
                snippet="session_current_doc",
                why="session_current_doc_id",
            )

    merged = list(out_map.values())
    merged.sort(key=lambda x: (-float(_score_float(x.get("score")) or 0.0), _source_priority(str(x.get("source_kind") or ""))))
    return merged


def _doc_id_from_chunk(ch: dict) -> str | None:
    file = os.path.basename(str(ch.get("file") or ""))
    return os.path.splitext(file)[0] if file else None


def _candidate_bundle(candidates: ContentCandidates) -> dict[str, Any]:
    ret = candidates.retrieval if isinstance(candidates.retrieval, dict) else {}
    cat = candidates.catalog if isinstance(candidates.catalog, dict) else {}
    alias = candidates.alias if isinstance(candidates.alias, dict) else {}
    return {
        "retrieval_candidate": ret,
        "catalog_candidate": cat,
        "alias_candidate": alias,
        "session_context_candidate": candidates.session,
    }


def _materialize_compact_row(
    *,
    row: dict[str, Any],
    cands: ContentCandidates,
    client_id: str | None,
) -> tuple[str, str | None, dict | None] | None:
    """Map one compact row to (selected_route, selected_doc_id, selected_chunk). None if not materializable."""
    sk = str(row.get("source_kind") or "").strip().lower()
    ref = str(row.get("ref") or "").strip()
    if not ref:
        return None
    want = canonical_ref(ref)

    ret = cands.retrieval if isinstance(cands.retrieval, dict) else {}
    cat = cands.catalog if isinstance(cands.catalog, dict) else {}
    alias = cands.alias if isinstance(cands.alias, dict) else {}

    if sk == "retrieval":
        ch = ret.get("chunk") if isinstance(ret.get("chunk"), dict) else None
        if not isinstance(ch, dict):
            return None
        rr = ref_from_chunk(ch)
        if not rr or canonical_ref(rr) != want:
            return None
        return ("retrieval_chunk", _doc_id_from_chunk(ch), ch)

    if sk == "alias":
        ch = alias.get("leader_chunk") if isinstance(alias.get("leader_chunk"), dict) else None
        if not isinstance(ch, dict):
            return None
        rr = ref_from_chunk(ch)
        if not rr or canonical_ref(rr) != want:
            return None
        return ("retrieval_chunk", _doc_id_from_chunk(ch), ch)

    if sk == "catalog":
        md_ref = with_default_anchor(str(cat.get("md_entry_ref") or ""))
        if not md_ref or canonical_ref(md_ref) != want:
            return None
        left = md_ref.split("#", 1)[0].strip()
        base = os.path.basename(left)
        doc_id = base[:-3] if base.lower().endswith(".md") else base or None
        return ("catalog_md_first", doc_id or None, None)

    if sk == "session":
        ch = get_chunk_by_ref(ref, client_id=client_id)
        if not isinstance(ch, dict):
            return None
        rr = ref_from_chunk(ch)
        if not rr or canonical_ref(rr) != want:
            return None
        return ("retrieval_chunk", _doc_id_from_chunk(ch), ch)

    return None


def _row_for_selected_ref(compact: list[dict[str, Any]], selected_ref: str) -> dict[str, Any] | None:
    w = canonical_ref(selected_ref)
    for row in compact:
        if not isinstance(row, dict):
            continue
        r = str(row.get("ref") or "").strip()
        if r and canonical_ref(r) == w:
            return row
    return None


def decide_content_route(
    *,
    q: str,
    sid: str,
    client_id: str | None,
    candidates: ContentCandidates,
    decision_frame: DecisionFrame | dict[str, Any] | None = None,
) -> ContentRouteResult:
    """A5 ON: 0 compact → guided (или catalog_facts); 1 → shortcut; 2+ → LLM Arbiter."""
    _ = sid
    compact = build_compact_content_candidates(candidates, client_id=client_id)
    bundle = _candidate_bundle(candidates)
    base_debug = dict(candidates.debug_meta) if isinstance(candidates.debug_meta, dict) else {}
    cat = candidates.catalog if isinstance(candidates.catalog, dict) else {}
    cat_mode = str(cat.get("mode") or "none")
    min_c = float(THRESHOLDS.arbiter.min_confidence)

    def _refs_list() -> list[str]:
        return [str(x.get("ref") or "") for x in compact if isinstance(x, dict) and str(x.get("ref") or "").strip()]

    def guided_result(
        *,
        reason: str,
        selected_by: str,
        trace: dict[str, Any],
        rejected: list[dict] | None = None,
    ) -> ContentRouteResult:
        dm = {**base_debug, "selected_by": selected_by, **trace}
        return ContentRouteResult(
            kind="guided",
            selected_route="guided",
            selected_doc_id=None,
            selected_chunk=None,
            reason=reason,
            debug_meta=dm,
            candidates=bundle,
            rejected_candidates=list(rejected or []),
        )

    n = len(compact)
    if n == 0:
        if cat_mode == "facts":
            dm = {
                **base_debug,
                "selected_by": "catalog_facts",
                "candidate_count": 0,
                "candidate_refs": [],
                "min_confidence": min_c,
                "arbiter_status": "not_invoked",
                "arbiter_selected_ref": None,
                "arbiter_confidence": None,
                "arbiter_reason": None,
                "arbiter_alternative": None,
            }
            return ContentRouteResult(
                kind="service",
                selected_route="catalog_facts",
                selected_doc_id=None,
                selected_chunk=None,
                reason="catalog_facts_no_compact_candidates",
                debug_meta=dm,
                candidates=bundle,
                rejected_candidates=[],
            )
        return guided_result(
            reason="no_content_candidates",
            selected_by="guided_no_candidates",
            trace={
                "candidate_count": 0,
                "candidate_refs": [],
                "min_confidence": min_c,
                "arbiter_status": "not_invoked",
                "arbiter_selected_ref": None,
                "arbiter_confidence": None,
                "arbiter_reason": None,
                "arbiter_alternative": None,
            },
        )

    if n == 1:
        row = compact[0]
        if not isinstance(row, dict):
            return guided_result(
                reason="selected_ref_unmaterializable",
                selected_by="guided_selected_ref_unmaterializable",
                trace={
                    "candidate_count": 1,
                    "candidate_refs": _refs_list(),
                    "min_confidence": min_c,
                    "arbiter_status": "not_invoked",
                    "arbiter_selected_ref": None,
                    "arbiter_confidence": None,
                    "arbiter_reason": None,
                    "arbiter_alternative": None,
                },
            )
        cr = _refs_list()
        trace_one = {
            "candidate_count": 1,
            "candidate_refs": cr,
            "min_confidence": min_c,
            "arbiter_status": "not_invoked",
            "arbiter_selected_ref": cr[0] if cr else None,
            "arbiter_confidence": None,
            "arbiter_reason": None,
            "arbiter_alternative": None,
        }
        mat = _materialize_compact_row(row=row, cands=candidates, client_id=client_id)
        if mat is None:
            return guided_result(
                reason="selected_ref_unmaterializable",
                selected_by="guided_selected_ref_unmaterializable",
                trace=trace_one,
            )
        route, doc_id, chunk = mat
        dm = {**base_debug, "selected_by": "shortcut_single_candidate", **trace_one}
        return ContentRouteResult(
            kind="chunk",
            selected_route=route,
            selected_doc_id=doc_id,
            selected_chunk=chunk,
            reason="shortcut_single_candidate",
            debug_meta=dm,
            candidates=bundle,
            rejected_candidates=[],
        )

    arb_dec, run_status, err = arbitrate_among_candidates(
        question=q,
        candidates=compact,
        decision_frame=decision_frame,
        call_type="v5_arbiter",
    )
    refs = _refs_list()
    trace_head = {
        "candidate_count": n,
        "candidate_refs": refs,
        "min_confidence": min_c,
    }

    if str(run_status or "") != "ok":
        return guided_result(
            reason=str(err or run_status or "arbiter_not_ok"),
            selected_by="guided_arbiter_fallback",
            trace={
                **trace_head,
                "arbiter_status": str(run_status or ""),
                "arbiter_selected_ref": None,
                "arbiter_confidence": None,
                "arbiter_reason": (err or str(run_status or ""))[:800],
                "arbiter_alternative": None,
            },
        )

    assert arb_dec is not None
    trace_ok = {
        **trace_head,
        "arbiter_status": "ok",
        "arbiter_selected_ref": arb_dec.selected_ref,
        "arbiter_confidence": float(arb_dec.confidence),
        "arbiter_reason": arb_dec.reason,
        "arbiter_alternative": arb_dec.alternative,
    }

    if float(arb_dec.confidence) < min_c:
        dm = {**base_debug, "selected_by": "guided_low_confidence", **trace_ok}
        return ContentRouteResult(
            kind="guided",
            selected_route="guided",
            selected_doc_id=None,
            selected_chunk=None,
            reason="arbiter_below_min_confidence",
            debug_meta=dm,
            candidates=bundle,
            rejected_candidates=[],
        )

    row = _row_for_selected_ref(compact, arb_dec.selected_ref)
    if row is None:
        dm = {**base_debug, "selected_by": "guided_selected_ref_unmaterializable", **trace_ok}
        return ContentRouteResult(
            kind="guided",
            selected_route="guided",
            selected_doc_id=None,
            selected_chunk=None,
            reason="selected_ref_unmaterializable",
            debug_meta=dm,
            candidates=bundle,
            rejected_candidates=[],
        )

    mat = _materialize_compact_row(row=row, cands=candidates, client_id=client_id)
    if mat is None:
        dm = {**base_debug, "selected_by": "guided_selected_ref_unmaterializable", **trace_ok}
        return ContentRouteResult(
            kind="guided",
            selected_route="guided",
            selected_doc_id=None,
            selected_chunk=None,
            reason="selected_ref_unmaterializable",
            debug_meta=dm,
            candidates=bundle,
            rejected_candidates=[],
        )

    route, doc_id, chunk = mat
    dm = {**base_debug, "selected_by": "v5_arbiter_on", **trace_ok}
    return ContentRouteResult(
        kind="chunk",
        selected_route=route,
        selected_doc_id=doc_id,
        selected_chunk=chunk,
        reason="v5_arbiter_selected",
        debug_meta=dm,
        candidates=bundle,
        rejected_candidates=[],
    )


def _fallback_from_candidates(candidates: list[dict[str, Any]]) -> ArbiterDecision:
    if not candidates:
        return ArbiterDecision(
            selected_ref="clinic__info__consultation.md#korotko",
            confidence=0.0,
            reason="arbiter_fallback",
            alternative=None,
        )
    best = max(
        candidates,
        key=lambda c: (
            float(_score_float(c.get("score")) or 0.0),
            -_source_priority(str(c.get("source_kind") or "")),
        ),
    )
    ref = str(best.get("ref") or "").strip() or "missing"
    alts = [c for c in candidates if canonical_ref(str(c.get("ref") or "")) != canonical_ref(ref)]
    alt_ref = str(alts[0].get("ref")).strip() if alts else None
    return ArbiterDecision(
        selected_ref=ref,
        confidence=float(_score_float(best.get("score")) or 0.0),
        reason="arbiter_fallback",
        alternative=alt_ref,
    )


ARBITER_SYSTEM_PROMPT = (
    "Ты — Arbiter слоя A5 (v5). По вопросу пациента и списку кандидатов выбери ОДИН лучший источник "
    "(markdown-документ с якорем).\n"
    "Верни только JSON (без markdown) со строго этими ключами:\n"
    "selected_ref, confidence, reason, alternative\n"
    "\n"
    "Правила:\n"
    "- selected_ref ДОЛЖЕН быть ТОЧНО одной из строк поля `ref` кандидатов (копируй буквально).\n"
    "- alternative — вторая по полезности строка `ref` из того же списка, или null.\n"
    "- confidence: число от 0 до 1.\n"
    "- reason: кратко по-русски (1–2 предложения), без выдуманных фактов.\n"
    "- Учитывай doc_type/subtype/topic, score и snippet только как сигналы релевантности.\n"
    "- Для узкого конкретного вопроса предпочитай faq/info/pricing/doctor вместо широкого service overview.\n"
    "- Для явного вопроса про врачей предпочитай doctor-документ.\n"
    "- Для вопроса про адрес/телефон/режим — contacts.\n"
    "- Для вопроса про гарантию — warranty info, если такой кандидат есть.\n"
)


def _arbiter_user_payload(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    decision_frame: DecisionFrame | dict[str, Any] | None,
) -> str:
    ctx: dict[str, Any] = {"question": (question or "").strip()}
    if isinstance(decision_frame, DecisionFrame):
        ctx["decision_frame"] = {
            "route_intent": decision_frame.route_intent,
            "service_topic": decision_frame.service_topic,
            "service_id": decision_frame.service_id,
            "query_mode": decision_frame.query_mode,
        }
    elif isinstance(decision_frame, dict):
        ctx["decision_frame"] = {
            k: decision_frame.get(k)
            for k in ("route_intent", "service_topic", "service_id", "query_mode")
            if k in decision_frame
        }
    slim_cands = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        slim_cands.append(
            {
                "ref": c.get("ref"),
                "source_kind": c.get("source_kind"),
                "doc_type": c.get("doc_type"),
                "subtype": c.get("subtype"),
                "topic": c.get("topic"),
                "service_id": c.get("service_id"),
                "score": c.get("score"),
                "snippet": c.get("snippet"),
                "why": c.get("why"),
            }
        )
    ctx["candidates"] = slim_cands
    return json.dumps(ctx, ensure_ascii=False)


def _validate_refs(decision: ArbiterDecision, allowed: set[str]) -> bool:
    canon_allowed = {canonical_ref(x) for x in allowed}
    if canonical_ref(decision.selected_ref) not in canon_allowed:
        return False
    if decision.alternative is None:
        return True
    alt = str(decision.alternative).strip()
    if not alt:
        return True
    if canonical_ref(alt) not in canon_allowed:
        return False
    if canonical_ref(alt) == canonical_ref(decision.selected_ref):
        return False
    return True


def arbitrate_among_candidates(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    decision_frame: DecisionFrame | dict[str, Any] | None = None,
    call_type: ArbiterCallType = "v5_arbiter",
) -> tuple[ArbiterDecision | None, ArbiterRunStatus, str | None]:
    """
    Один LLM-вызов Arbiter. При ошибке/таймауте/невалидных ref — fallback на max score.

    Returns (decision, status, error_message_or_none). decision is None only when status is skipped.
    """
    q = (question or "").strip()
    cands = [c for c in candidates if isinstance(c, dict) and str(c.get("ref") or "").strip()]
    distinct = {canonical_ref(str(c.get("ref") or "")) for c in cands}
    distinct.discard(canonical_ref(""))

    if len(distinct) < 2 or not q:
        return None, "skipped", "less_than_two_distinct_refs_or_empty_question" if q else "empty_question"

    allowed_refs = {str(c["ref"]).strip() for c in cands if str(c.get("ref") or "").strip()}

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            max_completion_tokens=350,
            response_format={"type": "json_object"},
            timeout=_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
                {"role": "user", "content": _arbiter_user_payload(question=q, candidates=cands, decision_frame=decision_frame)},
            ],
        )
        log_llm_usage(logger, resp, call_type=call_type, model=_MODEL)
        raw = (resp.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return _fallback_from_candidates(cands), "fallback", f"json_decode:{str(e)[:200]}"
        decision = ArbiterDecision.model_validate(obj)
        if not _validate_refs(decision, allowed_refs):
            return _fallback_from_candidates(cands), "fallback", "invalid_ref_not_in_candidates"
        return decision, "ok", None
    except ValidationError as e:
        try:
            logger.warning(
                "arbiter_validation_failed",
                extra={
                    "extra_data": {
                        "call_type": call_type,
                        "model": _MODEL,
                        "raw_output": (raw or "")[:2000],
                        "error": str(e)[:2000],
                    }
                },
            )
        except Exception:
            pass
        return _fallback_from_candidates(cands), "fallback", f"validation:{str(e)[:400]}"
    except Exception as e:
        log_llm_error(logger, call_type=call_type, err=str(e), model=_MODEL)
        return _fallback_from_candidates(cands), "fallback", str(e)[:500]
