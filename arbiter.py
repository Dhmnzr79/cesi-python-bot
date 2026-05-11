"""A5 Arbiter — выбор лучшего content-источника (structured ArbiterDecision).

PR #1.6: shadow-only в /ask; legacy `select_content_route` остаётся единственным маршрутом ответа.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import ValidationError

from contracts.arbiter_decision import ArbiterDecision
from contracts.decision_frame import DecisionFrame
from content_arbiter import ContentCandidates
from config import CHAT_MODEL
from llm import client
from retriever import get_chunk_by_ref
from logging_setup import get_logger, log_llm_error, log_llm_usage

logger = get_logger("bot")

_MODEL = (os.getenv("MODEL_ARBITER") or "").strip() or CHAT_MODEL
_TIMEOUT_SEC = float(os.getenv("V5_ARBITER_TIMEOUT_SEC", "12"))

ArbiterRunStatus = Literal["ok", "skipped", "error", "fallback"]
ArbiterCallType = Literal["v5_arbiter", "v5_arbiter_shadow"]


def is_arbiter_shadow_enabled() -> bool:
    """Телеметрия shadow Arbiter в /ask. По умолчанию включено; V5_ARBITER_SHADOW_ON=0 — без LLM."""
    return (os.getenv("V5_ARBITER_SHADOW_ON") or "1").strip().lower() in ("1", "true", "yes")


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


def compute_agrees_with_legacy(arbiter_ref: str | None, legacy_ref: str | None) -> bool | None:
    if not arbiter_ref or not legacy_ref:
        return None
    return canonical_ref(arbiter_ref) == canonical_ref(legacy_ref)


def legacy_content_ref_from_route(
    *,
    sel_route: str,
    sel_chunk: dict | None,
    catalog_snapshot: dict[str, Any] | None,
) -> str | None:
    """Определить ref выбранного legacy-маршрута (если применимо)."""
    if sel_route == "catalog_md_first" and isinstance(catalog_snapshot, dict):
        return with_default_anchor(str(catalog_snapshot.get("md_entry_ref") or "")) or None
    if sel_route == "retrieval_chunk" and isinstance(sel_chunk, dict):
        return ref_from_chunk(sel_chunk)
    return None
