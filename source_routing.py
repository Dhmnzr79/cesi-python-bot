"""A3 deterministic source routing (post-Resolver, pre-retrieval)."""
from __future__ import annotations

import re
from typing import Any

from contracts.decision_frame import DecisionFrame
from contracts.source_route_result import SourceRouteResult, SourceType

from core.routing_loader import THRESHOLDS
from doctors_lookup import doctor_intent_probe, doctor_name_probe, doctors_lookup
from query_selector import (
    catalog_service_session_context,
    match_service_from_catalog,
    price_rules_hint,
    select_price_service_route,
)

# Client default for price_concern without service match (see service_catalog concern_ref pattern).
DEFAULT_PRICE_CONCERN_REF = "implantation__faq__cost.md#korotko"


def _short_followup(q: str, *, max_tokens: int = 8) -> bool:
    qn = re.sub(r"\s+", " ", (q or "").strip(), flags=re.U)
    toks = [t for t in qn.split(" ") if t.strip()]
    return 1 <= len(toks) <= max_tokens


def _with_korotko_anchor(md_entry_ref: str) -> str:
    r = (md_entry_ref or "").strip()
    if not r:
        return r
    if "#" not in r:
        if r.lower().endswith(".md"):
            return f"{r}#korotko"
        return f"{r}.md#korotko"
    return r


def _facts_nonempty(service: dict[str, Any]) -> list[str]:
    return [str(x).strip() for x in (service.get("facts") or []) if str(x).strip()]


def _resolve_route_intent(*, q: str, decision: DecisionFrame | None, app_intent: str) -> str:
    hint = price_rules_hint(q)
    if hint:
        return hint
    if decision is not None:
        return str(decision.route_intent or "unknown").strip().lower()
    return str(app_intent or "content").strip().lower()


def _source_type_from_price_route(pr: dict[str, Any]) -> SourceType:
    if str(pr.get("intent") or "") == "price_concern":
        return "price_concern"
    rs = str(pr.get("route_source") or "")
    if rs == "prices_json":
        return "price_card"
    return "price_ref"


def _match_method_from_price_route(pr: dict[str, Any]) -> MatchMethod:
    if str(pr.get("fallback_reason") or "") == "context_session":
        return "session_fallback"
    return "catalog_containment"


def _price_route_to_source_result(pr: dict[str, Any]) -> SourceRouteResult:
    return SourceRouteResult(
        source=_source_type_from_price_route(pr),
        service_id=str(pr.get("matched_service_id") or "") or None,
        ref=str(pr.get("price_ref") or "").strip() or None,
        concern_ref=None,
        payload={"price_route": pr},
        match_score=float(pr.get("match_score") or 0.0),
        match_method=_match_method_from_price_route(pr),
    )


def route_source(
    q: str,
    *,
    sid: str,
    client_id: str,
    decision: DecisionFrame | None,
    app_intent: str,
) -> SourceRouteResult:
    """Run A3 routing. Caller handles `source == none` via legacy branches."""
    ri = _resolve_route_intent(q=q, decision=decision, app_intent=app_intent)
    q0 = (q or "").strip()

    doctors_gate = doctor_name_probe(q0, client_id=client_id) or doctor_intent_probe(q0)
    if doctors_gate and ri not in ("price_lookup", "price_concern"):
        hit = doctors_lookup(q0, client_id=client_id)
        if hit:
            routing = str(hit.get("routing") or "doc")
            if routing == "cards":
                return SourceRouteResult(
                    source="doctor",
                    service_id=None,
                    ref=None,
                    concern_ref=None,
                    payload={"doctor": hit},
                    match_score=1.0,
                    match_method="doctors_lookup",
                )
            did = str(hit.get("doc_id") or "").strip().removesuffix(".md")
            ref = _with_korotko_anchor(f"{did}.md")
            return SourceRouteResult(
                source="doctor",
                service_id=None,
                ref=ref,
                concern_ref=None,
                payload={"doctor": hit},
                match_score=1.0,
                match_method="doctors_lookup",
            )

    match = match_service_from_catalog(q0, client_id=client_id)
    score = float(match.get("match_score") or 0.0)
    contain = score >= float(THRESHOLDS.catalog_match.containment_min)
    mid = match.get("matched_service_id")
    svc = match.get("service") if isinstance(match.get("service"), dict) else {}
    svc = dict(svc)

    if contain and ri == "content":
        facts = _facts_nonempty(svc)
        if facts:
            return SourceRouteResult(
                source="catalog_facts",
                service_id=str(mid or "") or None,
                ref=None,
                concern_ref=None,
                payload={
                    "service": svc,
                    "matched_service_id": str(mid or "") or None,
                    "facts": facts,
                },
                match_score=score,
                match_method="catalog_containment",
            )
        md_raw = str(svc.get("md_entry_ref") or "").strip()
        if md_raw:
            return SourceRouteResult(
                source="catalog_md",
                service_id=str(mid or "") or None,
                ref=_with_korotko_anchor(md_raw),
                concern_ref=None,
                payload={"service": svc},
                match_score=score,
                match_method="catalog_containment",
            )

    if ri == "content" and (not contain) and _short_followup(q0):
        ctx = catalog_service_session_context(sid, client_id)
        if ctx:
            s_ctx = ctx.get("service") if isinstance(ctx.get("service"), dict) else {}
            md_raw = str(s_ctx.get("md_entry_ref") or "").strip()
            if md_raw:
                sid_svc = str(ctx.get("service_id") or "").strip()
                return SourceRouteResult(
                    source="catalog_md",
                    service_id=sid_svc or None,
                    ref=_with_korotko_anchor(md_raw),
                    concern_ref=None,
                    payload={"service": s_ctx},
                    match_score=float(THRESHOLDS.catalog_match.containment_min),
                    match_method="session_fallback",
                )

    if contain and ri == "price_concern":
        cref = str(svc.get("concern_ref") or "").strip()
        if cref:
            rref = _with_korotko_anchor(cref)
            return SourceRouteResult(
                source="price_concern",
                service_id=str(mid or "") or None,
                ref=rref,
                concern_ref=rref,
                payload=None,
                match_score=score,
                match_method="catalog_containment",
            )

    if ri == "price_concern":
        ctx = catalog_service_session_context(sid, client_id)
        if ctx:
            s2 = ctx.get("service") if isinstance(ctx.get("service"), dict) else {}
            cref = str((s2 or {}).get("concern_ref") or "").strip()
            if cref:
                rref = _with_korotko_anchor(cref)
                return SourceRouteResult(
                    source="price_concern",
                    service_id=str(ctx.get("service_id") or "") or None,
                    ref=rref,
                    concern_ref=rref,
                    payload=None,
                    match_score=1.0,
                    match_method="session_fallback",
                )
        rref = _with_korotko_anchor(DEFAULT_PRICE_CONCERN_REF)
        return SourceRouteResult(
            source="price_concern",
            service_id=None,
            ref=rref,
            concern_ref=rref,
            payload=None,
            match_score=1.0,
            match_method="concern_default",
        )

    if ri == "price_lookup":
        pr = select_price_service_route(q0, client_id=client_id, sid=sid, intent_override="price_lookup")
        if pr.get("mode") == "matched":
            return _price_route_to_source_result(pr)
        return SourceRouteResult(
            source="price_lookup_clarify",
            service_id=None,
            ref=None,
            concern_ref=None,
            payload={"price_route": pr},
            match_score=0.0,
            match_method="none",
        )

    return SourceRouteResult(
        source="none",
        service_id=None,
        ref=None,
        concern_ref=None,
        payload=None,
        match_score=0.0,
        match_method="none",
    )


def slim_source_route_payload(sr: SourceRouteResult) -> dict[str, Any]:
    """Dashboard-safe summary (no payload bodies)."""
    return {
        "source": sr.source,
        "ref": sr.ref,
        "service_id": sr.service_id,
        "concern_ref": sr.concern_ref,
        "match_method": sr.match_method,
        "match_score": sr.match_score,
    }
