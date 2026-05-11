from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal


LayerName = Literal["resolver", "arbiter", "verifier", "generator", "all"]

# Ensure project root is importable when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass(frozen=True)
class EvalResult:
    layer: str
    status: Literal["OK", "FAIL", "SKIP"]
    details: dict[str, Any]


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"golden set must be a JSON array: {path}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"golden row #{i} must be an object: {path}")
        out.append(row)
    return out


def _print_result(res: EvalResult) -> None:
    line = json.dumps(
        {"layer": res.layer, "status": res.status, "details": res.details},
        ensure_ascii=False,
    )
    print(line)


def _norm_expected(v: Any) -> str | None:
    if v is None:
        return None
    return str(v).strip().lower() or None


def eval_resolver() -> EvalResult:
    cases = _load_json(_here("resolver_golden.json"))
    try:
        # PR #1.2.6: same LLM path + RESOLVER_SYSTEM_PROMPT as /ask (not shadow-only copy).
        from resolver import RESOLVER_SYSTEM_PROMPT, resolve_decision_frame

        assert "Пример формата" in RESOLVER_SYSTEM_PROMPT and "route_intent" in RESOLVER_SYSTEM_PROMPT
    except Exception as e:
        return EvalResult(
            layer="resolver",
            status="SKIP",
            details={"cases": len(cases), "reason": f"resolver_import_failed: {str(e)[:200]}"},
        )

    total = 0
    ok = 0
    bad: list[dict[str, Any]] = []

    for row in cases:
        cid = str(row.get("id") or "")
        q = str(row.get("question") or "")
        exp = row.get("expected") or {}
        if not isinstance(exp, dict) or not q.strip():
            continue
        total += 1
        try:
            df = resolve_decision_frame(question=q, history=[])
        except Exception as e:
            bad.append({"id": cid, "error": f"call_failed: {str(e)[:200]}"})
            continue

        want_intent = _norm_expected(exp.get("route_intent"))
        want_topic = _norm_expected(exp.get("service_topic"))
        want_mode = _norm_expected(exp.get("query_mode"))

        got_intent = _norm_expected(getattr(df, "route_intent", None))
        got_topic = _norm_expected(getattr(df, "service_topic", None))
        got_mode = _norm_expected(getattr(df, "query_mode", None))

        passed = True
        if want_intent and got_intent != want_intent:
            passed = False
        if want_topic and got_topic != want_topic:
            passed = False
        if want_mode and got_mode != want_mode:
            passed = False

        if passed:
            ok += 1
        else:
            bad.append(
                {
                    "id": cid,
                    "question": q[:120],
                    "expected": {"route_intent": want_intent, "service_topic": want_topic, "query_mode": want_mode},
                    "got": {"route_intent": got_intent, "service_topic": got_topic, "query_mode": got_mode},
                }
            )

    if total == 0:
        return EvalResult(layer="resolver", status="SKIP", details={"cases": 0, "reason": "no_cases"})

    acc = ok / total
    status: Literal["OK", "FAIL"] = "OK" if acc >= 0.9 else "FAIL"
    return EvalResult(
        layer="resolver",
        status=status,
        details={
            "cases": total,
            "ok": ok,
            "accuracy": round(acc, 4),
            "bad_examples": bad[:10],
        },
    )


def eval_arbiter() -> EvalResult:
    cases = _load_json(_here("arbiter_golden.json"))
    try:
        from arbiter import arbitrate_among_candidates, canonical_ref
    except Exception as e:
        return EvalResult(
            layer="arbiter",
            status="SKIP",
            details={"cases": len(cases), "reason": f"arbiter_import_failed: {str(e)[:200]}"},
        )

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return EvalResult(
            layer="arbiter",
            status="SKIP",
            details={"cases": len(cases), "reason": "no_openai_api_key"},
        )

    total = 0
    strict_ok = 0
    status_counts: dict[str, int] = {"ok": 0, "fallback": 0, "skipped": 0, "error": 0}
    bad: list[dict[str, Any]] = []

    for row in cases:
        cid = str(row.get("id") or "")
        q = str(row.get("question") or "")
        exp = row.get("expected") or {}
        raw_cands = row.get("candidates")
        if not isinstance(exp, dict) or not q.strip() or not isinstance(raw_cands, list):
            continue
        want_ref = canonical_ref(str(exp.get("selected_ref") or ""))
        if not want_ref or want_ref == canonical_ref(""):
            continue

        cands: list[dict[str, Any]] = []
        for c in raw_cands:
            if not isinstance(c, dict):
                continue
            ref = str(c.get("ref") or "").strip()
            if not ref:
                continue
            entry: dict[str, Any] = {
                "ref": ref,
                "source_kind": str(c.get("source_kind") or "eval_golden"),
                "doc_type": c.get("doc_type"),
                "subtype": c.get("subtype"),
                "topic": c.get("topic"),
                "service_id": c.get("service_id"),
                "snippet": c.get("snippet"),
                "why": c.get("why"),
            }
            if "score" in c and c.get("score") is not None:
                entry["score"] = c.get("score")
            else:
                entry["score"] = 0.5
            cands.append(entry)
        distinct = {canonical_ref(str(c.get("ref") or "")) for c in cands if str(c.get("ref") or "").strip()}
        if len(distinct) < 2:
            bad.append({"id": cid, "error": "golden_needs_two_distinct_refs", "question": q[:120]})
            continue

        total += 1
        try:
            decision, run_status, err = arbitrate_among_candidates(
                question=q,
                candidates=cands,
                decision_frame=None,
                call_type="v5_arbiter",
            )
        except Exception as e:
            status_counts["error"] += 1
            bad.append({"id": cid, "error": f"call_failed: {str(e)[:200]}", "question": q[:120]})
            continue

        if decision is None:
            rs = str(run_status or "skipped")
            if rs in status_counts:
                status_counts[rs] += 1
            else:
                status_counts["skipped"] += 1
            bad.append({"id": cid, "error": f"no_decision:{run_status}", "question": q[:120]})
            continue

        rs = str(run_status or "")
        if rs in status_counts:
            status_counts[rs] += 1
        else:
            status_counts["error"] += 1

        got_ref = canonical_ref(decision.selected_ref)
        ref_match = got_ref == want_ref
        strict_pass = rs == "ok" and ref_match
        if strict_pass:
            strict_ok += 1
        else:
            bad.append(
                {
                    "id": cid,
                    "question": q[:120],
                    "expected": want_ref,
                    "got": got_ref,
                    "run_status": rs,
                    "ref_match": ref_match,
                    "reason": (decision.reason or "")[:200],
                    "error": err,
                }
            )

    if total == 0:
        return EvalResult(layer="arbiter", status="SKIP", details={"cases": 0, "reason": "no_cases"})

    acc = strict_ok / total
    status: Literal["OK", "FAIL"] = "OK" if acc >= 0.85 else "FAIL"
    return EvalResult(
        layer="arbiter",
        status=status,
        details={
            "cases": total,
            "strict_ok": strict_ok,
            "accuracy": round(acc, 4),
            "accuracy_note": "PASS only if run_status==ok and selected_ref matches; fallback/skipped/error are never passes",
            "status_counts": dict(status_counts),
            "bad_examples": bad[:25],
        },
    )


def eval_verifier() -> EvalResult:
    cases = _load_json(_here("verifier_golden.json"))
    return EvalResult(
        layer="verifier",
        status="SKIP",
        details={"cases": len(cases), "reason": "verifier_not_implemented_yet"},
    )


def eval_generator() -> EvalResult:
    cases = _load_json(_here("generator_golden.json"))
    return EvalResult(
        layer="generator",
        status="SKIP",
        details={"cases": len(cases), "reason": "generator_eval_not_implemented_yet"},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layer",
        required=True,
        choices=["resolver", "arbiter", "verifier", "generator", "all"],
        help="Which layer eval to run.",
    )
    args = p.parse_args(argv)
    layer: LayerName = args.layer

    results: list[EvalResult] = []
    if layer in ("resolver", "all"):
        results.append(eval_resolver())
    if layer in ("arbiter", "all"):
        results.append(eval_arbiter())
    if layer in ("verifier", "all"):
        results.append(eval_verifier())
    if layer in ("generator", "all"):
        results.append(eval_generator())

    for r in results:
        _print_result(r)

    # Do not fail Phase 0 because layers aren't implemented yet.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

