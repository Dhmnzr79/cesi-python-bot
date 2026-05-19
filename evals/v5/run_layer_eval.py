from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Literal


LayerName = Literal["resolver", "arbiter", "verifier", "generator", "ingress", "all"]

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
    try:
        print(line)
    except UnicodeEncodeError:
        print(
            json.dumps(
                {"layer": res.layer, "status": res.status, "details": res.details},
                ensure_ascii=True,
            )
        )


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


def _gen_norm(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def _generator_faithfulness_violations(source_text: str, answer: str) -> list[str]:
    """Детерминированные high-risk сигналы: числа/деньги/%/сроки/абсолютные обещания вне source_text."""
    s0 = source_text or ""
    sn = _gen_norm(s0)
    an = _gen_norm(answer or "")
    bad: list[str] = []
    if ("₽" in (answer or "")) and ("₽" not in s0):
        bad.append("currency_ruble_sign")
    if re.search(r"\bруб", an) and not re.search(r"\bруб", sn):
        bad.append("currency_rub_word")
    if re.search(r"\bр\.\s*\d", an) and not re.search(r"\bр\.\s*\d", sn):
        bad.append("currency_rp_dot")
    # проценты, включая десятичные с запятой/точкой (99,8%, 12.5%)
    for m in re.finditer(r"\d+(?:[.,]\d+)?\s*%", answer or ""):
        frag_ans = m.group(0)
        norm_ans = frag_ans.replace(" ", "").replace(",", ".")
        norm_src = s0.replace(" ", "").replace(",", ".")
        if norm_ans not in norm_src:
            bad.append("percent:" + frag_ans)
    if re.search(r"\b100\s*%", an) and "100" not in sn:
        bad.append("hundred_percent")
    for m in re.finditer(r"\d{2,}", answer or ""):
        if m.group(0) not in s0:
            bad.append("number2+:" + m.group(0))
    # сроки: годы, месяцы, дни/сутки (разные падежи)
    duration_re = re.compile(
        r"\b\d{1,3}\s*(?:лет|года|год|месяц|месяца|месяцев|недел\w*|день|дня|дней|сутки|суток)\b",
        re.U,
    )
    for m in duration_re.finditer(an):
        t = re.sub(r"\s+", " ", m.group(0)).strip()
        if t.replace(" ", "") not in sn.replace(" ", ""):
            bad.append("duration:" + t)
    # абсолютные обещания безопасности/гарантий
    for w in (
        "абсолютно безопасно",
        "абсолютно безопасна",
        "гарантированно",
        "100% безопасно",
        "пожизненная гарантия",
    ):
        if w in an and w not in sn:
            bad.append("absolute:" + w)
    return bad


def eval_verifier() -> EvalResult:
    cases = _load_json(_here("verifier_golden.json"))
    try:
        from verifier import collect_high_risk_signals, verify_answer_structured
    except Exception as e:
        return EvalResult(
            layer="verifier",
            status="SKIP",
            details={"cases": len(cases), "reason": f"verifier_import_failed: {str(e)[:200]}"},
        )

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return EvalResult(
            layer="verifier",
            status="SKIP",
            details={"cases": len(cases), "reason": "no_openai_api_key"},
        )

    total = 0
    ok = 0
    bad: list[dict[str, Any]] = []

    for row in cases:
        cid = str(row.get("id") or "")
        src = str(row.get("source_snippet") or "")
        ans = str(row.get("answer") or "")
        exp = row.get("expected") or {}
        if not isinstance(exp, dict) or not ans.strip():
            continue
        want_g = exp.get("grounded")
        want_tr = exp.get("triggered")
        if want_g is None and want_tr is None:
            continue

        signals = collect_high_risk_signals(ans)
        triggered = len(signals) > 0
        if want_tr is not None and bool(want_tr) != triggered:
            total += 1
            bad.append(
                {
                    "id": cid,
                    "error": "trigger_mismatch",
                    "expected_triggered": bool(want_tr),
                    "got_triggered": triggered,
                    "signals": signals,
                }
            )
            continue

        if not triggered:
            total += 1
            if want_g is False:
                bad.append(
                    {
                        "id": cid,
                        "error": "expected_not_grounded_but_no_high_risk_trigger",
                        "answer_preview": ans[:120],
                    }
                )
            else:
                ok += 1
            continue

        total += 1
        ref = f"eval#{cid}"
        verdict, run_st, err = verify_answer_structured(answer=ans, source_snippet=src, source_ref=ref, call_type="v5_verifier_eval")
        if run_st != "ok" or verdict is None:
            bad.append(
                {
                    "id": cid,
                    "error": f"verifier_call:{run_st}",
                    "detail": err,
                    "answer_preview": ans[:120],
                }
            )
            continue

        if want_g is None:
            ok += 1
            continue

        g = bool(verdict.grounded)
        if g == bool(want_g):
            ok += 1
        else:
            bad.append(
                {
                    "id": cid,
                    "expected_grounded": bool(want_g),
                    "got_grounded": g,
                    "confidence": verdict.confidence,
                    "facts": verdict.hallucinated_facts[:5],
                }
            )

    if total == 0:
        return EvalResult(layer="verifier", status="SKIP", details={"cases": 0, "reason": "no_cases"})

    acc = ok / total
    status: Literal["OK", "FAIL"] = "OK" if acc >= 0.75 else "FAIL"
    return EvalResult(
        layer="verifier",
        status=status,
        details={
            "cases": total,
            "ok": ok,
            "accuracy": round(acc, 4),
            "accuracy_note": "trigger must match expected; if triggered, LLM grounded must match expected (shadow quality)",
            "bad_examples": bad[:25],
        },
    )


def eval_ingress() -> EvalResult:
    cases = _load_json(_here("ingress_golden.json"))
    try:
        from core.clinic_policies_loader import match_clinic_policy_key
        from ingress_gate import ingress_entity_offered
    except Exception as e:
        return EvalResult(
            layer="ingress",
            status="SKIP",
            details={"cases": len(cases), "reason": f"ingress_import_failed: {str(e)[:200]}"},
        )

    total = 0
    ok = 0
    bad: list[dict[str, Any]] = []
    client_id = "default"
    for row in cases:
        cid = str(row.get("id") or "")
        q = str(row.get("question") or "")
        mode = str(row.get("mode") or "policy_only")
        total += 1
        passed = False
        if mode == "policy_only":
            pk = match_clinic_policy_key(q, client_id)
            exp_route = str(row.get("expected_route") or "")
            exp_pk = str(row.get("expected_policy_key") or "")
            passed = pk == exp_pk and exp_route == "not_offered_policy"
            if not passed:
                bad.append({"id": cid, "got_policy_key": pk, "expected_policy_key": exp_pk})
        elif mode in ("catalog_check", "offered_check"):
            offered = ingress_entity_offered(q, client_id)
            want = bool(row.get("expected_offered"))
            passed = offered == want
            if not passed:
                bad.append({"id": cid, "got_offered": offered, "expected_offered": want})
        else:
            bad.append({"id": cid, "reason": f"unknown_mode:{mode}"})
            passed = False
        if passed:
            ok += 1

    if total == 0:
        return EvalResult(layer="ingress", status="SKIP", details={"cases": 0, "reason": "no_cases"})
    acc = ok / total
    status: Literal["OK", "FAIL"] = "OK" if acc >= 1.0 else "FAIL"
    return EvalResult(
        layer="ingress",
        status=status,
        details={
            "cases": total,
            "ok": ok,
            "accuracy": round(acc, 4),
            "note": "deterministic policy + catalog/doctor offered ground truth (no LLM)",
            "bad_examples": bad[:25],
        },
    )


def eval_generator() -> EvalResult:
    cases = _load_json(_here("generator_golden.json"))
    total = 0
    ok = 0
    bad: list[dict[str, Any]] = []
    for row in cases:
        cid = str(row.get("id") or "")
        src = str(row.get("source_text") or "")
        ans = str(row.get("answer") or "")
        exp = row.get("expected") or {}
        if not isinstance(exp, dict):
            continue
        want = exp.get("faithful")
        if want is None:
            continue
        total += 1
        v = _generator_faithfulness_violations(src, ans)
        passed = (len(v) == 0) if bool(want) else (len(v) > 0)
        if passed:
            ok += 1
        else:
            bad.append(
                {
                    "id": cid,
                    "faithful_expected": bool(want),
                    "violations": v,
                    "answer_preview": ans[:120],
                }
            )
    if total == 0:
        return EvalResult(layer="generator", status="SKIP", details={"cases": 0, "reason": "no_cases"})
    acc = ok / total
    status: Literal["OK", "FAIL"] = "OK" if acc >= 0.95 else "FAIL"
    return EvalResult(
        layer="generator",
        status=status,
        details={
            "cases": total,
            "ok": ok,
            "accuracy": round(acc, 4),
            "bad_examples": bad[:25],
        },
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layer",
        required=True,
        choices=["resolver", "arbiter", "verifier", "generator", "ingress", "all"],
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
    if layer in ("ingress", "all"):
        results.append(eval_ingress())

    for r in results:
        _print_result(r)

    if any(r.status == "FAIL" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

