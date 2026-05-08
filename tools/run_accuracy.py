#!/usr/bin/env python3
"""
Launch-accuracy runner: POST /ask по кейсам из JSON (одиночные и многоходовые).

Пример:
  python tools/run_accuracy.py --base-url http://127.0.0.1:9000

Кейсы по умолчанию: evals/accuracy_launch.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import DEFAULT_CLIENT_ID  # noqa: E402


def normalize_text(s: str) -> str:
    t = (s or "").strip().lower().replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def check_text(answer: str, must_contain: list[str] | None, must_not_contain: list[str] | None) -> list[str]:
    reasons: list[str] = []
    a = normalize_text(answer or "")
    for sub in must_contain or []:
        if not sub:
            continue
        if normalize_text(sub) not in a:
            reasons.append(f"must_contain_missing:{sub!r}")
    for sub in must_not_contain or []:
        if not sub:
            continue
        if normalize_text(sub) in a:
            reasons.append(f"must_not_contain_present:{sub!r}")
    return reasons


def _actual_doc_id(meta: dict[str, Any]) -> str | None:
    did = meta.get("doc_id")
    if isinstance(did, str) and did.strip():
        return did.strip()
    f = meta.get("file")
    if isinstance(f, str) and f.strip():
        return os.path.splitext(os.path.basename(f.strip()))[0] or None
    return None


def extract_actual(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    route = meta.get("route")
    if route is None:
        route = meta.get("selected_route")
    return {
        "answer": payload.get("answer"),
        "route": route,
        "doc_id": _actual_doc_id(meta),
        "h3_id": meta.get("h3_id"),
        "intent": meta.get("intent"),
        "matched_service_id": meta.get("matched_service_id"),
        "price_key": meta.get("price_key"),
        "lead_step": meta.get("lead_step"),
        "lead_flow": bool(meta.get("lead_flow")),
        "fallback_reason": meta.get("fallback_reason"),
        "cta_present": payload.get("cta") is not None,
        "quick_replies_count": len(payload.get("quick_replies") or []),
    }


def _eq_or_in(actual: Any, expected: Any, expected_any: list[Any] | None, field: str) -> list[str]:
    if expected_any is not None:
        ok = False
        for x in expected_any:
            if x is None and actual is None:
                ok = True
                break
            if x is not None and actual is not None and str(actual).strip() == str(x).strip():
                ok = True
                break
        if not ok:
            return [f"{field}_any_mismatch"]
        return []
    if expected is None and field not in ("route",):  # caller handles route separately
        return []
    if expected is not None:
        if actual is None and expected is None:
            return []
        if actual is None:
            return [f"{field}_missing"]
        if str(actual).strip() != str(expected).strip():
            return [f"{field}_mismatch"]
    return []


def matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Возвращает (fail_reasons, warnings)."""
    fails: list[str] = []
    warns: list[str] = []

    exp = expected or {}

    # route
    if exp.get("route") is not None:
        ar = actual.get("route")
        if ar is None:
            fails.append("expected_route_missing_in_payload")
        elif str(ar).strip() != str(exp["route"]).strip():
            fails.append("expected_route_mismatch")
    elif exp.get("route_any"):
        ar = actual.get("route")
        if ar is None:
            warns.append("route_not_exposed")
        else:
            allowed = {str(x).strip() for x in (exp.get("route_any") or []) if x is not None}
            if str(ar).strip() not in allowed:
                fails.append("route_any_mismatch")

    # doc_id
    if exp.get("doc_id") is not None:
        fails.extend(_eq_or_in(actual.get("doc_id"), exp.get("doc_id"), None, "doc_id"))
    elif exp.get("doc_id_any") is not None:
        ad = actual.get("doc_id")
        allowed_raw = list(exp.get("doc_id_any") or [])
        allowed: set[str | None] = set()
        has_null = False
        for x in allowed_raw:
            if x is None:
                has_null = True
            else:
                allowed.add(str(x).strip())
        ok = False
        if ad is None or (isinstance(ad, str) and not ad.strip()):
            if has_null:
                ok = True
        elif isinstance(ad, str) and ad.strip() in allowed:
            ok = True
        if not ok:
            fails.append("doc_id_any_mismatch")

    if exp.get("h3_id") is not None:
        fails.extend(_eq_or_in(actual.get("h3_id"), exp.get("h3_id"), None, "h3_id"))
    elif exp.get("h3_id_any") is not None:
        ah3 = actual.get("h3_id")
        allowed_raw = list(exp.get("h3_id_any") or [])
        allowed: set[str] = set()
        has_null = False
        for x in allowed_raw:
            if x is None:
                has_null = True
            else:
                allowed.add(str(x).strip())
        ok = False
        if ah3 is None or (isinstance(ah3, str) and not ah3.strip()):
            if has_null:
                ok = True
        elif isinstance(ah3, str) and ah3.strip() in allowed:
            ok = True
        if not ok:
            fails.append("h3_id_any_mismatch")

    if exp.get("intent") is not None:
        fails.extend(_eq_or_in(actual.get("intent"), exp.get("intent"), None, "intent"))

    if "matched_service_id" in exp:
        ev = exp["matched_service_id"]
        av = actual.get("matched_service_id")
        if ev is None:
            if av is not None and str(av).strip():
                fails.append("matched_service_id_expected_null")
        else:
            fails.extend(_eq_or_in(av, ev, None, "matched_service_id"))

    if exp.get("price_key") is not None:
        fails.extend(_eq_or_in(actual.get("price_key"), exp.get("price_key"), None, "price_key"))

    if exp.get("lead_step") is not None:
        fails.extend(_eq_or_in(actual.get("lead_step"), exp.get("lead_step"), None, "lead_step"))

    if exp.get("fallback_reason") is not None:
        fails.extend(_eq_or_in(actual.get("fallback_reason"), exp.get("fallback_reason"), None, "fallback_reason"))
    elif exp.get("fallback_reason_any"):
        fr = actual.get("fallback_reason")
        allowed = {str(x).strip() for x in (exp.get("fallback_reason_any") or [])}
        if fr is None or str(fr).strip() not in allowed:
            fails.append("fallback_reason_any_mismatch")

    return fails, warns


def _check_ux(payload: dict[str, Any], actual: dict[str, Any], ux: dict[str, Any] | None) -> list[str]:
    if not ux:
        return []
    reasons: list[str] = []
    if ux.get("expect_lead_flow") is True and not actual.get("lead_flow"):
        reasons.append("ux_expect_lead_flow")
    if ux.get("expect_cta") is True and not actual.get("cta_present"):
        reasons.append("ux_expect_cta")
    if ux.get("expect_cta") is False and actual.get("cta_present"):
        reasons.append("ux_expect_no_cta")
    return reasons


def _post_ask(base_url: str, client_id: str, sid: str, q: str, timeout_sec: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/ask"
    body = json.dumps({"q": q, "client_id": client_id, "sid": sid}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError("cases file must be a list or {\"cases\": [...]}")
    if not isinstance(cases, list):
        raise ValueError("`cases` must be a list")
    return [c for c in cases if isinstance(c, dict) and not c.get("skip")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch accuracy eval via POST /ask")
    ap.add_argument("--file", default=os.path.join(_ROOT, "evals", "accuracy_launch.json"))
    ap.add_argument("--base-url", default=os.environ.get("ACC_BASE_URL", "http://127.0.0.1:9000"))
    ap.add_argument("--client-id", default=os.environ.get("ACC_CLIENT_ID", DEFAULT_CLIENT_ID))
    ap.add_argument("--group", default=None, help="Только эта group")
    ap.add_argument("--id", dest="case_id", default=None, help="Только кейс с этим id")
    ap.add_argument("--fail-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-snapshot", action="store_true", help="Не писать JSON snapshot")
    args = ap.parse_args()

    import urllib.error

    cases_path = os.path.abspath(args.file)
    cases = _load_cases(cases_path)
    if args.group:
        cases = [c for c in cases if str(c.get("group") or "") == args.group]
    if args.case_id:
        cases = [c for c in cases if str(c.get("id") or "") == args.case_id]

    snap_dir = os.path.join(_ROOT, "evals", "snapshots")
    if not args.no_snapshot:
        os.makedirs(snap_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_path = os.path.join(snap_dir, f"{ts}_accuracy_launch.json") if not args.no_snapshot else None

    meta_run = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": cases_path,
        "base_url": args.base_url,
        "client_id": args.client_id,
        "case_filter": {"group": args.group, "id": args.case_id},
    }
    snap: dict[str, Any] = {"run": meta_run, "results": []}
    fails_detail: list[dict[str, Any]] = []

    for case in cases:
        cid = str(case.get("id") or "?").strip()
        grp = str(case.get("group") or "ungrouped").strip() or "ungrouped"
        turns = case.get("turns")
        turn_specs: list[dict[str, Any]]
        if isinstance(turns, list) and turns:
            turn_specs = [t for t in turns if isinstance(t, dict)]
        else:
            q0 = (case.get("q") or "").strip()
            turn_specs = [{"q": q0, "expected": case.get("expected"), "must_contain": case.get("must_contain"), "must_not_contain": case.get("must_not_contain"), "ux": case.get("ux")}]

        sid = uuid.uuid4().hex
        case_entry: dict[str, Any] = {"id": cid, "group": grp, "sid": sid, "turns": []}
        case_ok = True
        case_warns = 0

        for ti, spec in enumerate(turn_specs):
            q = (spec.get("q") or "").strip()
            exp = spec.get("expected")
            if exp is None and ti == 0 and not isinstance(turns, list):
                exp = case.get("expected")
            exp = exp if isinstance(exp, dict) else {}

            must_c = spec.get("must_contain")
            if must_c is None and ti == 0:
                must_c = case.get("must_contain")
            must_nc = spec.get("must_not_contain")
            if must_nc is None and ti == 0:
                must_nc = case.get("must_not_contain")

            ux = spec.get("ux") if spec.get("ux") is not None else case.get("ux")

            turn_rec: dict[str, Any] = {"index": ti, "q": q, "request": {"q": q, "sid": sid, "client_id": args.client_id}}
            if not q:
                reasons = ["empty_question"]
                turn_rec["error"] = reasons
                turn_rec["pass"] = False
                case_ok = False
                case_entry["turns"].append(turn_rec)
                continue

            try:
                payload = _post_ask(args.base_url, args.client_id, sid, q, args.timeout)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:800]
                except Exception:
                    pass
                turn_rec["error"] = f"HTTP {e.code}: {body}"
                turn_rec["pass"] = False
                case_ok = False
                case_entry["turns"].append(turn_rec)
                continue
            except Exception as e:
                turn_rec["error"] = str(e)
                turn_rec["pass"] = False
                case_ok = False
                case_entry["turns"].append(turn_rec)
                continue

            actual = extract_actual(payload)
            mfails, mwarns = matches_expected(actual, exp)
            tfails = check_text(str(actual.get("answer") or ""), must_c, must_nc)
            forb = case.get("forbidden_doc_ids") or []
            adoc = actual.get("doc_id")
            if isinstance(forb, list) and adoc:
                if adoc in forb:
                    mfails.append("forbidden_doc_id")
            ux_fails = _check_ux(payload, actual, ux if isinstance(ux, dict) else None)

            reasons = mfails + tfails + ux_fails
            turn_rec["response"] = payload
            turn_rec["actual"] = actual
            turn_rec["warnings"] = mwarns
            turn_rec["fail_reasons"] = reasons
            turn_rec["pass"] = len(reasons) == 0
            if not turn_rec["pass"]:
                case_ok = False
            case_warns += len(mwarns)
            case_entry["turns"].append(turn_rec)

        case_entry["pass"] = case_ok
        case_entry["warnings_count"] = case_warns
        snap["results"].append(case_entry)

        if not case_ok:
            last_turn = case_entry["turns"][-1] if case_entry["turns"] else {}
            fails_detail.append(
                {
                    "id": cid,
                    "group": grp,
                    "last_q": last_turn.get("q"),
                    "fail_reasons": last_turn.get("fail_reasons")
                    or (["HTTP/error"] if last_turn.get("error") else []),
                    "warnings": last_turn.get("warnings"),
                    "actual": last_turn.get("actual"),
                    "answer_preview": str((last_turn.get("response") or {}).get("answer") or "")[:400],
                }
            )

    group_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0})
    for case_entry in snap["results"]:
        grp = str(case_entry.get("group") or "ungrouped")
        if case_entry.get("pass"):
            group_stats[grp]["pass"] += 1
        else:
            group_stats[grp]["fail"] += 1
        group_stats[grp]["warn"] += int(case_entry.get("warnings_count") or 0)

    # print table
    rows = []
    total_p = total_f = total_w = 0
    for grp in sorted(group_stats.keys()):
        st = group_stats[grp]
        p, f, w = st["pass"], st["fail"], st["warn"]
        total_p += p
        total_f += f
        total_w += w
        tot = p + f
        pct = 100 if tot == 0 else int(round(100.0 * p / tot))
        rows.append((grp, p, f, tot, pct))
    tot_all = total_p + total_f
    pct_all = 100 if tot_all == 0 else int(round(100.0 * total_p / tot_all))

    if not args.fail_only:
        print(f"FILE {cases_path}")
        print(f"BASE {args.base_url}")
        print()
        hdr = f"{'GROUP':<22} {'PASS':>4} {'FAIL':>4} {'TOTAL':>5} {'%':>4}"
        print(hdr)
        print("-" * len(hdr))
        for grp, p, f, tot, pct in rows:
            print(f"{grp:<22} {p:>4} {f:>4} {tot:>5} {pct:>4}")
        print("-" * len(hdr))
        print(f"{'TOTAL':<22} {total_p:>4} {total_f:>4} {tot_all:>5} {pct_all:>4}")
        print()
    else:
        print(f"SUMMARY pass={total_p} fail={total_f} total={tot_all} pct={pct_all}")

    if fails_detail and not args.fail_only:
        print("FAIL CASES")
        print("-" * 60)
    for fd in fails_detail:
        if args.fail_only:
            print(f"FAIL {fd['id']}")
        else:
            print(f"FAIL {fd['id']} [{fd['group']}]")
        print(f"Q: {fd['last_q']}")
        print(f"Reasons: {', '.join(fd['fail_reasons'] or [])}")
        if fd.get("warnings"):
            print(f"Warnings: {', '.join(fd['warnings'])}")
        print(f"Actual: {fd.get('actual')}")
        if fd.get("answer_preview"):
            print(f"Answer: {fd['answer_preview']}")
        print()

    if snap_path:
        with open(snap_path, "w", encoding="utf-8") as out:
            json.dump(snap, out, ensure_ascii=False, indent=2)
        if not args.fail_only:
            print(f"Snapshot: {snap_path}")

    return 1 if total_f else 0


if __name__ == "__main__":
    raise SystemExit(main())
