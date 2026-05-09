from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import urllib.request
import urllib.error


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str  # PASS | FAIL | ERROR
    reason: str
    coverage_class: str = "UNKNOWN"


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("root must be an object")
    return obj


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _contains_ci(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _http_post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise ValueError("response is not a JSON object")
    return out


def _post_ask_json(bot_url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    """POST /ask JSON: HTTP (default) или Flask test_client при E2E_USE_TEST_CLIENT=1."""
    if (os.getenv("E2E_USE_TEST_CLIENT") or "").strip().lower() in {"1", "true", "yes"}:
        # Репозиторий в PYTHONPATH может отсутствовать при запуске как evals/v5/run_e2e_smoke.py
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from app import app  # локальный импорт — тяжёлый модуль только для in-proc smoke

        _ = bot_url  # URL игнорируется
        _ = timeout_sec
        client = app.test_client()
        resp = client.post("/ask", json=payload)
        out = resp.get_json()
        if not isinstance(out, dict):
            raise ValueError("response is not a JSON object")
        return out
    return _http_post_json(bot_url, payload, timeout_sec=timeout_sec)


def _infer_route_from_response(resp: dict[str, Any]) -> str:
    meta = resp.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    quick_replies = resp.get("quick_replies") or []

    # Explicit meta flags first.
    if bool(meta.get("handoff_filter")):
        return "handoff_filter"
    if bool(meta.get("lead_flow")) or bool(meta.get("booking_intent")):
        return "lead_flow"
    if bool(meta.get("low_score")):
        return "low_score_fallback"
    if str(meta.get("error") or "") == "rate_limited":
        return "rate_limited"

    intent = str(meta.get("intent") or "").strip().lower()
    if intent in {"price_lookup", "price_concern"}:
        return intent
    if intent == "offtopic":
        return "offtopic"
    if intent == "catalog_facts":
        return "catalog_facts"

    file = str(meta.get("file") or "").strip()
    if file == "clinic__info__contacts.md":
        return "contacts_chunk"
    if "__pricing__" in file:
        return "price_lookup"
    if file:
        return "retrieval_chunk"

    # Guided typically has quick replies but no file.
    if isinstance(quick_replies, list) and len(quick_replies) > 0:
        return "guided"

    return ""


def _print_table(rows: list[CaseResult]) -> None:
    w_id = max(10, max((len(r.case_id) for r in rows), default=10))
    w_status = 6
    w_reason = max(20, min(80, max((len(r.reason) for r in rows), default=20)))

    def line(a: str, b: str, c: str) -> str:
        return f"| {a:<{w_id}} | {b:<{w_status}} | {c:<{w_reason}} |"

    sep = f"+-{'-' * w_id}-+-{'-' * w_status}-+-{'-' * w_reason}-+"

    print(sep)
    print(line("id", "status", "reason (если fail)"))
    print(sep)
    for r in rows:
        reason = r.reason[:w_reason]
        print(line(r.case_id, r.status, reason))
    print(sep)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    path = os.getenv("E2E_SMOKE_PATH") or _here("e2e_smoke.json")
    bot_url = (os.getenv("BOT_URL") or "http://localhost:5000/ask").strip()
    timeout_sec = float(os.getenv("BOT_TIMEOUT_SEC") or "20")

    spec = _load_json(path)
    baseline = spec.get("baseline")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty array")

    if baseline is not None and not isinstance(baseline, int):
        raise ValueError("baseline must be null or int")

    results: list[CaseResult] = []
    passed = 0
    failed = 0
    errors = 0

    ts = int(time.time())

    for row in cases:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id") or "").strip() or f"case_{uuid.uuid4().hex[:8]}"
        history = row.get("history") or []
        question = str(row.get("question") or "")
        expected_route = row.get("expected_route")
        expected_route_any = row.get("expected_route_any")
        must_contain = row.get("must_contain") or []
        must_not_contain = row.get("must_not_contain") or []

        if expected_route is not None:
            expected_route = str(expected_route).strip()
        if expected_route_any is not None:
            if isinstance(expected_route_any, list) and all(isinstance(x, str) for x in expected_route_any):
                expected_route_any = [str(x).strip() for x in expected_route_any if str(x).strip()]
            else:
                expected_route_any = None
        if not isinstance(must_contain, list) or not all(isinstance(x, str) for x in must_contain):
            must_contain = []
        if not isinstance(must_not_contain, list) or not all(isinstance(x, str) for x in must_not_contain):
            must_not_contain = []

        sid = f"smoke_{case_id}_{ts}"
        client_id = os.getenv("CLIENT_ID") or "default"

        # Replay history as prior user turns (same sid, fresh session per case).
        if isinstance(history, list) and history:
            for h in history:
                if not isinstance(h, dict):
                    continue
                hq = str(h.get("question") or "")
                if not hq.strip():
                    continue
                try:
                    _post_ask_json(bot_url, {"q": hq, "sid": sid, "client_id": client_id}, timeout_sec=timeout_sec)
                except Exception:
                    # If history replay fails, still try to run main question for visibility.
                    pass

        payload = {"q": question, "sid": sid, "client_id": client_id}

        try:
            resp = _post_ask_json(bot_url, payload, timeout_sec=timeout_sec)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            errors += 1
            cc = str(row.get("coverage_class") or "UNKNOWN").strip().upper()
            results.append(
                CaseResult(case_id=case_id, status="ERROR", reason=f"http_error: {str(e)[:120]}", coverage_class=cc),
            )
            continue
        except Exception as e:
            errors += 1
            cc = str(row.get("coverage_class") or "UNKNOWN").strip().upper()
            results.append(
                CaseResult(
                    case_id=case_id, status="ERROR", reason=f"request_failed: {str(e)[:120]}", coverage_class=cc
                ),
            )
            continue

        answer = str(resp.get("answer") or "")
        route = _infer_route_from_response(resp)
        cov = str(row.get("coverage_class") or "UNKNOWN").strip().upper()

        # Validations
        if expected_route_any:
            if _norm(route) not in {_norm(x) for x in expected_route_any}:
                failed += 1
                results.append(
                    CaseResult(
                        case_id=case_id,
                        status="FAIL",
                        reason=f"route: got={route!r} want_any={expected_route_any!r}",
                        coverage_class=cov,
                    )
                )
                continue
        elif expected_route and _norm(route) != _norm(expected_route):
            failed += 1
            results.append(
                CaseResult(
                    case_id=case_id,
                    status="FAIL",
                    reason=f"route: got={route!r} want={expected_route!r}",
                    coverage_class=cov,
                ),
            )
            continue

        missing = [x for x in must_contain if x and not _contains_ci(answer, x)]
        if missing:
            failed += 1
            results.append(
                CaseResult(
                    case_id=case_id,
                    status="FAIL",
                    reason=f"must_contain_missing: {missing[:3]}",
                    coverage_class=cov,
                ),
            )
            continue

        forbidden_hit = [x for x in must_not_contain if x and _contains_ci(answer, x)]
        if forbidden_hit:
            failed += 1
            results.append(
                CaseResult(
                    case_id=case_id,
                    status="FAIL",
                    reason=f"must_not_contain_hit: {forbidden_hit[:3]}",
                    coverage_class=cov,
                ),
            )
            continue

        passed += 1
        results.append(CaseResult(case_id=case_id, status="PASS", reason="ok", coverage_class=cov))

    _print_table(results)
    total = passed + failed + errors
    acc = (passed / total) if total else 0.0
    print(f"SUMMARY: passed={passed}, failed={failed}, errors={errors}, total={total} (accuracy={acc:.1%})")
    print()

    _classes = ["STRONG", "WEAK", "TEMPLATE", "UNKNOWN"]
    by_tot: dict[str, int] = {c: 0 for c in _classes}
    by_ok: dict[str, int] = {c: 0 for c in _classes}
    for r in results:
        cc = r.coverage_class if r.coverage_class in by_tot else "UNKNOWN"
        by_tot[cc] = by_tot.get(cc, 0) + 1
        if r.status == "PASS":
            by_ok[cc] = by_ok.get(cc, 0) + 1
    print("┌──────────────┬─────────┬─────────┐")
    print("│ class        │ passed  │ total   │")
    print("├──────────────┼─────────┼─────────┤")
    for c in _classes:
        print(f"│ {c:<12} │ {by_ok[c]:>7} │ {by_tot[c]:>7} │")
    print("└──────────────┴─────────┴─────────┘")

    # Exit code policy:
    # - If baseline is null: exit 0 iff no ERROR (runner can still be used to set baseline).
    # - If baseline is set: require passed >= baseline-2, else exit 1.
    if baseline is None:
        return 0 if errors == 0 else 2

    min_ok = max(0, int(baseline) - 2)
    return 0 if passed >= min_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

