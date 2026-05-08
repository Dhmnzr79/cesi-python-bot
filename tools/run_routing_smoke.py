#!/usr/bin/env python3
"""
HTTP smoke: POST /ask и проверка выбранной темы (doc / service) и fallback (low_score).

Не оценивает текст ответа — только meta из JSON.

Запуск (бот должен быть поднят, напр. PORT=9000):
  python tools/run_routing_smoke.py --base-url http://127.0.0.1:9000

Переменные окружения:
  SMOKE_CLIENT_ID — client_id (по умолчанию из config.DEFAULT_CLIENT_ID)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

# repo root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import DEFAULT_CLIENT_ID  # noqa: E402


def _primary_topic_id(meta: dict[str, Any]) -> str:
    """Идентификатор «темы» для сопоставления с eval: chunk file → doc_id либо каталог/цена."""
    if not isinstance(meta, dict):
        return ""
    did = str(meta.get("doc_id") or "").strip()
    if did:
        return did
    f = str(meta.get("file") or "").strip()
    if f:
        base = os.path.basename(f)
        return os.path.splitext(base)[0] if base else ""
    ms = str(meta.get("matched_service_id") or "").strip()
    if ms:
        return ms
    return ""


def _post_ask(base_url: str, client_id: str, sid: str, q: str, timeout_sec: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/ask"
    body = json.dumps(
        {"q": q, "client_id": client_id, "sid": sid},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _check_case(case: dict[str, Any], meta: dict[str, Any]) -> tuple[bool, str]:
    ls = bool(meta.get("low_score"))
    topic = _primary_topic_id(meta)

    if "expect_low_score" in case and bool(case["expect_low_score"]) != ls:
        return False, f"low_score: got {ls}, expected {case['expect_low_score']}"

    exp_docs = case.get("expected_doc_any") or case.get("expected_topic_any")
    if isinstance(exp_docs, str):
        exp_docs = [exp_docs]
    if exp_docs:
        ok = topic in {str(x).strip() for x in exp_docs if str(x).strip()}
        if not ok:
            return False, f"topic_id={topic!r} not in expected {exp_docs!r}"

    forb = (case.get("forbidden_doc_id") or "").strip()
    if forb and topic == forb:
        return False, f"forbidden topic_id matched: {forb!r}"

    exp_intent = case.get("expect_intent")
    if exp_intent is not None:
        got = str(meta.get("intent") or "").strip().lower()
        want = str(exp_intent).strip().lower()
        if got != want:
            return False, f"intent: got {got!r}, expected {want!r}"

    if case.get("expect_handoff") is True:
        if not bool(meta.get("handoff_filter")):
            return False, "expected handoff_filter in meta"

    return True, ""


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
    ap = argparse.ArgumentParser(description="Routing smoke via POST /ask")
    ap.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:9000"),
        help="Bot origin (no trailing slash)",
    )
    ap.add_argument(
        "--cases",
        default=os.path.join(_ROOT, "evals", "routing_smoke_cases.json"),
        help="Path to JSON cases file",
    )
    ap.add_argument(
        "--client-id",
        default=os.environ.get("SMOKE_CLIENT_ID", DEFAULT_CLIENT_ID),
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (sec)")
    args = ap.parse_args()

    cases_path = os.path.abspath(args.cases)
    cases = _load_cases(cases_path)

    fail = 0
    for case in cases:
        cid = str(case.get("id") or case.get("case_id") or "").strip() or "?"
        q = (case.get("q") or "").strip()
        if not q:
            print(f"FAIL {cid}: empty q")
            fail += 1
            continue

        sid = str(case.get("sid") or "").strip() or uuid.uuid4().hex
        try:
            data = _post_ask(args.base_url, args.client_id, sid, q, args.timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            print(f"FAIL {cid}: HTTP {e.code} {body}")
            fail += 1
            continue
        except Exception as e:
            print(f"FAIL {cid}: request error: {e}")
            fail += 1
            continue

        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        ok, reason = _check_case(case, meta)
        topic = _primary_topic_id(meta)
        if ok:
            print(f"OK   {cid} topic={topic!r} low_score={bool(meta.get('low_score'))}")
        else:
            print(f"FAIL {cid} topic={topic!r} low_score={bool(meta.get('low_score'))}: {reason}")
            fail += 1

    n = len(cases)
    print(f"--- {n - fail}/{n} passed", file=sys.stderr)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
