from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def _load_cases(path: str) -> list[dict[str, Any]]:
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


def _norm(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return (s.lower() if s else None)


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> int:
    from resolver import RESOLVER_SYSTEM_PROMPT, resolve_decision_frame

    assert "Пример формата" in RESOLVER_SYSTEM_PROMPT and "route_intent" in RESOLVER_SYSTEM_PROMPT

    cases = _load_cases(_here("resolver_golden.json"))
    field_fail_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    total = 0
    ok = 0

    for row in cases:
        cid = str(row.get("id") or "")
        q = str(row.get("question") or "").strip()
        expected = row.get("expected") or {}
        if not q or not isinstance(expected, dict):
            continue

        total += 1
        actual_df = resolve_decision_frame(question=q, history=[])
        actual = actual_df.model_dump()

        diff_fields: list[str] = []
        for field in ("route_intent", "service_topic", "query_mode", "service_id"):
            if field not in expected:
                continue
            ev = expected.get(field)
            av = actual.get(field)
            if _norm(ev) != _norm(av):
                diff_fields.append(field)
                field_fail_counts[field] += 1

        # query_mode confusion matrix (only when expected has query_mode)
        if "query_mode" in expected:
            exp_qm = _norm(expected.get("query_mode")) or "<missing>"
            act_qm = _norm(actual.get("query_mode")) or "<missing>"
            confusion[exp_qm][act_qm] += 1

        if not diff_fields:
            ok += 1
            continue

        print("=== FAIL ===")
        print(f"id: {cid}")
        print(f"question: {q}")
        print("expected:")
        print(_pretty(expected))
        print("actual:")
        print(_pretty(actual))
        print(f"diff_fields: {diff_fields}")
        print()

    print("=== SUMMARY ===")
    print(json.dumps({"cases": total, "ok": ok, "accuracy": (ok / total if total else None)}, ensure_ascii=False))

    print("\nfield_fail_counts:")
    for k, v in field_fail_counts.most_common():
        print(f"- {k}: {v}")

    print("\nquery_mode confusion_matrix (expected -> actual counts):")
    for exp, rowc in sorted(confusion.items(), key=lambda x: x[0]):
        row = {act: cnt for act, cnt in rowc.most_common()}
        print(f"- {exp}: {json.dumps(row, ensure_ascii=False)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

