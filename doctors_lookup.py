"""A3.3 deterministic doctor index (names + coarse specialty routing)."""
from __future__ import annotations

import glob
import os
import re
from typing import Any

import alias_lexical
import yaml

_MD_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md")

_IMPLANT_QUERY_LEMMAS = frozenset({"имплант", "имплантация", "имплантолог"})


def _read_md_split(path: str) -> tuple[dict[str, Any], str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    base = os.path.basename(path)
    doc_stem = base[:-3] if base.lower().endswith(".md") else base
    if not raw.lstrip().startswith("---"):
        return {}, raw, doc_stem
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw, doc_stem
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2], doc_stem


def _h1(body: str) -> str:
    m = re.search(r"^##\s+(.+)$", body, flags=re.M)
    return (m.group(1) or "").strip() if m else ""


def _lemma_set(text: str) -> set[str]:
    q = (text or "").lower().replace("ё", "е")
    q = re.sub(r"[^\w\s]", " ", q, flags=re.U)
    toks = [t for t in q.split() if len(t) >= 2]
    return set(alias_lexical.lemma_forms_for_tokens(toks))


def _name_hit_score(q_lem: set[str], display_phrase: str) -> float:
    p_lem = _lemma_set(display_phrase)
    if not p_lem or not q_lem:
        return 0.0
    if p_lem <= q_lem:
        return 1.0
    inter = len(q_lem & p_lem)
    if inter == 0:
        return 0.0
    return inter / max(len(p_lem), 1)


def _is_staff_implant_question(q_raw: str, q_lem: set[str]) -> bool:
    low = q_raw.lower()
    if "врач" not in low and "доктор" not in low:
        return False
    if not (q_lem & _IMPLANT_QUERY_LEMMAS):
        return False
    return bool(re.search(r"\b(кто|какой|какие|чей)\b", low, flags=re.I))


def doctors_lookup(q: str, *, client_id: str) -> dict[str, Any] | None:
    """Return {doc_id, doctor_name, specialty} for a single-doc answer, or None."""
    _ = client_id
    q0 = (q or "").strip()
    if len(q0) < 2:
        return None

    q_lem = _lemma_set(q0)
    paths = sorted(
        p
        for p in glob.glob(os.path.join(_MD_BASE, "doctors__doctor__*.md"))
        if os.path.basename(p).lower() != "doctors__doctor__overview.md"
    )
    if not paths:
        return None

    best: tuple[float, str, str, str | None] | None = None  # score, doc_id, name, specialty

    for path in paths:
        fm, body, stem = _read_md_split(path)
        doc_id = str(fm.get("doc_id") or stem).strip()
        h1 = _h1(body)
        aliases = [str(x).strip() for x in (fm.get("aliases") or []) if str(x).strip()]
        spec = str(fm.get("specialty") or "").strip() or None

        local = 0.0
        for phrase in [h1, *aliases]:
            if not phrase:
                continue
            local = max(local, _name_hit_score(q_lem, phrase))

        if local >= 0.88:
            cand = (local, doc_id, h1 or (aliases[0] if aliases else doc_id), spec)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is not None:
        return {
            "doc_id": best[1],
            "doctor_name": best[2],
            "specialty": best[3],
        }

    if _is_staff_implant_question(q0, q_lem):
        return {
            "doc_id": "doctors__doctor__overview",
            "doctor_name": "Наши врачи",
            "specialty": "implantation",
        }

    return None


def doctor_name_probe(q: str) -> bool:
    """Cheap trigger for doctors_lookup when Resolver topic is not doctors."""
    x = (q or "").strip().lower()
    if not x:
        return False
    if re.search(r"\b(?:доктор|врач)\s+[а-яё]", x):
        return True
    if re.search(
        r"\b[а-яё]{3,}(?:ович|евич|ская|кой|ко|ук|ова|ева|ина|ёва)\b",
        x,
        flags=re.I,
    ):
        return True
    return False
