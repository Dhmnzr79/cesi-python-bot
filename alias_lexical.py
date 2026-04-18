"""Леммы (pymorphy3) и char-trigram similarity для alias recall; без импорта retriever."""

from __future__ import annotations

from typing import Iterable

_MORPH = None
_MORPH_FAILED = False


def _analyzer():
    """Ленивая загрузка pymorphy3; при отсутствии пакета — только lower()."""
    global _MORPH, _MORPH_FAILED
    if _MORPH_FAILED:
        return None
    if _MORPH is not None:
        return _MORPH
    try:
        from pymorphy3 import MorphAnalyzer

        _MORPH = MorphAnalyzer()
    except ImportError:
        _MORPH_FAILED = True
        _MORPH = None
    return _MORPH


def lemma_normal_form(word: str) -> str:
    w = (word or "").strip().lower()
    if len(w) < 2:
        return w
    m = _analyzer()
    if m is None:
        return w
    try:
        return m.parse(w)[0].normal_form
    except Exception:
        return w


def lemma_forms_for_tokens(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        t = (t or "").strip().lower()
        if len(t) < 2:
            continue
        out.append(lemma_normal_form(t))
    return out


def _trigram_set(s: str) -> frozenset[str]:
    s = (s or "").lower().strip()
    if len(s) < 3:
        return frozenset()
    padded = f"  {s}  "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def trigram_jaccard(a: str, b: str) -> float:
    """Jaccard по символьным триграммам (устойчивость к опечаткам / окончаниям)."""
    ta = _trigram_set(a)
    tb = _trigram_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb) or 1
    return inter / union


def trigram_alias_boost(q_norm: str, alias_norm: str) -> float:
    """Оценка 0..~0.88 для пары нормализованных строк."""
    if len(q_norm) < 2 or len(alias_norm) < 2:
        return 0.0
    j = trigram_jaccard(q_norm, alias_norm)
    if j >= 0.78:
        return 0.88
    if j >= 0.68:
        return 0.82
    if j >= 0.58:
        return 0.76
    # «парковку» vs «парковка» ~0.54 — нужен проход мягкого алиаса (0.72) без pymorphy3
    if j >= 0.52:
        return 0.75
    if j >= 0.48:
        return 0.7
    return 0.0
