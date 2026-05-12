"""PR #1.10 transitional: legacy 12-band alias scorer for shadow telemetry only.

Runtime decisions MUST NOT use this module. See `retriever.run_alias_pipeline`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import alias_lexical
from config import CORPUS_PATH

_RE_H2 = re.compile(r"^##\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)
_RE_H3 = re.compile(r"^###\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\{#.*?\}", " ", s)
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.U)
    return re.sub(r"\s+", " ", s).strip()


_ALIAS_STOP_WORDS = frozenset(
    {
        "а",
        "у",
        "в",
        "во",
        "на",
        "по",
        "за",
        "к",
        "ко",
        "с",
        "со",
        "о",
        "об",
        "от",
        "до",
        "из",
        "при",
        "про",
        "без",
        "для",
        "над",
        "под",
        "вас",
        "вам",
        "нас",
        "мне",
        "меня",
        "есть",
        "ли",
        "можно",
        "нельзя",
        "получить",
        "получается",
        "скажите",
        "подскажите",
        "расскажите",
        "хочу",
        "нужно",
        "надо",
        "будет",
        "это",
        "то",
        "так",
        "как",
        "что",
        "где",
        "когда",
        "почему",
        "зачем",
        "или",
        "и",
        "же",
        "ли",
        "бы",
        "не",
        "ни",
        "уже",
        "еще",
        "ещё",
        "только",
        "лишь",
        "очень",
        "все",
        "всё",
        "там",
        "тут",
        "здесь",
    }
)


def _core_tokens(text: str) -> list[str]:
    qn = _norm_text(text)
    out: list[str] = []
    for t in qn.split():
        if len(t) < 2:
            continue
        if t in _ALIAS_STOP_WORDS:
            continue
        out.append(t)
    return out


def _strong_core_tokens(core: list[str]) -> list[str]:
    return [t for t in core if len(t) >= 3 or any(ch.isdigit() for ch in t)]


def _all_tokens_in_text(tokens: list[str], an: str) -> bool:
    if not tokens:
        return False
    padded = f" {an} "
    for t in tokens:
        if len(t) < 2:
            return False
        if f" {t} " not in padded:
            return False
    return True


def _heading_plain(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^#{1,6}\s*", "", s)
    return s


def _chunk_alias_terms(ch: dict) -> list[str]:
    if not isinstance(ch, dict):
        return []
    terms: list[str] = []
    aliases = ch.get("aliases") or []
    if isinstance(aliases, list):
        for a in aliases:
            if isinstance(a, str) and a.strip():
                terms.append(a.strip())
    h2 = _heading_plain(str(ch.get("h2") or ""))
    h3 = _heading_plain(str(ch.get("h3") or ""))
    h2_id = str(ch.get("h2_id") or "").strip()
    h3_id = str(ch.get("h3_id") or "").strip()
    if h2:
        terms.append(h2)
    if h3:
        terms.append(h3)
    if h2_id:
        terms.append(h2_id.replace("-", " "))
    if h3_id:
        terms.append(h3_id.replace("-", " "))
    return terms


def _build_alias_index(corpus: list) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for ch in corpus or []:
        if not isinstance(ch, dict):
            continue
        seen_terms: set[str] = set()
        for raw in _chunk_alias_terms(ch):
            norm = _norm_text(raw)
            if len(norm) < 2 or norm in seen_terms:
                continue
            seen_terms.add(norm)
            idx.setdefault(norm, []).append(ch)
    return idx


def _alias_probe_terms(q: str) -> list[str]:
    probes: list[str] = []
    qn = _norm_text(q)
    if qn:
        probes.append(qn)
    core = _core_tokens(q)
    probes.extend(core)
    if len(core) >= 2:
        probes.extend(f"{core[i]} {core[i + 1]}" for i in range(len(core) - 1))
    out: list[str] = []
    seen: set[str] = set()
    for p in probes:
        pn = _norm_text(p)
        if len(pn) < 2 or pn in seen:
            continue
        seen.add(pn)
        out.append(pn)
    return out


def _alias_hit_score_raw_for_chunk(q: str, ch: dict) -> float:
    qn = _norm_text(q)
    if not qn:
        return 0.0
    q_core = _core_tokens(q)
    q_core_joint = " ".join(q_core) if q_core else ""
    q_tokens = {t for t in qn.split() if len(t) >= 2}
    q_core_set = {t for t in q_core if len(t) >= 2}
    best = 0.0
    for raw in _chunk_alias_terms(ch):
        an = _norm_text(raw)
        if not an or len(an) < 2:
            continue
        a_core = _core_tokens(raw)
        a_core_joint = " ".join(a_core) if a_core else ""
        a_core_set = {t for t in a_core if len(t) >= 2}

        if q_core_joint and len(q_core_joint) >= 4 and q_core_joint in an:
            best = max(best, 0.92)
        if a_core_joint and len(a_core_joint) >= 4 and a_core_joint in qn:
            best = max(best, 0.92)

        strong_q = _strong_core_tokens(q_core)
        if len(strong_q) == 2 and _all_tokens_in_text(strong_q, an):
            best = max(best, 0.9)
        if len(q_core) == 2 and all(len(t) >= 2 for t in q_core) and _all_tokens_in_text(q_core, an):
            best = max(best, 0.9)

        if 2 <= len(q_core) <= 3 and q_core_set and q_core_set.issubset(a_core_set):
            best = max(best, 0.88 if len(q_core) == 3 else 0.9)

        if q_core_set and a_core_set:
            inter_c = len(q_core_set & a_core_set)
            if inter_c > 0:
                q_cov_c = inter_c / max(len(q_core_set), 1)
                if len(q_core_set) <= 3 and q_cov_c >= 0.67:
                    best = max(best, 0.86)
                elif q_cov_c >= 0.5:
                    best = max(best, 0.8)

        if qn == an:
            best = max(best, 1.0)
            continue
        if qn in an or an in qn:
            ratio = min(len(qn), len(an)) / max(len(qn), len(an))
            best = max(best, 0.93 if ratio >= 0.85 else 0.82)
            continue
        a_tokens = {t for t in an.split() if len(t) >= 2}
        if not q_tokens or not a_tokens:
            continue
        inter = len(q_tokens & a_tokens)
        if inter == 0:
            continue
        overlap = inter / max(len(q_tokens), len(a_tokens))
        q_cover = inter / max(len(q_tokens), 1)
        a_cover = inter / max(len(a_tokens), 1)
        if q_cover >= 0.9 and a_cover >= 0.4:
            best = max(best, 0.9)
        elif q_cover >= 0.75 and a_cover >= 0.35:
            best = max(best, 0.85)
        elif q_cover >= 0.6:
            best = max(best, 0.8)
        elif overlap >= 0.55:
            best = max(best, 0.72)
    return round(best, 4)


def _lemma_join_token_match(inner: str, outer: str) -> bool:
    inner_t = inner.split()
    outer_t = outer.split()
    if not inner_t or not outer_t:
        return False
    if len(inner_t) == 1:
        return inner_t[0] in outer_t
    for i in range(len(outer_t) - len(inner_t) + 1):
        if outer_t[i : i + len(inner_t)] == inner_t:
            return True
    return False


def _lemma_alias_channel(q: str, ch: dict) -> float:
    q_core = _core_tokens(q)
    if not q_core:
        return 0.0
    q_lem = alias_lexical.lemma_forms_for_tokens(q_core)
    q_set = {x for x in q_lem if len(x) >= 2}
    if not q_set:
        return 0.0
    best = 0.0
    q_join = " ".join(q_lem)

    for raw in _chunk_alias_terms(ch):
        a_core = _core_tokens(raw)
        if a_core:
            a_lem = alias_lexical.lemma_forms_for_tokens(a_core)
        else:
            toks = [
                t
                for t in _norm_text(raw).split()
                if len(t) >= 2 and t not in _ALIAS_STOP_WORDS
            ]
            a_lem = alias_lexical.lemma_forms_for_tokens(toks)
        a_set = {x for x in a_lem if len(x) >= 2}
        if not a_set:
            continue

        if q_set <= a_set:
            best = max(best, 0.92)
        if len(a_set) <= 5 and a_set <= q_set:
            best = max(best, 0.88)

        inter = len(q_set & a_set)
        union = len(q_set | a_set) or 1
        j = inter / union
        if len(q_set) >= 2 and j >= 0.55:
            best = max(best, 0.86)
        elif j >= 0.45:
            best = max(best, 0.78)

        a_join = " ".join(a_lem)
        if len(q_join) >= 3 and _lemma_join_token_match(q_join, a_join):
            best = max(best, 0.93)
        if len(a_join) >= 4 and _lemma_join_token_match(a_join, q_join):
            best = max(best, 0.9)

    return round(best, 4)


def _trigram_alias_channel(q: str, ch: dict) -> float:
    qn = _norm_text(q)
    if len(qn) < 2:
        return 0.0
    best = 0.0
    for raw in _chunk_alias_terms(ch):
        an = _norm_text(raw)
        if len(an) < 2:
            continue
        b = alias_lexical.trigram_alias_boost(qn, an)
        for tok in an.split():
            if len(tok) < 4:
                continue
            b = max(b, alias_lexical.trigram_alias_boost(qn, tok))
        if b > best:
            best = b
    return round(best, 4)


def alias_hit_score_legacy_for_chunk(q: str, ch: dict) -> float:
    raw = _alias_hit_score_raw_for_chunk(q, ch)
    lem = _lemma_alias_channel(q, ch)
    tri = _trigram_alias_channel(q, ch)
    return round(max(raw, lem, tri), 4)


_CORPUS_CACHE: list | None = None
_ALIAS_INDEX_CACHE: dict[str, list[dict]] | None = None


def _corpus_and_index() -> tuple[list, dict[str, list[dict]]]:
    global _CORPUS_CACHE, _ALIAS_INDEX_CACHE
    if _CORPUS_CACHE is None:
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                _CORPUS_CACHE = [json.loads(line) for line in f if line.strip()]
        except OSError:
            _CORPUS_CACHE = []
        _ALIAS_INDEX_CACHE = _build_alias_index(_CORPUS_CACHE)
    return _CORPUS_CACHE, _ALIAS_INDEX_CACHE or {}


def corpus_alias_leader_legacy(q: str, *, client_id: str | None = None) -> tuple[dict | None, float]:
    """Historical behavior: same candidate expansion + legacy scorer."""
    corpus, alias_idx = _corpus_and_index()
    probe_terms = _alias_probe_terms(q)
    candidate_map: dict[tuple[Any, Any, Any], dict] = {}
    for term in probe_terms:
        for ch in alias_idx.get(term, []):
            if client_id and ch.get("client_id") != client_id:
                continue
            key = (
                ch.get("file"),
                ch.get("h2_id") or ch.get("h2"),
                ch.get("h3_id") or ch.get("h3"),
            )
            candidate_map[key] = ch
    cands = list(candidate_map.values())
    if not cands:
        cands = [
            ch
            for ch in corpus
            if isinstance(ch, dict) and (not client_id or ch.get("client_id") == client_id)
        ]
    best_chunk = None
    best_score = 0.0
    for ch in cands:
        sc = alias_hit_score_legacy_for_chunk(q, ch)
        if sc > best_score:
            best_score = sc
            best_chunk = ch
    if not best_chunk:
        return None, 0.0
    return dict(best_chunk), round(best_score, 4)


def legacy_chunk_key(ch: dict | None) -> str | None:
    if not isinstance(ch, dict):
        return None
    return "|".join(
        [
            str(ch.get("client_id") or ""),
            str(ch.get("file") or ""),
            str(ch.get("h2_id") or ch.get("h2") or ""),
            str(ch.get("h3_id") or ch.get("h3") or ""),
        ]
    )
