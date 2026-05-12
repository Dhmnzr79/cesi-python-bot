"""Индекс, эмбеддинги, поиск, rerank; метаданные чанков."""
import json
import os
import re
import time
import threading
from typing import Any

import numpy as np

from config import (
    ALIAS_EMB_PATH,
    ALIAS_ROWS_PATH,
    BROAD_QUERY_MAX_WORDS,
    CORPUS_PATH,
    EMB_PATH,
    EMB_MODEL,
    RERANK_MODEL,
    RETRIEVE_CACHE_MAXSIZE,
    RETRIEVE_CACHE_TTL_SEC,
)
from core.routing_loader import THRESHOLDS
from llm import client
from logging_setup import get_logger, log_json, log_llm_usage
from meta_loader import get_doc_meta, get_doc_path

logger = get_logger("bot")

# Термины, из-за которых вопрос не считаем «широким» (см. bot_architecture_v3)
_BROAD_EXCLUDE_TERMS = (
    "цена",
    "стоимость",
    "адрес",
    "телефон",
    "прайс",
    "руб",
    "whatsapp",
    "контакт",
)

_CORPUS: list | None = None
_RE_H2 = re.compile(r"^##\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)
_RE_H3 = re.compile(r"^###\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)
_SECTION_CACHE: dict[str, dict] = {}
_EMB: np.ndarray | None = None
_EMB_LOAD_ERROR: str | None = None
_ALIAS_INDEX: dict[str, list[dict]] | None = None
_RETRIEVE_CACHE_LOCK = threading.RLock()
_RETRIEVE_CACHE: dict[tuple[str, int, str, str], tuple[float, list[dict]]] = {}


def load_corpus_if_needed() -> list:
    global _CORPUS, _ALIAS_INDEX
    if _CORPUS is None:
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                _CORPUS = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            _CORPUS = []
        _ALIAS_INDEX = _build_alias_index(_CORPUS)
    return _CORPUS


def _get_embeddings() -> np.ndarray | None:
    global _EMB, _EMB_LOAD_ERROR
    if _EMB is not None:
        return _EMB
    if _EMB_LOAD_ERROR is not None:
        return None
    try:
        _EMB = np.load(EMB_PATH)
        return _EMB
    except Exception as e:
        _EMB_LOAD_ERROR = str(e)
        log_json(logger, "embeddings_load_failed", emb_path=EMB_PATH, err=_EMB_LOAD_ERROR)
        return None


def extract_id_from_heading(txt: str) -> str | None:
    if not isinstance(txt, str):
        return None
    m = re.search(r"\{\s*#([^\}]+)\s*\}", txt)
    return m.group(1).strip() if m else None


def get_chunk_by_ref(ref: str, *, client_id: str | None = None) -> dict | None:
    if not ref or "#" not in ref:
        return None
    fname, anchor = ref.split("#", 1)
    base = os.path.basename(fname)
    if not base.endswith(".md"):
        base = base + ".md"
    a = (anchor or "").strip().lower()
    corpus = load_corpus_if_needed()
    cands = [ch for ch in corpus if os.path.basename(ch.get("file", "") or "") == base]
    if client_id:
        client_cands = [ch for ch in cands if (ch.get("client_id") or "") == client_id]
        if not client_cands:
            return None
        cands = client_cands
    if not cands:
        return None
    if a in ("overview", "korotko", "", None):
        for ch in cands:
            h3_id = (ch.get("h3_id") or "").strip().lower()
            if (not ch.get("h2_id") and not ch.get("h3_id")) or h3_id in {"overview", "korotko"}:
                ch["_score"] = 1.0
                return ch
        ch = cands[0]
        ch["_score"] = 1.0
        return ch
    for ch in cands:
        hid2 = ch.get("h2_id") or extract_id_from_heading(ch.get("h2"))
        hid3 = ch.get("h3_id") or extract_id_from_heading(ch.get("h3"))
        if a in {
            (hid3 or "").lower(),
            (hid2 or "").lower(),
            str(ch.get("h3") or "").lower(),
            str(ch.get("h2") or "").lower(),
        }:
            ch["_score"] = 1.0
            return ch
    return None


def _load_doc_text(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8-sig") as f:
        return f.read()


def _build_section_index(md_path: str) -> dict:
    abs_path = os.path.abspath(md_path)
    cached = _SECTION_CACHE.get(abs_path)
    if cached:
        return cached
    try:
        text = _load_doc_text(abs_path)
    except OSError:
        text = ""
    h2 = [(m.start(), m.group(1)) for m in _RE_H2.finditer(text)]
    h3 = [(m.start(), m.group(1)) for m in _RE_H3.finditer(text)]
    data = {"text": text, "h2": h2, "h3": h3}
    _SECTION_CACHE[abs_path] = data
    return data


def _infer_section_ids(md_path: str, fragment: str) -> tuple[str | None, str | None]:
    if not md_path or not fragment:
        return (None, None)
    idx = _build_section_index(md_path)
    doc_text = idx["text"] or ""

    lines = (fragment or "").splitlines()
    needles = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("<!--"):
            continue
        needles.append(s[:120])
        break
    if not needles:
        needles.append((fragment or "").strip()[:120])

    pos = -1
    for nd in needles:
        if not nd:
            continue
        pos = doc_text.find(nd)
        if pos >= 0:
            break
    if pos >= 0:
        h2_id = None
        h3_id = None
        for p, hid in idx["h2"]:
            if p <= pos:
                h2_id = hid
            else:
                break
        for p, hid in idx["h3"]:
            if p <= pos:
                h3_id = hid
            else:
                break
        return (h2_id, h3_id)

    def _norm_local(s: str) -> str:
        s = s or ""
        s = re.sub(r"[*_`]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    needles_n = [_norm_local(x) for x in needles if x]
    if not needles_n:
        return (idx["h2"][0][1], None) if len(idx["h2"]) == 1 else (None, None)

    h2s = idx["h2"]
    if not h2s:
        return (None, None)
    bounds = []
    for i, (p, hid) in enumerate(h2s):
        p2 = h2s[i + 1][0] if i + 1 < len(h2s) else len(doc_text)
        bounds.append((p, p2, hid))

    for start, end, hid in bounds:
        block = _norm_local(doc_text[start:end])
        if any(nd and nd in block for nd in needles_n):
            h3_id = None
            for p, h3id in idx["h3"]:
                if start <= p < end:
                    h3_id = h3_id or h3id
            return (hid, h3_id)

    if len(h2s) == 1:
        return (h2s[0][1], None)
    return (None, None)


def chunk_doc_type(item: Any) -> str | None:
    if isinstance(item, dict):
        dt = item.get("doc_type") or item.get("topic")
        if dt:
            return dt
        base = os.path.basename(item.get("file") or "")
        if base:
            fm = get_doc_meta(base, client_id=item.get("client_id")) or {}
            return fm.get("doc_type") or fm.get("topic")
    return None


def chunk_score(item: Any) -> float | None:
    try:
        return float(item[1])
    except Exception:
        try:
            return float(item.get("_score"))
        except Exception:
            return None


def chunk_info(ch: dict, sc: float | None = None) -> dict:
    meta = {}
    text = None
    cid = None
    doc = None
    h2 = None
    h3 = None
    doc_type = None
    subtype = None

    if isinstance(ch, dict):
        meta = ch.get("meta", {}) or {}
        text = ch.get("text")
        cid = ch.get("id")
        doc = ch.get("file") or meta.get("doc") or ch.get("doc")
        h2 = ch.get("h2_id") or meta.get("h2_id")
        h3 = ch.get("h3_id") or meta.get("h3_id")
        doc_type = meta.get("doc_type") or ch.get("doc_type")
        subtype = meta.get("subtype") or ch.get("subtype")
    else:
        meta = getattr(ch, "meta", {}) or {}
        text = getattr(ch, "text", None)
        cid = getattr(ch, "id", None)
        doc = meta.get("doc") or getattr(ch, "file", None)
        h2 = meta.get("h2_id")
        h3 = meta.get("h3_id")
        doc_type = meta.get("doc_type")
        subtype = meta.get("subtype")

    doc_base = os.path.basename(doc) if doc else None
    ch_client_id = ch.get("client_id") if isinstance(ch, dict) else None
    full_md_path = None
    if doc_base:
        full_md_path = get_doc_path(doc_base, client_id=ch_client_id)
    if not full_md_path:
        guess = doc if os.path.exists(doc or "") else os.path.join("md", doc_base or "")
        full_md_path = guess if os.path.exists(guess) else None

    if (h2 is None and h3 is None) and full_md_path and text:
        h2_guess, h3_guess = _infer_section_ids(full_md_path, text)
        h2 = h2 or h2_guess
        h3 = h3 or h3_guess

    doc_base = os.path.basename(doc) if doc else None
    fm = get_doc_meta(doc_base, client_id=ch_client_id) if doc_base else {}
    if not doc_type:
        doc_type = fm.get("doc_type")
    if not subtype:
        subtype = fm.get("subtype")

    return {
        "id": cid,
        "doc": doc,
        "doc_type": doc_type,
        "subtype": subtype,
        "h2_id": h2,
        "h3_id": h3,
        "score": (round(float(sc), 4) if sc is not None else None),
        "snippet": (text[:180] if isinstance(text, str) else None),
    }


def chunk_is_overview(c: dict) -> bool:
    h2 = (c.get("h2_id") or "").strip().lower()
    h3 = (c.get("h3_id") or "").strip().lower()
    return (not h2 and not h3) or h2 in {"overview", "korotko"} or h3 in {"overview", "korotko"}


_LEADING_QUERY_FILLERS = re.compile(
    r"^(?:[ауоыэи]+\s+|ну\s+|а\s+|э\s+|эм\s+)+",
    re.I,
)


def normalize_retrieval_query(q: str) -> str:
    """Единая политика перед embed: частицы в начале, ё/е, пробелы.

    Не трогаем смысловое тело; пустой результат после снятия префиксов — норма.
    """
    s = (q or "").strip()
    if not s:
        return ""
    s = s.replace("ё", "е").replace("Ё", "Е")
    prev = None
    while prev != s:
        prev = s
        s = _LEADING_QUERY_FILLERS.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def broad_query_detect(q: str) -> bool:
    qn = (q or "").strip().lower()
    words = qn.split()
    if len(words) > BROAD_QUERY_MAX_WORDS:
        return False
    return not any(t in qn for t in _BROAD_EXCLUDE_TERMS)


def prefer_overview_if_broad(cands: list, broad: bool) -> list:
    if not broad or len(cands) < 2:
        return cands
    top_files = [os.path.basename(c.get("file") or "") for c in cands[:3]]
    if len(set(top_files)) != 1:
        return cands
    for i, c in enumerate(cands):
        if chunk_is_overview(c):
            if i > 0:
                return [c] + [x for j, x in enumerate(cands) if j != i]
            return cands
    return cands


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\{#.*?\}", " ", s)
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.U)
    return re.sub(r"\s+", " ", s).strip()


# Служебные слова для alias matching: сравниваем ядро запроса, не бытовую оболочку.
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
    """Токены смыслового ядра: без служебных слов, порядок сохраняется."""
    qn = _norm_text(text)
    out: list[str] = []
    for t in qn.split():
        if len(t) < 2:
            continue
        if t in _ALIAS_STOP_WORDS:
            continue
        out.append(t)
    return out


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
    """Индекс по нормализованным alias-термам -> список чанков."""
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
    """Запросные ключи для alias-индекса: фраза, токены и биграммы ядра."""
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


_ALIAS_EMB_MATRIX: np.ndarray | None = None
_ALIAS_ROW_CORPUS_IDX: np.ndarray | None = None
_ALIAS_ROW_CLIENT: list[str] | None = None
_ALIAS_ARTIFACTS_ERROR: str | None = None


def _legacy_shadow_enabled() -> bool:
    return os.getenv("ALIAS_LEGACY_SHADOW", "1").lower() in ("1", "true", "yes")


def _embedding_dim_for_empty_alias_matrix() -> int:
    try:
        if os.path.isfile(EMB_PATH):
            arr = np.load(EMB_PATH)
            if arr.ndim == 2 and arr.shape[1] > 0:
                return int(arr.shape[1])
    except OSError:
        pass
    return 1536


def _load_alias_embed_artifacts() -> None:
    global _ALIAS_EMB_MATRIX, _ALIAS_ROW_CORPUS_IDX, _ALIAS_ROW_CLIENT, _ALIAS_ARTIFACTS_ERROR
    if _ALIAS_EMB_MATRIX is not None:
        return
    dim0 = _embedding_dim_for_empty_alias_matrix()
    try:
        if not os.path.isfile(ALIAS_EMB_PATH) or not os.path.isfile(ALIAS_ROWS_PATH):
            _ALIAS_ARTIFACTS_ERROR = "missing_alias_files"
            _ALIAS_EMB_MATRIX = np.zeros((0, dim0), dtype=np.float32)
            _ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
            _ALIAS_ROW_CLIENT = []
            return
        emb = np.load(ALIAS_EMB_PATH)
        clients: list[str] = []
        idxs: list[int] = []
        with open(ALIAS_ROWS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                idxs.append(int(o["corpus_idx"]))
                clients.append(str(o.get("client_id") or ""))
        if emb.shape[0] != len(idxs):
            _ALIAS_ARTIFACTS_ERROR = "alias_rows_emb_shape_mismatch"
            _ALIAS_EMB_MATRIX = np.zeros((0, int(emb.shape[1]) if emb.ndim == 2 else dim0), dtype=np.float32)
            _ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
            _ALIAS_ROW_CLIENT = []
            return
        if emb.shape[0] == 0:
            _ALIAS_EMB_MATRIX = emb.astype(np.float32)
            _ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
            _ALIAS_ROW_CLIENT = []
            return
        _ALIAS_EMB_MATRIX = emb.astype(np.float32)
        _ALIAS_ROW_CORPUS_IDX = np.array(idxs, dtype=np.int32)
        _ALIAS_ROW_CLIENT = clients
        _ALIAS_ARTIFACTS_ERROR = ""
    except Exception as e:
        _ALIAS_ARTIFACTS_ERROR = str(e)
        _ALIAS_EMB_MATRIX = np.zeros((0, dim0), dtype=np.float32)
        _ALIAS_ROW_CORPUS_IDX = np.array([], dtype=np.int32)
        _ALIAS_ROW_CLIENT = []


def _chunk_key_tuple(ch: dict) -> tuple[Any, Any, Any]:
    return (
        ch.get("file"),
        ch.get("h2_id") or ch.get("h2"),
        ch.get("h3_id") or ch.get("h3"),
    )


def _corpus_index_for_chunk(corpus: list, ch: dict) -> int | None:
    key = _chunk_key_tuple(ch)
    for i, c0 in enumerate(corpus):
        if isinstance(c0, dict) and _chunk_key_tuple(c0) == key:
            return int(i)
    return None


def _deterministic_alias_for_chunk(q_norm: str, ch: dict) -> tuple[float, bool, str]:
    """Exact / near-exact only. Returns (score, exact_hit, tier_label)."""
    if not q_norm:
        return 0.0, False, "none"
    thr = THRESHOLDS.alias
    for raw in _chunk_alias_terms(ch):
        an = _norm_text(raw)
        if not an or len(an) < 2:
            continue
        if q_norm == an:
            return 1.0, True, "exact"
    best_near = 0.0
    for raw in _chunk_alias_terms(ch):
        an = _norm_text(raw)
        if not an or len(an) < 2:
            continue
        if q_norm in an or an in q_norm:
            ratio = min(len(q_norm), len(an)) / max(len(q_norm), len(an), 1)
            if ratio >= float(thr.near_exact_length_ratio_min):
                best_near = max(best_near, float(thr.near_exact_score))
    if best_near > 0:
        return best_near, False, "near_exact"
    return 0.0, False, "none"


def _tier_rank(tier: str) -> int:
    return {
        "exact": 5,
        "near_exact": 4,
        "embed_high": 3,
        "rescue": 2,
        "embed_medium": 1,
        "none": 0,
    }.get(tier, 0)


def _classify_alias_tier_for_chunk(
    *,
    q_norm: str,
    ch: dict,
    emb_sim: float,
    thr: Any,
    rescue_env: bool,
    top_emb_corpus_idx: int | None,
    corpus_pos: int | None,
) -> tuple[float, str]:
    """Returns (effective_score, tier)."""
    det, exact_hit, det_tier = _deterministic_alias_for_chunk(q_norm, ch)
    sim = float(emb_sim)

    if exact_hit or det >= 1.0:
        return 1.0, "exact"

    if det_tier == "near_exact" and det >= float(thr.near_exact_score):
        return float(thr.near_exact_score), "near_exact"

    hi = float(thr.embedding_high_min)
    if sim >= hi:
        return sim, "embed_high"

    if (
        rescue_env
        and corpus_pos is not None
        and top_emb_corpus_idx is not None
        and corpus_pos == top_emb_corpus_idx
        and sim >= float(thr.rescue_min_sim)
    ):
        return min(sim, float(thr.rescue_effective_cap)), "rescue"

    med_lo = float(thr.embedding_medium_min)
    med_hi = float(thr.embedding_medium_max)
    cap_med = float(thr.embedding_medium_score_cap)
    if med_lo <= sim < med_hi:
        return min(sim, cap_med), "embed_medium"

    return 0.0, "none"


def _expand_alias_candidates_with_embed_topk(
    corpus: list,
    chunk_max: np.ndarray,
    *,
    client_id: str | None,
    k_extra: int,
) -> list[dict]:
    """Add top-k chunks by alias-embedding max-sim to the candidate pool."""
    if chunk_max.size == 0 or k_extra <= 0:
        return []
    scores = chunk_max.copy()
    for i, c0 in enumerate(corpus):
        if not isinstance(c0, dict):
            scores[i] = -1.0
            continue
        if client_id and c0.get("client_id") != client_id:
            scores[i] = -1.0
    order = np.argsort(-scores)[: max(k_extra, 0)]
    out: list[dict] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for i in order:
        if float(scores[int(i)]) < 0:
            continue
        ch = corpus[int(i)]
        if not isinstance(ch, dict):
            continue
        key = _chunk_key_tuple(ch)
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out


def run_alias_pipeline(q: str, *, client_id: str | None = None) -> dict[str, Any]:
    """PR #1.10: exact / near-exact + build-time alias embeddings + controlled rescue (+ optional legacy shadow)."""
    thr = THRESHOLDS.alias
    q_raw = (q or "").strip()
    q_norm = _norm_text(q)
    corpus = load_corpus_if_needed()
    n = len(corpus)
    _load_alias_embed_artifacts()

    alias_idx = _ALIAS_INDEX or {}
    probe_terms = _alias_probe_terms(q)
    candidate_map: dict[tuple[Any, Any, Any], dict] = {}
    for term in probe_terms:
        for ch in alias_idx.get(term, []):
            if client_id and ch.get("client_id") != client_id:
                continue
            candidate_map[_chunk_key_tuple(ch)] = ch
    cands = list(candidate_map.values())
    if not cands:
        cands = [
            ch
            for ch in corpus
            if isinstance(ch, dict) and (not client_id or ch.get("client_id") == client_id)
        ]

    chunk_max = np.full(n, -1.0, dtype=np.float32)
    top_emb_corpus_idx: int | None = None
    sim_top = 0.0
    sim_second = 0.0
    q_embed = normalize_retrieval_query(q_raw) or q_raw
    if (
        _ALIAS_EMB_MATRIX is not None
        and int(_ALIAS_EMB_MATRIX.shape[0]) > 0
        and _ALIAS_ROW_CORPUS_IDX is not None
        and q_embed.strip()
    ):
        try:
            v = embed_q(q_embed)
            sims_rows = _ALIAS_EMB_MATRIX @ v
            row_ok = np.ones(sims_rows.shape[0], dtype=bool)
            if client_id:
                row_ok = np.array(
                    [(not cid) or (cid == client_id) for cid in (_ALIAS_ROW_CLIENT or [])],
                    dtype=bool,
                )
            sims_f = sims_rows.astype(np.float32).copy()
            sims_f[~row_ok] = -1.0
            np.maximum.at(chunk_max, _ALIAS_ROW_CORPUS_IDX, sims_f)
            for i in range(n):
                c0 = corpus[i]
                if not isinstance(c0, dict):
                    chunk_max[i] = -1.0
                elif client_id and c0.get("client_id") != client_id:
                    chunk_max[i] = -1.0
            valid = chunk_max[chunk_max >= 0.0]
            if valid.size >= 1:
                sim_top = float(np.max(valid))
                top_emb_corpus_idx = int(np.argmax(chunk_max))
            if valid.size >= 2:
                srt = np.sort(valid)
                sim_second = float(srt[-2])
        except Exception as e:
            log_json(logger, "alias_embed_query_failed", err=str(e)[:200])

    margin = float(sim_top - sim_second) if sim_top > 0 and sim_second >= 0 else 1.0
    core = _core_tokens(q_raw)
    short = len(q_norm) <= int(thr.rescue_max_query_chars) and len(core) <= int(
        thr.rescue_max_core_tokens
    )
    rescue_env = (
        bool(short)
        and margin >= float(thr.rescue_margin_min)
        and sim_top >= float(thr.rescue_min_sim)
    )

    extra = _expand_alias_candidates_with_embed_topk(
        corpus,
        chunk_max,
        client_id=client_id,
        k_extra=int(thr.embed_matrix_top_chunks),
    )
    merged: dict[tuple[Any, Any, Any], dict] = { _chunk_key_tuple(ch): ch for ch in cands}
    for ch in extra:
        merged.setdefault(_chunk_key_tuple(ch), ch)
    merged_cands = list(merged.values())

    best_ch: dict | None = None
    best_eff = -1.0
    best_tier = "none"
    best_sim = -1.0
    best_exact = False
    rank = 0

    scored: list[tuple[int, float, float, str, dict]] = []
    for ch in merged_cands:
        pos = _corpus_index_for_chunk(corpus, ch)
        emb_val = float(chunk_max[pos]) if pos is not None and 0 <= pos < n else -1.0
        eff, tier = _classify_alias_tier_for_chunk(
            q_norm=q_norm,
            ch=ch,
            emb_sim=emb_val,
            thr=thr,
            rescue_env=rescue_env,
            top_emb_corpus_idx=top_emb_corpus_idx,
            corpus_pos=pos,
        )
        scored.append((_tier_rank(tier), eff, emb_val, tier, ch))

    scored = [x for x in scored if x[0] > 0 or x[1] > 1e-6]
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    if scored:
        tr, eff, emb_val, tier, ch = scored[0]
        best_ch = dict(ch)
        best_eff = float(eff)
        best_tier = tier
        best_sim = float(emb_val)
        best_exact = tier == "exact"
        rank = 1
        if len(scored) >= 2:
            _, e2, _, _, _ = scored[1]
            if e2 > 0:
                margin = float(best_eff - e2)

    diag: dict[str, Any] = {
        "alias_exact_hit": bool(best_exact),
        "alias_similarity": round(float(best_sim), 4) if best_sim >= 0 else None,
        "alias_margin": round(float(margin), 4),
        "alias_rank": int(rank),
        "alias_decision": str(best_tier),
        "alias_effective_score": round(float(max(best_eff, 0.0)), 4),
        "alias_artifact_error": (_ALIAS_ARTIFACTS_ERROR or None),
        "alias_rescue_env": bool(rescue_env),
    }

    if _legacy_shadow_enabled():
        try:
            import alias_scorer_legacy_shadow as _leg

            old_ch, old_sc = _leg.corpus_alias_leader_legacy(q, client_id=client_id)
            new_key = _leg.legacy_chunk_key(best_ch)
            old_key = _leg.legacy_chunk_key(old_ch)
            diag["old_alias_score"] = round(float(old_sc or 0.0), 4)
            diag["old_alias_pick"] = (
                {
                    "file": old_ch.get("file") if isinstance(old_ch, dict) else None,
                    "h2_id": old_ch.get("h2_id") if isinstance(old_ch, dict) else None,
                    "h3_id": old_ch.get("h3_id") if isinstance(old_ch, dict) else None,
                }
                if isinstance(old_ch, dict)
                else None
            )
            diag["new_alias_pick"] = (
                {
                    "file": best_ch.get("file") if isinstance(best_ch, dict) else None,
                    "h2_id": best_ch.get("h2_id") if isinstance(best_ch, dict) else None,
                    "h3_id": best_ch.get("h3_id") if isinstance(best_ch, dict) else None,
                }
                if isinstance(best_ch, dict)
                else None
            )
            diag["old_vs_new_changed_decision"] = bool(new_key != old_key)
        except Exception as e:
            diag["old_vs_new_changed_decision"] = None
            diag["alias_legacy_shadow_error"] = str(e)[:200]

    eff_out = round(float(max(best_eff, 0.0)), 4)
    if isinstance(best_ch, dict):
        best_ch["_alias_score"] = eff_out
        best_ch["_score"] = eff_out
        best_ch["_alias_decision"] = str(best_tier)
        best_ch["_alias_similarity"] = round(float(best_sim), 4) if best_sim >= 0 else None
        h3 = (best_ch.get("h3_id") or best_ch.get("h2_id") or "korotko") or "korotko"
        diag["alias_candidate_ref"] = f"{best_ch.get('file')}#{h3}"

    log_json(
        logger,
        "alias_pipeline_result",
        client_id=client_id,
        query_preview=q_raw[:200],
        **{k: v for k, v in diag.items() if isinstance(v, (str, int, float, bool, type(None)))},
    )
    return {
        "leader": best_ch,
        "effective_score": eff_out,
        "diag": diag,
    }


def corpus_alias_leader(
    q: str,
    *,
    client_id: str | None = None,
) -> tuple[dict | None, float, dict[str, Any]]:
    """Best corpus chunk by alias pipeline and diagnostics dict (third element)."""
    r = run_alias_pipeline(q, client_id=client_id)
    return r["leader"], float(r["effective_score"]), dict(r["diag"])


def alias_debug_score_for_chunk(q: str, ch: dict, *, client_id: str | None = None) -> dict[str, Any]:
    """Debug-only: alias components for one chunk (used by /__debug/retrieval)."""
    q_raw = (q or "").strip()
    q_norm = _norm_text(q)
    corpus = load_corpus_if_needed()
    _load_alias_embed_artifacts()
    pos = _corpus_index_for_chunk(corpus, ch)
    n = len(corpus)
    chunk_max = np.full(n, -1.0, dtype=np.float32)
    q_embed = normalize_retrieval_query(q_raw) or q_raw
    if (
        pos is not None
        and _ALIAS_EMB_MATRIX is not None
        and int(_ALIAS_EMB_MATRIX.shape[0]) > 0
        and q_embed.strip()
    ):
        try:
            v = embed_q(q_embed)
            sims_rows = _ALIAS_EMB_MATRIX @ v
            row_ok = np.ones(sims_rows.shape[0], dtype=bool)
            if client_id:
                row_ok = np.array(
                    [(not cid) or (cid == client_id) for cid in (_ALIAS_ROW_CLIENT or [])],
                    dtype=bool,
                )
            sims_f = sims_rows.astype(np.float32).copy()
            sims_f[~row_ok] = -1.0
            np.maximum.at(chunk_max, _ALIAS_ROW_CORPUS_IDX, sims_f)
        except Exception:
            pass
    emb_val = float(chunk_max[pos]) if pos is not None else -1.0
    det, exact_hit, det_tier = _deterministic_alias_for_chunk(q_norm, ch)
    eff, tier = _classify_alias_tier_for_chunk(
        q_norm=q_norm,
        ch=ch,
        emb_sim=emb_val,
        thr=THRESHOLDS.alias,
        rescue_env=False,
        top_emb_corpus_idx=None,
        corpus_pos=pos,
    )
    return {
        "alias_effective": round(float(eff), 4),
        "alias_tier": tier,
        "alias_exact_hit": bool(exact_hit),
        "alias_det_score": round(float(det), 4),
        "alias_det_tier": det_tier,
        "alias_emb_sim": round(float(emb_val), 4) if emb_val >= 0 else None,
    }


def best_alias_hit_in_corpus(
    q: str,
    *,
    client_id: str | None = None,
    strong_threshold: float | None = None,
) -> tuple[dict | None, float]:
    thr_val = (
        float(THRESHOLDS.alias.strong_effective_min)
        if strong_threshold is None
        else float(strong_threshold)
    )
    leader, score, _diag = corpus_alias_leader(q, client_id=client_id)
    if leader and score >= thr_val:
        chosen = dict(leader)
        chosen["_alias_score"] = round(score, 4)
        chosen["_score"] = round(score, 4)
        return chosen, score
    return None, score


def is_point_literal_query(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    qn = _norm_text(q)
    tokens = [t for t in qn.split() if t]
    if not tokens:
        return False
    if any(ch.isdigit() for ch in q):
        return True
    if len(tokens) <= 4:
        question_words = {"как", "что", "почему", "зачем", "когда", "какие", "какой", "какая"}
        if not any(t in question_words for t in tokens):
            return True
    return False


def embed_q(q: str) -> np.ndarray:
    v = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
    v = np.array(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v


def merge_retrieval_candidates(*lists: list) -> list:
    """Объединить несколько списков кандидатов из retrieve, для каждого чанка оставить больший _score."""
    best: dict[tuple, dict] = {}
    for lst in lists:
        if not lst:
            continue
        for c in lst:
            key = (
                c.get("file"),
                c.get("h2_id") or c.get("h2"),
                c.get("h3_id") or c.get("h3"),
            )
            sc = float(c.get("_score") or 0.0)
            prev = best.get(key)
            if prev is None or sc > float(prev.get("_score") or 0.0):
                best[key] = dict(c)
    return sorted(best.values(), key=lambda x: float(x.get("_score") or 0.0), reverse=True)


def _active_scope_topic(scope_topic: str | None) -> str | None:
    """Return normalized topic slug for retrieval scope, or None for full corpus."""
    if scope_topic is None:
        return None
    st = str(scope_topic).strip().lower()
    if not st or st == "unknown":
        return None
    return st


def _corpus_indices_for_scope_topic(
    corpus: list, *, scope_slug: str, client_id: str | None
) -> list[int]:
    """Row indices matching frontmatter-derived chunk topic; excludes chunks without topic."""
    out_idx: list[int] = []
    want = scope_slug.strip().lower()
    for i, c in enumerate(corpus):
        if not isinstance(c, dict):
            continue
        if client_id and c.get("client_id") != client_id:
            continue
        ct = c.get("topic")
        if ct is None or str(ct).strip() == "":
            continue
        if str(ct).strip().lower() != want:
            continue
        out_idx.append(int(i))
    return out_idx


def _gather_retrieval_candidates(
    *,
    emb: np.ndarray,
    corpus: list,
    v: np.ndarray,
    topk: int,
    client_id: str | None,
    scoped_indices: list[int] | None,
) -> list[dict]:
    """Cosine-ranked chunks; scoped_indices=None searches full corpus."""
    out: list[dict] = []
    seen: set[tuple[Any, Any, Any]] = set()

    if scoped_indices:
        ix = np.array(scoped_indices, dtype=np.intp)
        sub = emb[ix]
        sims_local = sub @ v
        ord_local = np.argsort(-sims_local)[: max(topk, 8)]
        for li in ord_local:
            global_i = int(ix[int(li)])
            c = corpus[global_i]
            if client_id and c.get("client_id") != client_id:
                continue
            key = (c["file"], c.get("h2_id") or c.get("h2"), c.get("h3_id") or c.get("h3"))
            if key in seen:
                continue
            seen.add(key)
            c2 = dict(c)
            c2["_score"] = float(sims_local[int(li)])
            out.append(c2)
            if len(out) == topk:
                break
        return out

    sims = emb @ v
    idx = np.argsort(-sims)[: max(topk, 8)]
    for i in idx:
        c = corpus[int(i)]
        if client_id and c.get("client_id") != client_id:
            continue
        key = (c["file"], c.get("h2_id") or c.get("h2"), c.get("h3_id") or c.get("h3"))
        if key in seen:
            continue
        seen.add(key)
        c2 = dict(c)
        c2["_score"] = float(sims[int(i)])
        out.append(c2)
        if len(out) == topk:
            break
    return out


def retrieve(
    q: str,
    topk: int = 4,
    *,
    client_id: str | None = None,
    silent: bool = False,
    scope_topic: str | None = None,
    telemetry: dict[str, Any] | None = None,
) -> list:
    if telemetry is not None:
        telemetry.setdefault("scope_widen_fallback", False)

    emb = _get_embeddings()
    q_in = (q or "").strip()
    q_norm = normalize_retrieval_query(q_in)
    q_embed = q_norm if q_norm else q_in
    if not q_embed:
        log_json(
            logger,
            "retrieval_skipped_empty_query",
            query_raw=q_in[:200],
            used_query="",
        )
        return []
    if emb is None:
        log_json(
            logger,
            "retrieval_skipped_no_embeddings",
            used_query=q_embed[:500],
            query_raw=q_in[:200],
            emb_path=EMB_PATH,
        )
        return []

    scope_key = ""
    applied_scope = _active_scope_topic(scope_topic)
    if applied_scope:
        scope_key = applied_scope

    cache_key = (q_embed, int(topk), scope_key, str(client_id or ""))
    now = time.time()
    with _RETRIEVE_CACHE_LOCK:
        cached = _RETRIEVE_CACHE.get(cache_key)
        if cached and (now - float(cached[0]) <= RETRIEVE_CACHE_TTL_SEC):
            out_cached = [dict(item) for item in (cached[1] or [])]
            if telemetry is not None:
                # Cached paths are stored only without widen fallback (see below).
                telemetry["scope_widen_fallback"] = bool(telemetry.get("scope_widen_fallback"))
            if not silent:
                log_json(
                    logger,
                    "retrieval_cache_hit",
                    used_query=q_embed[:500],
                    query_raw=q_in[:500],
                    query_normalized=(q_norm[:500] if q_norm else None),
                    k=topk,
                    client_id=client_id,
                    scope_topic=scope_key or None,
                    size=len(out_cached),
                )
            return out_cached

    corpus = load_corpus_if_needed()
    v = embed_q(q_embed)
    widen_used = False
    prior_scope: str | None = str(scope_topic).strip() if scope_topic else None

    scoped_indices: list[int] | None = None
    if applied_scope:
        scoped_indices = _corpus_indices_for_scope_topic(
            corpus, scope_slug=applied_scope, client_id=client_id
        )
        if not scoped_indices:
            widen_used = True
            if telemetry is not None:
                telemetry["scope_widen_fallback"] = True
            log_json(
                logger,
                "retrieval_scope_widen_fallback",
                used_query=q_embed[:500],
                query_raw=q_in[:200],
                details={"prior_scope_topic": prior_scope},
            )
            scoped_indices = None

    out = _gather_retrieval_candidates(
        emb=emb,
        corpus=corpus,
        v=v,
        topk=int(topk),
        client_id=client_id,
        scoped_indices=scoped_indices,
    )

    if applied_scope and not widen_used and len(out) == 0:
        widen_used = True
        if telemetry is not None:
            telemetry["scope_widen_fallback"] = True
        log_json(
            logger,
            "retrieval_scope_widen_fallback",
            used_query=q_embed[:500],
            query_raw=q_in[:200],
            details={"prior_scope_topic": prior_scope},
        )
        out = _gather_retrieval_candidates(
            emb=emb,
            corpus=corpus,
            v=v,
            topk=int(topk),
            client_id=client_id,
            scoped_indices=None,
        )

    try:
        chunks_used = [chunk_info(item, item.get("_score")) for item in out[:topk]]
    except Exception:
        chunks_used = []

    if not silent:
        log_json(
            logger,
            "retrieval_result",
            used_query=q_embed[:500],
            query_raw=q_in[:500],
            query_normalized=(q_norm[:500] if q_norm else None),
            k=topk,
            scope_topic=(scope_key or None),
            scope_widen_fallback=bool(widen_used),
            dedup_keys=["file", "h2_id", "h3_id"],
            chunks_used=chunks_used,
            top_score=(chunks_used[0]["score"] if chunks_used else None),
        )
    with _RETRIEVE_CACHE_LOCK:
        # Avoid caching widen results: keyed query+scope must not replay full-corpus mixes.
        if not widen_used:
            _RETRIEVE_CACHE[cache_key] = (now, [dict(item) for item in out])

        if len(_RETRIEVE_CACHE) > max(32, int(RETRIEVE_CACHE_MAXSIZE)):
            stale_keys = sorted(_RETRIEVE_CACHE.items(), key=lambda kv: kv[1][0])
            drop_n = len(_RETRIEVE_CACHE) - int(RETRIEVE_CACHE_MAXSIZE)
            for i in range(max(0, drop_n)):
                _RETRIEVE_CACHE.pop(stale_keys[i][0], None)

    return out


def llm_rerank(q: str, cands: list) -> dict:
    t0 = time.time()
    try:
        cand_infos = [chunk_info(ch, ch.get("_score")) for ch in cands]
    except Exception:
        cand_infos = [chunk_info(ch, None) for ch in cands]
    log_json(
        logger,
        "rerank",
        question=q[:200],
        candidates=cand_infos,
        model_used=RERANK_MODEL,
    )

    prompt = (
        "Выбери самый уместный фрагмент для ответа на вопрос пользователя. "
        'Верни только JSON-объект вида {"choice": 1}, где choice — номер 1, 2 или 3.'
    )
    def _cand_block(ch: dict) -> str:
        if not isinstance(ch, dict):
            return ""
        return (
            f"{(ch.get('h2') or '').strip()}\n"
            f"{(ch.get('h3') or '').strip()}\n"
            f"{(ch.get('text') or '')[:500]}"
        ).strip()
    msgs = [
        {
            "role": "system",
            "content": (
                "Ты помощник стоматологической клиники. Тебе нужно выбрать фрагмент "
                "из базы знаний, который наиболее точно отвечает на вопрос пациента. "
                'Отвечай только JSON-объектом вида {"choice": 1}, где choice — 1, 2 или 3.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n\nВопрос: {q}\n\n"
                f"1) {_cand_block(cands[0])}\n\n"
                f"2) {_cand_block(cands[1]) if len(cands) > 1 else ''}\n\n"
                f"3) {_cand_block(cands[2]) if len(cands) > 2 else ''}"
            ),
        },
    ]
    fallback_reason = None
    try:
        out = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=msgs,
            temperature=0,
            response_format={"type": "json_object"},
        )
        log_llm_usage(logger, out, call_type="rerank", model=RERANK_MODEL)
        raw = (out.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
            fallback_reason = "invalid_json"
        if not isinstance(obj, dict):
            fallback_reason = fallback_reason or "invalid_json_object"
            result = cands[0]
        else:
            choice = obj.get("choice")
            if not isinstance(choice, int):
                fallback_reason = "missing_or_nonint_choice"
                result = cands[0]
            else:
                idx = int(choice) - 1
                max_idx = min(3, len(cands)) - 1
                if 0 <= idx <= max_idx:
                    result = cands[idx]
                else:
                    fallback_reason = "choice_out_of_range"
                    result = cands[0]
    except Exception:
        fallback_reason = "api_error"
        result = cands[0]

    lat = int((time.time() - t0) * 1000)
    log_json(
        logger,
        "rerank_result",
        model_used=RERANK_MODEL,
        latency_ms=lat,
        fallback_reason=fallback_reason,
        chosen=chunk_info(
            result, result.get("_score") if isinstance(result, dict) else None
        ),
    )

    return result
