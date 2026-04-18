"""Индекс, эмбеддинги, поиск, rerank; метаданные чанков."""
import json
import os
import re
import time
from typing import Any

import numpy as np

from config import (
    BROAD_QUERY_MAX_WORDS,
    CHAT_MODEL,
    CORPUS_PATH,
    EMB_PATH,
    EMB_MODEL,
    ALIAS_STRONG_THRESHOLD,
)
from llm import client
from logging_setup import get_logger, log_json
from meta_loader import get_doc_meta, get_doc_path

import alias_lexical

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


def load_corpus_if_needed() -> list:
    global _CORPUS
    if _CORPUS is None:
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                _CORPUS = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            _CORPUS = []
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
    with open(md_path, "r", encoding="utf-8") as f:
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
    try:
        return getattr(item[0], "meta", {}).get("doc_type") or None
    except Exception:
        pass
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


def _strong_core_tokens(core: list[str]) -> list[str]:
    """«Сильные» токены: длина >= 3 или есть цифры (адрес, сумма)."""
    return [t for t in core if len(t) >= 3 or any(ch.isdigit() for ch in t)]


def _all_tokens_in_text(tokens: list[str], an: str) -> bool:
    """Каждый токен — отдельное слово в тексте (границы по пробелам)."""
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

        # --- Ядро: подстрока целиком (быстрый путь для «налоговый вычет» vs длинный alias) ---
        if q_core_joint and len(q_core_joint) >= 4 and q_core_joint in an:
            best = max(best, 0.92)
        if a_core_joint and len(a_core_joint) >= 4 and a_core_joint in qn:
            best = max(best, 0.92)

        # --- Короткое точечное ядро: 2 «сильных» токена запроса оба есть в alias как слова ---
        strong_q = _strong_core_tokens(q_core)
        if len(strong_q) == 2 and _all_tokens_in_text(strong_q, an):
            best = max(best, 0.9)
        # Два любых токена ядра (после стоп-слов), если ядро ровно из двух слов ---
        if len(q_core) == 2 and all(len(t) >= 2 for t in q_core) and _all_tokens_in_text(q_core, an):
            best = max(best, 0.9)

        # --- 2–3 токена ядра полностью покрыты alias-ядром (без требования почти полной фразы) ---
        if 2 <= len(q_core) <= 3 and q_core_set and q_core_set.issubset(a_core_set):
            best = max(best, 0.88 if len(q_core) == 3 else 0.9)

        # --- Пересечение ядер (мягче, чем только полный qn) ---
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
    """Совпадение по целым токенам (последовательность), не подстрока внутри одного слова.

    Иначе лемма «имплант» из алиаса попадает внутрь «имплантолог» в запросе и даёт ложный strong-alias.
    """
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
    """Склонения: max с raw; pymorphy3 при наличии, иначе fallback на lower."""
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
    """Опечатки / близкие формы по триграммам (не заменяет raw/lemma).

    Для короткого запроса и длинного алиаса целая строка даёт низкий Jaccard;
    дополнительно сравниваем запрос с **отдельными словами** алиаса (парковку vs парковка).
    """
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


def alias_hit_score_for_chunk(q: str, ch: dict) -> float:
    raw = _alias_hit_score_raw_for_chunk(q, ch)
    lem = _lemma_alias_channel(q, ch)
    tri = _trigram_alias_channel(q, ch)
    return round(max(raw, lem, tri), 4)


def best_alias_hit(q: str, cands: list, *, strong_threshold: float = 0.9) -> tuple[dict | None, float]:
    best_chunk = None
    best_score = 0.0
    for ch in cands or []:
        sc = alias_hit_score_for_chunk(q, ch)
        if sc > best_score:
            best_score = sc
            best_chunk = ch
    if best_score >= strong_threshold:
        return best_chunk, best_score
    return None, best_score


def corpus_alias_leader(
    q: str,
    *,
    client_id: str | None = None,
) -> tuple[dict | None, float]:
    """Лучший чанк по алиасам и его score (без порога)."""
    corpus = load_corpus_if_needed()
    best_chunk = None
    best_score = 0.0
    for ch in corpus:
        if client_id and ch.get("client_id") != client_id:
            continue
        sc = alias_hit_score_for_chunk(q, ch)
        if sc > best_score:
            best_score = sc
            best_chunk = ch
    if not best_chunk:
        return None, 0.0
    return dict(best_chunk), round(best_score, 4)


def best_alias_hit_in_corpus(
    q: str,
    *,
    client_id: str | None = None,
    strong_threshold: float | None = None,
) -> tuple[dict | None, float]:
    thr = ALIAS_STRONG_THRESHOLD if strong_threshold is None else strong_threshold
    leader, score = corpus_alias_leader(q, client_id=client_id)
    if leader and score >= thr:
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


def retrieve(q: str, topk: int = 4, *, client_id: str | None = None) -> list:
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
    v = embed_q(q_embed)
    sims = emb @ v
    idx = np.argsort(-sims)[: max(topk, 8)]
    seen, out = set(), []
    corpus = load_corpus_if_needed()
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

    try:
        chunks_used = [chunk_info(item, item.get("_score")) for item in out[:topk]]
    except Exception:
        chunks_used = []

    log_json(
        logger,
        "retrieval_result",
        used_query=q_embed[:500],
        query_raw=q_in[:500],
        query_normalized=(q_norm[:500] if q_norm else None),
        k=topk,
        dedup_keys=["file", "h2_id", "h3_id"],
        chunks_used=chunks_used,
        top_score=(chunks_used[0]["score"] if chunks_used else None),
    )

    return out


def llm_rerank(q: str, cands: list) -> dict:
    t0 = time.time()
    try:
        cand_infos = [chunk_info(ch, ch.get("_score")) for ch in cands]
    except Exception:
        cand_infos = [chunk_info(ch, None) for ch in cands]
    log_json(logger, "rerank", question=q[:200], candidates=cand_infos)

    prompt = (
        "Выбери самый уместный фрагмент для ответа на вопрос пользователя. Ответи номером 1, 2 или 3."
    )
    msgs = [
        {"role": "system", "content": "Ты выбираешь лучший фрагмент."},
        {
            "role": "user",
            "content": f"{prompt}\n\nВопрос: {q}\n\n1) {cands[0]['text'][:600]}\n\n2) {cands[1]['text'][:600] if len(cands) > 1 else ''}\n\n3) {cands[2]['text'][:600] if len(cands) > 2 else ''}",
        },
    ]
    try:
        out = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0)
        n = "".join([ch for ch in out.choices[0].message.content if ch.isdigit()])[:1]
        idx = int(n) - 1
        result = cands[idx] if 0 <= idx < len(cands) else cands[0]
    except Exception:
        result = cands[0]

    lat = int((time.time() - t0) * 1000)
    log_json(
        logger,
        "rerank_result",
        latency_ms=lat,
        chosen=chunk_info(
            result, result.get("_score") if isinstance(result, dict) else None
        ),
    )

    return result
