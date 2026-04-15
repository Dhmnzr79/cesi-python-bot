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
)
from llm import client
from logging_setup import get_logger, log_json
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


def load_corpus_if_needed() -> list:
    global _CORPUS
    if _CORPUS is None:
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                _CORPUS = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            _CORPUS = []
    return _CORPUS


EMB = np.load(EMB_PATH)


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


def embed_q(q: str) -> np.ndarray:
    v = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
    v = np.array(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v


def retrieve(q: str, topk: int = 4, *, client_id: str | None = None) -> list:
    v = embed_q(q)
    sims = EMB @ v
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
        used_query=q,
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
