# meta_loader.py
from os.path import abspath, basename
import os, re, yaml

from config import DEFAULT_CLIENT_ID

# front-matter между --- ... ---
_FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.S)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _parse_front_matter(text: str) -> dict:
    m = _FM_RE.match(text)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            return {}
        return fm
    except Exception:
        return {}

def load_doc_meta(md_root: str = "md") -> dict:
    meta = {}
    for root, _, files in os.walk(md_root):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, md_root)
            parts = rel.split(os.sep)
            client_id = parts[0] if len(parts) > 1 else None
            if client_id:
                _DOC_PATHS[(client_id, basename(name))] = abspath(path)
            else:
                _DOC_PATHS[basename(name)] = abspath(path)
            fm = _parse_front_matter(_read_file(path))
            item = {
                "client_id": client_id,
                "doc_id": fm.get("doc_id"),
                "doc_type": fm.get("doc_type"),
                "subtype": fm.get("subtype"),
                "topic": fm.get("topic"),
                "subtopic": fm.get("subtopic"),
                "cta_text": fm.get("cta_text"),
                "cta_action": fm.get("cta_action"),
                "cta_from_turn": _safe_int(fm.get("cta_from_turn", 0), 0),
                "verbatim_ids": fm.get("verbatim_ids") or [],
                "aliases": fm.get("aliases") or [],
                "suggest_h3": fm.get("suggest_h3") or [],
                "suggest_refs": fm.get("suggest_refs") or [],
                "situation_allowed": bool(fm.get("situation_allowed", False)),
                "video_key": fm.get("video_key"),
                # ↓↓↓ ЭМПАТИЯ ↓↓↓
                "empathy_enabled": bool(fm.get("empathy_enabled", False)),
                "empathy_tag": fm.get("empathy_tag"),
            }
            meta[(client_id, name)] = item
            if client_id is None:
                meta[name] = item
    return meta

_DOC_META = None
_DOC_PATHS = {}  # basename -> абсолютный путь

def get_doc_path(doc_name: str, client_id: str | None = None):
    global _DOC_PATHS, _DOC_META
    if not _DOC_PATHS:
        _DOC_META = load_doc_meta()
    if client_id:
        item = _DOC_PATHS.get((client_id, doc_name))
        if item:
            return item
        # Single-client compatibility mode: keep root-level md/ files working
        # for the default tenant until content is moved under md/{client_id}/.
        if client_id == DEFAULT_CLIENT_ID:
            return _DOC_PATHS.get(doc_name)
    return _DOC_PATHS.get(doc_name)

def get_doc_meta(doc_name: str, client_id: str | None = None) -> dict:
    """doc_name — basename файла, например 'clinic-contacts.md'"""
    global _DOC_META
    if _DOC_META is None:
        _DOC_META = load_doc_meta()
    if client_id:
        item = _DOC_META.get((client_id, doc_name))
        if item:
            return item
        wanted_doc_id = os.path.splitext(doc_name)[0]
        for _, cand in _DOC_META.items():
            if not isinstance(cand, dict):
                continue
            if cand.get("client_id") == client_id and cand.get("doc_id") == wanted_doc_id:
                return cand
        if client_id == DEFAULT_CLIENT_ID:
            item = _DOC_META.get(doc_name)
            if item:
                return item
            for _, cand in _DOC_META.items():
                if not isinstance(cand, dict):
                    continue
                if cand.get("client_id") is None and cand.get("doc_id") == wanted_doc_id:
                    return cand
        return {}
    return _DOC_META.get(doc_name, {})
