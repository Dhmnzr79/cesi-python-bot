"""Состояние сессии: история, профиль, эмпатия, поля для policy — в SQLite."""
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime

from config import DATA_DIR, MAX_IDLE_SEC, MAX_TURNS, SQLITE_PATH

PHONE_RX = re.compile(r"(?:\+7|8)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-()]?\d{2}[\s\-()]?\d{2}")

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            _conn = sqlite3.connect(
                SQLITE_PATH, check_same_thread=False, isolation_level=None
            )
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    sid TEXT PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
        return _conn


def _fresh_defaults() -> dict:
    return {
        "hist": deque(maxlen=MAX_TURNS * 2),
        "profile": {},
        "ts": time.time(),
        "last_doc_key": None,
        "last_empathy_at": None,
        "turn_count": 0,
        "last_bot_action": "none",
        "last_offer_type": None,
        "last_presented_buttons": [],
        "situation_pending": False,
        "lead_intent": None,
        "shown_cta_topics": [],
    }


def _deserialize_row(payload_json: str) -> dict:
    raw = json.loads(payload_json)
    st = _fresh_defaults()
    for k, v in raw.items():
        if k == "hist" and isinstance(v, list):
            st["hist"] = deque(v, maxlen=MAX_TURNS * 2)
        else:
            st[k] = v
    return st


def _serialize_state(st: dict) -> str:
    d = {k: (list(v) if k == "hist" else v) for k, v in st.items()}
    return json.dumps(d, ensure_ascii=False)


def _persist_unlocked(sid: str, st: dict) -> None:
    st["ts"] = time.time()
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (sid, updated_at, payload) VALUES (?,?,?)",
        (sid, st["ts"], _serialize_state(st)),
    )


def _now() -> float:
    return time.time()


def sid_from_body(body: dict) -> str:
    sid = (body or {}).get("sid") or ""
    sid = str(sid).strip()
    return sid or uuid.uuid4().hex


def mem_get(session_id: str) -> dict:
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT payload, updated_at FROM sessions WHERE sid = ?",
            (session_id,),
        ).fetchone()
        if not row:
            st = _fresh_defaults()
            _persist_unlocked(session_id, st)
            return st
        st = _deserialize_row(row[0])
        if _now() - float(st.get("ts") or 0) > MAX_IDLE_SEC:
            st = _fresh_defaults()
            _persist_unlocked(session_id, st)
            return st
        return st


def mem_add_user(session_id: str, text: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["hist"].append({"role": "user", "content": text})
        st["turn_count"] = int(st.get("turn_count") or 0) + 1
        m = PHONE_RX.search(text)
        if m:
            st["profile"]["phone"] = m.group().replace(" ", "")
        if "меня зовут" in text.lower():
            parts = text.lower().split("меня зовут", 1)
            if len(parts) > 1:
                name_parts = parts[1].strip().split()
                if name_parts:
                    name = name_parts[0]
                    if name:
                        st["profile"]["name"] = name.capitalize()
        _persist_unlocked(session_id, st)


def mem_add_bot(session_id: str, text: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["hist"].append({"role": "assistant", "content": text})
        _persist_unlocked(session_id, st)


def mem_context(session_id: str) -> tuple[str, dict]:
    st = mem_get(session_id)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in list(st["hist"]))
    return (f"Недавний диалог:\n{history}" if history else ""), st["profile"]


def mem_reset(session_id: str) -> None:
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM sessions WHERE sid = ?", (session_id,))


def is_first_in_topic(session_id: str, doc_key: str) -> bool:
    st = mem_get(session_id)
    return st.get("last_doc_key") != doc_key


def update_topic_empathy(session_id: str, doc_key: str, empathy_used: bool) -> None:
    with _lock:
        st = mem_get(session_id)
        st["last_doc_key"] = doc_key
        if empathy_used:
            st["last_empathy_at"] = datetime.utcnow().isoformat()
        _persist_unlocked(session_id, st)


def record_last_bot_payload(session_id: str, payload: dict) -> None:
    """После policy: фиксируем last_bot_action и кнопки для трактовки «да» и т.д."""
    with _lock:
        st = mem_get(session_id)
        meta = payload.get("meta") or {}
        cta = payload.get("cta")
        fup = meta.get("followups") or []
        qr = payload.get("quick_replies") or []
        buttons = []
        for x in fup[:2]:
            if isinstance(x, dict):
                buttons.append({"label": x.get("label"), "ref": x.get("ref")})
        for x in qr[:2]:
            if isinstance(x, dict):
                buttons.append({"label": x.get("label"), "ref": x.get("ref")})
        st["last_presented_buttons"] = buttons[:6]
        if cta:
            st["last_bot_action"] = "offered_cta"
            st["last_offer_type"] = "cta"
        elif fup:
            st["last_bot_action"] = "offered_subtopic"
            st["last_offer_type"] = "followup"
        elif qr:
            st["last_bot_action"] = "offered_subtopic"
            st["last_offer_type"] = "quick_reply"
        else:
            st["last_bot_action"] = "none"
            st["last_offer_type"] = None
        _persist_unlocked(session_id, st)
