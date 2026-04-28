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
YES_RX = re.compile(
    (
        r"^(?:да[\s,!.?-]*)?(да|ага|угу|ok|ок|хорошо|давай|хочу|расскажу|конечно|ладно|можно|"
        r"согласен|согласна|договорились|пожалуй)\W*$"
    ),
    re.I,
)

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
        "session_turn_count": 0,
        "current_doc_id": None,
        "last_doc_key": None,
        "last_empathy_at": None,
        "turn_count": 0,
        "last_bot_action": "none",
        "last_offer_type": None,
        "last_presented_buttons": [],
        "situation_pending": False,
        "situation_note": "",
        "lead_intent": "none",
        "booking_intent_ever": False,
        "anti_spam_redirect_shown": False,
        "lead_pending_name": "",
        "shown_cta_topics": [],
        "topic_state": {},
        "last_content_ui_payload": None,
        "last_catalog_service_id": None,
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
        st["session_turn_count"] = int(st.get("session_turn_count") or 0) + 1
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
        sit = payload.get("situation") or {}
        buttons = []
        for x in fup[:2]:
            if isinstance(x, dict):
                buttons.append({"label": x.get("label"), "ref": x.get("ref")})
        for x in qr[:2]:
            if isinstance(x, dict):
                buttons.append({"label": x.get("label"), "ref": x.get("ref")})
        st["last_presented_buttons"] = buttons[:6]
        if sit.get("show") and sit.get("mode") == "pending":
            st["last_bot_action"] = "situation_collect"
            st["last_offer_type"] = "situation"
        elif sit.get("show") and sit.get("mode") == "normal":
            st["last_bot_action"] = "offered_situation"
            st["last_offer_type"] = "situation"
        elif cta:
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

        if (
            not st.get("situation_pending")
            and not meta.get("lead_flow")
            and not meta.get("situation_collect")
            and not meta.get("low_score")
            and not meta.get("error")
        ):
            st["last_content_ui_payload"] = {
                "answer": payload.get("answer"),
                "quick_replies": list(payload.get("quick_replies") or []),
                "cta": payload.get("cta"),
                "video": payload.get("video"),
                "situation": dict(sit) if isinstance(sit, dict) else {"show": False, "mode": "normal"},
                "offer": payload.get("offer"),
                "meta": json.loads(json.dumps(meta, ensure_ascii=False)),
            }
        _persist_unlocked(session_id, st)


def get_last_content_ui_payload(session_id: str) -> dict | None:
    st = mem_get(session_id)
    snap = st.get("last_content_ui_payload")
    return snap if isinstance(snap, dict) else None


def is_active_lead_flow(session_state: dict) -> bool:
    return (session_state or {}).get("lead_intent") in {
        "collecting_name",
        "collecting_phone",
        "confirming_name",
    }


def _topic_state_container(st: dict) -> dict:
    ts = st.get("topic_state")
    if not isinstance(ts, dict):
        ts = {}
        st["topic_state"] = ts
    return ts


def get_topic_state(session_id: str, doc_id: str) -> dict:
    st = mem_get(session_id)
    ts = _topic_state_container(st)
    topic = ts.get(doc_id) or {}
    return {
        "doc_turn_count": int(topic.get("doc_turn_count") or 0),
        "covered_h3_ids": list(topic.get("covered_h3_ids") or []),
        "video_shown": bool(topic.get("video_shown", False)),
        "video_pending": bool(topic.get("video_pending", False)),
        "situation_offered": bool(topic.get("situation_offered", False)),
        "suggest_ref_used": bool(topic.get("suggest_ref_used", False)),
        "refs_deferred": list(topic.get("refs_deferred") or []),
        "cta_shown": bool(topic.get("cta_shown", False)),
    }


def _upsert_topic_state(st: dict, doc_id: str, patch: dict) -> None:
    ts = _topic_state_container(st)
    cur = ts.get(doc_id) or {
        "doc_turn_count": 0,
        "covered_h3_ids": [],
        "video_shown": False,
        "video_pending": False,
        "situation_offered": False,
        "suggest_ref_used": False,
        "refs_deferred": [],
        "cta_shown": False,
    }
    cur.update(patch or {})
    ts[doc_id] = cur


def set_last_catalog_service(session_id: str, service_id: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["last_catalog_service_id"] = (service_id or "").strip() or None
        _persist_unlocked(session_id, st)


def set_current_doc(session_id: str, doc_id: str) -> None:
    with _lock:
        st = mem_get(session_id)
        prev = (st.get("current_doc_id") or "").strip()
        new_id = (doc_id or "").strip()
        if new_id and prev and prev != new_id:
            _upsert_topic_state(st, new_id, {"suggest_ref_used": False})
        st["current_doc_id"] = doc_id
        _persist_unlocked(session_id, st)


def mark_h3_covered(session_id: str, doc_id: str, h3_id: str) -> None:
    if not doc_id or not h3_id:
        return
    with _lock:
        st = mem_get(session_id)
        cur = get_topic_state(session_id, doc_id)
        covered = list(cur.get("covered_h3_ids") or [])
        if h3_id not in covered:
            covered.append(h3_id)
        _upsert_topic_state(st, doc_id, {"covered_h3_ids": covered})
        _persist_unlocked(session_id, st)


def increment_doc_turn_if_contentful(
    session_id: str,
    doc_id: str,
    *,
    contentful: bool,
    is_low_score: bool,
    is_error: bool,
    lead_flow_active: bool,
) -> int | None:
    if not doc_id:
        return None
    if not contentful or is_low_score or is_error or lead_flow_active:
        return None
    with _lock:
        st = mem_get(session_id)
        cur = get_topic_state(session_id, doc_id)
        prev = int(cur.get("doc_turn_count") or 0)
        new_count = prev + 1
        _upsert_topic_state(st, doc_id, {"doc_turn_count": new_count})
        _persist_unlocked(session_id, st)
        return prev


def mark_video_pending(session_id: str, doc_id: str, pending: bool = True) -> None:
    with _lock:
        st = mem_get(session_id)
        _upsert_topic_state(st, doc_id, {"video_pending": bool(pending)})
        _persist_unlocked(session_id, st)


def mark_video_shown(session_id: str, doc_id: str) -> None:
    with _lock:
        st = mem_get(session_id)
        _upsert_topic_state(st, doc_id, {"video_shown": True, "video_pending": False})
        _persist_unlocked(session_id, st)


def mark_situation_offered(session_id: str, doc_id: str) -> None:
    with _lock:
        st = mem_get(session_id)
        _upsert_topic_state(st, doc_id, {"situation_offered": True})
        _persist_unlocked(session_id, st)


def mark_suggest_ref_used(session_id: str, doc_id: str, used: bool = True) -> None:
    with _lock:
        st = mem_get(session_id)
        _upsert_topic_state(st, doc_id, {"suggest_ref_used": bool(used)})
        _persist_unlocked(session_id, st)


def set_cta_shown(session_id: str, doc_id: str, shown: bool = True) -> None:
    with _lock:
        st = mem_get(session_id)
        # analytics/memory flag only; not a hard blocker
        _upsert_topic_state(st, doc_id, {"cta_shown": bool(shown)})
        _persist_unlocked(session_id, st)


def defer_refs(session_id: str, doc_id: str, refs: list[dict]) -> None:
    with _lock:
        st = mem_get(session_id)
        cur = get_topic_state(session_id, doc_id)
        existing = list(cur.get("refs_deferred") or [])
        for r in refs or []:
            if isinstance(r, dict) and r.get("ref") and r not in existing:
                existing.append(r)
        _upsert_topic_state(st, doc_id, {"refs_deferred": existing})
        _persist_unlocked(session_id, st)


def pop_deferred_ref(session_id: str, doc_id: str) -> dict | None:
    with _lock:
        st = mem_get(session_id)
        cur = get_topic_state(session_id, doc_id)
        refs = list(cur.get("refs_deferred") or [])
        if not refs:
            return None
        first = refs[0]
        _upsert_topic_state(st, doc_id, {"refs_deferred": refs[1:]})
        _persist_unlocked(session_id, st)
        return first


def set_situation_pending(session_id: str, pending: bool = True) -> None:
    with _lock:
        st = mem_get(session_id)
        st["situation_pending"] = bool(pending)
        _persist_unlocked(session_id, st)


def set_situation_note(session_id: str, note: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["situation_note"] = (note or "").strip()
        _persist_unlocked(session_id, st)


def set_lead_intent(session_id: str, intent: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["lead_intent"] = intent
        _persist_unlocked(session_id, st)


def mark_booking_intent_ever(session_id: str) -> None:
    with _lock:
        st = mem_get(session_id)
        st["booking_intent_ever"] = True
        _persist_unlocked(session_id, st)


def set_anti_spam_redirect_shown(session_id: str, shown: bool = True) -> None:
    with _lock:
        st = mem_get(session_id)
        st["anti_spam_redirect_shown"] = bool(shown)
        _persist_unlocked(session_id, st)


def parse_yes(text: str) -> bool:
    return bool(YES_RX.search((text or "").strip()))


def parse_no(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) > 48:
        return False
    return bool(
        re.fullmatch(
            r"(нет|неа|ну\s+нет|не\s+так|другой|другая|другим|по[- ]другому|изменить|введу\s+заново|отмена)[\s,.!?-]*",
            t,
            flags=re.I,
        )
    )


def set_lead_pending_name(session_id: str, name: str | None) -> None:
    with _lock:
        st = mem_get(session_id)
        v = (name or "").strip()
        if v:
            st["lead_pending_name"] = v
        else:
            st["lead_pending_name"] = ""
        _persist_unlocked(session_id, st)


def get_lead_pending_name(session_id: str) -> str:
    st = mem_get(session_id)
    return (st.get("lead_pending_name") or "").strip()


# Токены, которые нельзя принимать за имя после «я …» / однословный ответ.
_LEAD_NAME_REJECT = frozenset(
    {
        "боюсь", "хочу", "переживаю", "переживал", "переживала", "беспокоюсь",
        "думаю", "знаю", "понимаю", "слышал", "слышала", "видел", "видела",
        "устал", "устала", "устали", "надеюсь", "сомневаюсь",
        "хотел", "хотела", "хотели", "хотелось", "узнать", "узнаю", "спрашиваю",
        "бы", "же", "не", "тоже", "также", "просто", "очень", "уже", "ещё",
        "еще", "пока", "только", "да", "нет", "ок", "ага", "угу",
        "хорошо", "ладно", "спасибо", "привет", "здравствуйте", "понятно",
        "ясно", "конечно", "извините", "простите",
    }
)

_NAME_TOKEN_RX = re.compile(r"^[А-ЯЁA-Za-zа-яё\-]{2,40}$", re.U)


def _strip_lead_name_filler(s: str) -> str:
    return re.sub(
        r"^(?:[ауоыэи]+\s+|ну\s+|а\s+|э\s+|эм\s+)+",
        "",
        (s or "").strip(),
        flags=re.I,
    ).strip()


def _token_ok_for_name(tok: str) -> bool:
    t = (tok or "").strip()
    if not t or not _NAME_TOKEN_RX.fullmatch(t):
        return False
    if t.lower() in _LEAD_NAME_REJECT:
        return False
    return True


def extract_name(text: str) -> str | None:
    s0 = (text or "").strip()
    if not s0:
        return None
    s = _strip_lead_name_filler(s0)
    if not s:
        return None

    m = re.match(
        r"^меня\s+зовут\s+"
        r"([А-ЯЁA-Za-zа-яё\-]+(?:\s+[А-ЯЁA-Za-zа-яё\-]+){0,2})\s*$",
        s,
        re.I | re.U,
    )
    if m:
        parts = m.group(1).split()
        if not parts or not all(_token_ok_for_name(p) for p in parts):
            return None
        return " ".join(p[:1].upper() + p[1:].lower() if len(p) > 1 else p.capitalize() for p in parts)

    m_ya = re.match(r"^я\s+([А-ЯЁA-Za-zа-яё\-]+)\s*$", s, re.I | re.U)
    if m_ya:
        tok = m_ya.group(1)
        if not _token_ok_for_name(tok):
            return None
        t = tok
        return (t[:1].upper() + t[1:].lower()) if len(t) > 1 else t.capitalize()

    if re.fullmatch(r"[А-ЯЁA-Za-zа-яё\-]{2,40}", s, re.U):
        if not _token_ok_for_name(s):
            return None
        t = s
        return (t[:1].upper() + t[1:].lower()) if len(t) > 1 else t.capitalize()

    parts = s.split()
    if len(parts) == 2 and all(_token_ok_for_name(p) for p in parts):
        return " ".join(
            (p[:1].upper() + p[1:].lower()) if len(p) > 1 else p.capitalize()
            for p in parts
        )

    return None


def extract_phone(text: str) -> str | None:
    """Допускает ввод с маской +7(###) ###-##-## и любые нецифровые разделители."""
    return normalize_phone(text or "")


def normalize_phone(text: str) -> str | None:
    raw = re.sub(r"\D", "", text or "")
    if len(raw) == 11 and raw.startswith("8"):
        raw = "7" + raw[1:]
    if len(raw) == 10:
        raw = "7" + raw
    if len(raw) != 11:
        return None
    return f"+{raw}"


def update_profile(session_id: str, **fields) -> None:
    with _lock:
        st = mem_get(session_id)
        prof = dict(st.get("profile") or {})
        for k, v in (fields or {}).items():
            if v:
                prof[k] = v
        st["profile"] = prof
        _persist_unlocked(session_id, st)
