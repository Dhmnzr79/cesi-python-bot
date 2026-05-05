"""Non-blocking PostgreSQL sink for dashboard storage."""
from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timezone

_Q: queue.Queue[tuple[str, dict, int]] | None = None
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_LOGGER = None
_DSN = ""
_SINK_DISABLED = False
_QUEUE_MAX = int(os.getenv("BOT_PG_QUEUE_MAX", "5000"))
_DROP_WARN_EVERY = int(os.getenv("BOT_PG_DROP_WARN_EVERY", "100"))
_DROP_COUNT = 0
_MAX_RETRY = int(os.getenv("BOT_PG_MAX_RETRY", "3"))


def _log(level: str, msg: str, **fields) -> None:
    logger = _LOGGER
    if logger is None:
        return
    try:
        # Reuse structured logger when available.
        from logging_setup import log_json

        log_json(logger, msg, **fields)
    except Exception:
        try:
            getattr(logger, level, logger.info)(f"{msg} {fields}")
        except Exception:
            pass


def _parse_ts(ts: str | None):
    if not ts:
        return datetime.now(timezone.utc)
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_events (
                id BIGSERIAL PRIMARY KEY,
                occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                kind TEXT NOT NULL,
                event_type TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                request_id TEXT,
                sid TEXT,
                client_id TEXT,
                path TEXT,
                status TEXT,
                details JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bot_events_occurred_at
            ON bot_events (occurred_at DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bot_events_client_time
            ON bot_events (client_id, occurred_at DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bot_events_sid_time
            ON bot_events (sid, occurred_at ASC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bot_events_event_type_time
            ON bot_events (event_type, occurred_at DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_bot_events_request_id
            ON bot_events (request_id);
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id BIGSERIAL PRIMARY KEY,
                captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                request_id TEXT,
                sid TEXT,
                client_id TEXT,
                name TEXT,
                phone TEXT,
                topic TEXT,
                cta_action TEXT,
                turns_to_lead INTEGER,
                delivery_status TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_leads_captured_at
            ON leads (captured_at DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_leads_client_time
            ON leads (client_id, captured_at DESC);
            """
        )


def _insert_bot_event(conn, row: dict) -> None:
    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bot_events (
                occurred_at, kind, event_type, schema_version, request_id,
                sid, client_id, path, status, details
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                _parse_ts(row.get("ts")),
                str(row.get("kind") or "bot_event"),
                str(row.get("event_type") or "unknown"),
                int(row.get("schema_version") or 1),
                row.get("request_id"),
                row.get("sid") or row.get("session_id"),
                row.get("client_id"),
                row.get("path"),
                row.get("status"),
                Json(dict(row.get("details") or {})),
            ),
        )


def _insert_lead(conn, lead: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO leads (
                captured_at, request_id, sid, client_id, name, phone, topic,
                cta_action, turns_to_lead, delivery_status
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                _parse_ts(lead.get("captured_at")),
                lead.get("request_id"),
                lead.get("sid"),
                lead.get("client_id"),
                lead.get("name"),
                lead.get("phone"),
                lead.get("topic"),
                lead.get("cta_action"),
                lead.get("turns_to_lead"),
                lead.get("delivery_status"),
            ),
        )


def _worker() -> None:
    global _Q, _SINK_DISABLED
    assert _Q is not None
    try:
        import psycopg
    except Exception as e:
        _log("warning", "pg_sink_disabled_no_driver", err=str(e)[:200])
        _SINK_DISABLED = True
        _Q = None
        return

    while True:
        try:
            with psycopg.connect(_DSN, autocommit=True) as conn:
                _ensure_tables(conn)
                _log("info", "pg_sink_ready")
                while True:
                    item = _Q.get()
                    if item is None:
                        continue
                    kind, payload, retries = item
                    try:
                        if kind == "bot_event":
                            _insert_bot_event(conn, payload)
                        elif kind == "lead":
                            _insert_lead(conn, payload)
                    except Exception as e:
                        _log(
                            "warning",
                            "pg_sink_insert_failed",
                            kind=kind,
                            retry=retries,
                            err=str(e)[:300],
                        )
                        if retries < _MAX_RETRY and _Q is not None:
                            try:
                                _Q.put_nowait((kind, payload, retries + 1))
                            except queue.Full:
                                _log(
                                    "warning",
                                    "pg_sink_requeue_failed_queue_full",
                                    kind=kind,
                                    retry=retries,
                                )
        except Exception as e:
            _log("warning", "pg_sink_connect_failed", err=str(e)[:300])
            time.sleep(2.0)


def init_pg_sink(logger) -> bool:
    """Initialize background writer if BOT_PG_DSN is configured."""
    global _Q, _WORKER_STARTED, _LOGGER, _DSN, _SINK_DISABLED
    _LOGGER = logger
    _DSN = (os.getenv("BOT_PG_DSN") or "").strip()
    if not _DSN:
        return False
    if _SINK_DISABLED:
        return False

    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return True
        try:
            import psycopg  # noqa: F401
        except Exception as e:
            _SINK_DISABLED = True
            _log("warning", "pg_sink_disabled_no_driver", err=str(e)[:200])
            return False
        _Q = queue.Queue(maxsize=max(100, _QUEUE_MAX))
        t = threading.Thread(target=_worker, name="pg-sink", daemon=True)
        t.start()
        _WORKER_STARTED = True
        _log("info", "pg_sink_starting", queue_max=max(100, _QUEUE_MAX))
        return True


def _enqueue(kind: str, payload: dict) -> None:
    global _DROP_COUNT
    if _SINK_DISABLED:
        return
    q = _Q
    if q is None:
        return
    try:
        q.put_nowait((kind, payload, 0))
    except queue.Full:
        _DROP_COUNT += 1
        if _DROP_COUNT % max(1, _DROP_WARN_EVERY) == 0:
            _log("warning", "pg_sink_queue_full_drop", drops=_DROP_COUNT, kind=kind)


def enqueue_bot_event(row: dict) -> None:
    _enqueue("bot_event", row)


def enqueue_lead(row: dict) -> None:
    _enqueue("lead", row)

