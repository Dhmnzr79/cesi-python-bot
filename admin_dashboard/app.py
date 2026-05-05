"""Internal dashboard service (read-only) for bot analytics."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, render_template, request

try:
    import psycopg
except Exception:  # pragma: no cover - runtime guard
    psycopg = None


APP_ENV = (os.getenv("APP_ENV") or "local").strip().lower()
BOT_PG_DSN = (os.getenv("BOT_PG_DSN") or "").strip()
PORT = int(os.getenv("ADMIN_DASHBOARD_PORT", "9100"))
ADMIN_TOKEN = (os.getenv("ADMIN_DASHBOARD_TOKEN") or "").strip()
DB_CONNECT_TIMEOUT_SEC = int(os.getenv("ADMIN_DASHBOARD_DB_CONNECT_TIMEOUT_SEC", "3"))

app = Flask(__name__, template_folder="templates", static_folder="static")


def _utc_day_bounds(days_back: int = 0) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    d0 = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=days_back)
    d1 = d0 + timedelta(days=1)
    return d0, d1


def _guard():
    # Local mode: no token by default.
    if APP_ENV != "prod":
        return None
    expected = ADMIN_TOKEN
    if not expected:
        return jsonify({"error": "dashboard_token_not_set"}), 503
    got = (request.headers.get("X-Admin-Dashboard-Token") or request.args.get("token") or "").strip()
    if got == expected:
        return None
    return jsonify({"error": "not_found"}), 404


def _require_db():
    if not BOT_PG_DSN:
        return None, (jsonify({"error": "BOT_PG_DSN_not_set"}), 503)
    if psycopg is None:
        return None, (jsonify({"error": "psycopg_not_installed"}), 503)
    try:
        conn = psycopg.connect(BOT_PG_DSN, autocommit=True, connect_timeout=max(1, DB_CONNECT_TIMEOUT_SEC))
    except Exception as e:
        return None, (jsonify({"error": "db_connect_failed", "details": str(e)[:200]}), 503)
    return conn, None


def _client_id() -> str:
    return (request.args.get("client_id") or "default").strip() or "default"


def _to_int(value: str | None, default: int, min_v: int, max_v: int) -> int:
    try:
        v = int(value or default)
    except Exception:
        v = default
    return min(max(v, min_v), max_v)


@app.get("/")
def home():
    denied = _guard()
    if denied:
        return denied
    return render_template("index.html", app_env=APP_ENV)


@app.get("/api/overview")
def api_overview():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    d0, d1 = _utc_day_bounds(0)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  count(*) FILTER (WHERE event_type='user_turn_completed') AS user_turns,
                  count(DISTINCT sid) FILTER (WHERE sid IS NOT NULL) AS conversations,
                  count(*) FILTER (WHERE event_type='lead_submitted' AND status='ok') AS leads,
                  count(*) FILTER (WHERE event_type='llm_error') AS llm_errors,
                  count(*) FILTER (WHERE status='error' OR event_type IN ('llm_error', 'ask_failed', 'ask_stream_failed')) AS errors,
                  count(*) FILTER (WHERE event_type='turn_complete' AND details->>'route'='retrieval_no_candidates') AS no_candidates,
                  count(*) FILTER (WHERE event_type='turn_complete' AND details->>'route'='low_score_fallback') AS low_score_fallback,
                  COALESCE(avg((details->>'latency_ms')::numeric) FILTER (WHERE event_type='turn_complete' AND details ? 'latency_ms'), 0)::float AS avg_latency_ms
                FROM bot_events
                WHERE client_id = %s AND occurred_at >= %s AND occurred_at < %s
                """,
                (cid, d0, d1),
            )
            row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0, 0)
            cur.execute(
                """
                SELECT
                  COALESCE(sum((details->>'estimated_usd')::numeric), 0)::float
                FROM bot_events
                WHERE client_id=%s
                  AND event_type='llm_usage'
                  AND occurred_at >= %s AND occurred_at < %s
                """,
                (cid, d0, d1),
            )
            usd = float((cur.fetchone() or [0.0])[0] or 0.0)
    return jsonify(
        {
            "client_id": cid,
            "today_utc": d0.date().isoformat(),
            "user_turns": int(row[0] or 0),
            "conversations": int(row[1] or 0),
            "leads": int(row[2] or 0),
            "llm_errors": int(row[3] or 0),
            "errors": int(row[4] or 0),
            "no_candidates": int(row[5] or 0),
            "low_score_fallback": int(row[6] or 0),
            "fallbacks_total": int((row[5] or 0) + (row[6] or 0)),
            "avg_latency_ms": float(row[7] or 0.0),
            "estimated_usd": round(usd, 6),
            "conversion_percent": round(((int(row[2] or 0) / max(int(row[1] or 0), 1)) * 100.0), 2),
        }
    )


@app.get("/api/dialogs")
def api_dialogs():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    limit = _to_int(request.args.get("limit"), 30, 1, 200)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH turn_rows AS (
                  SELECT sid, client_id, occurred_at, status, details
                  FROM bot_events
                  WHERE client_id=%s AND event_type='turn_complete' AND sid IS NOT NULL
                ),
                latest_turn AS (
                  SELECT DISTINCT ON (sid)
                    sid,
                    client_id,
                    occurred_at AS last_ts,
                    status AS last_status,
                    details->>'user_text_redacted' AS last_user_text,
                    details->>'bot_text_redacted' AS last_bot_text,
                    details->>'route' AS last_route,
                    COALESCE((details->>'turn_number')::int, 0) AS turn_number
                  FROM turn_rows
                  ORDER BY sid, occurred_at DESC
                ),
                turns AS (
                  SELECT sid, count(*)::int AS turns
                  FROM turn_rows
                  GROUP BY sid
                ),
                leads_by_sid AS (
                  SELECT sid, bool_or(status='ok') AS has_lead
                  FROM bot_events
                  WHERE client_id=%s AND event_type='lead_submitted' AND sid IS NOT NULL
                  GROUP BY sid
                )
                SELECT
                  lt.sid,
                  lt.client_id,
                  lt.last_ts,
                  lt.last_status,
                  COALESCE(t.turns, 0) AS turns,
                  lt.turn_number,
                  lt.last_route,
                  lt.last_user_text,
                  lt.last_bot_text,
                  COALESCE(lb.has_lead, false) AS has_lead
                FROM latest_turn lt
                LEFT JOIN turns t ON t.sid = lt.sid
                LEFT JOIN leads_by_sid lb ON lb.sid = lt.sid
                ORDER BY lt.last_ts DESC
                LIMIT %s
                """,
                (cid, cid, limit),
            )
            rows = cur.fetchall()
    out = []
    for sid, item_client_id, last_ts, last_status, turns, turn_number, last_route, last_user_text, last_bot_text, has_lead in rows:
        status = "ok"
        if bool(has_lead):
            status = "lead"
        elif (last_status or "").lower() != "ok" or (last_route or "") in ("retrieval_no_candidates", "low_score_fallback"):
            status = "problem"
        out.append(
            {
                "sid": sid,
                "client_id": item_client_id,
                "last_ts": last_ts.isoformat() if last_ts else None,
                "turns": int(turns or 0),
                "turn_number": int(turn_number or 0),
                "last_route": last_route,
                "last_user_text": last_user_text,
                "last_bot_text": last_bot_text,
                "last_status": last_status,
                "has_lead": bool(has_lead),
                "status": status,
            }
        )
    return jsonify({"client_id": cid, "items": out})


@app.get("/api/problems")
def api_problems():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    limit = _to_int(request.args.get("limit"), 50, 1, 300)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH bad_events AS (
                  SELECT
                    occurred_at AS ts,
                    sid,
                    event_type,
                    COALESCE(details->>'user_text_redacted', details->>'user_preview_redacted') AS user_text,
                    COALESCE(details->>'route', '') AS route,
                    COALESCE(details->>'error', details->>'fallback_reason', status, event_type) AS reason,
                    COALESCE(details->>'doc_id', details->>'chunk_id') AS doc_id,
                    CASE
                      WHEN event_type IN ('llm_error', 'ask_failed', 'ask_stream_failed') OR status='error' THEN 'high'
                      WHEN details->>'route'='retrieval_no_candidates' THEN 'high'
                      WHEN details->>'route'='low_score_fallback' THEN 'medium'
                      ELSE 'low'
                    END AS priority
                  FROM bot_events
                  WHERE client_id=%s
                    AND (
                      status='error'
                      OR event_type IN ('llm_error', 'ask_failed', 'ask_stream_failed')
                      OR (event_type='turn_complete' AND details->>'route' IN ('retrieval_no_candidates', 'low_score_fallback'))
                    )
                ),
                bad_leads AS (
                  SELECT
                    captured_at AS ts,
                    sid,
                    'lead_delivery_issue' AS event_type,
                    NULL::text AS user_text,
                    NULL::text AS route,
                    COALESCE(delivery_status, 'unknown') AS reason,
                    NULL::text AS doc_id,
                    'medium'::text AS priority
                  FROM leads
                  WHERE client_id=%s
                    AND COALESCE(delivery_status, '') <> 'email'
                )
                SELECT ts, sid, event_type, user_text, route, reason, doc_id, priority
                FROM (
                  SELECT * FROM bad_events
                  UNION ALL
                  SELECT * FROM bad_leads
                ) p
                ORDER BY ts DESC
                LIMIT %s
                """,
                (cid, cid, limit),
            )
            rows = cur.fetchall()
    items = []
    for ts, sid, event_type, user_text, route, reason, doc_id, priority in rows:
        items.append(
            {
                "ts": ts.isoformat(),
                "sid": sid,
                "event_type": event_type,
                "user_text": user_text,
                "route": route,
                "reason": reason,
                "doc_id": doc_id,
                "priority": priority,
            }
        )
    return jsonify({"client_id": cid, "items": items})


@app.get("/api/leads")
def api_leads():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    limit = min(max(int(request.args.get("limit", 50)), 1), 300)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT captured_at, request_id, sid, name, phone, topic, cta_action, turns_to_lead, delivery_status
                FROM leads
                WHERE client_id=%s
                ORDER BY captured_at DESC
                LIMIT %s
                """,
                (cid, limit),
            )
            rows = cur.fetchall()
    items = []
    for ts, request_id, sid, name, phone, topic, cta_action, turns_to_lead, delivery_status in rows:
        items.append(
            {
                "captured_at": ts.isoformat() if ts else None,
                "request_id": request_id,
                "sid": sid,
                "name": name,
                "phone": phone,
                "topic": topic,
                "cta_action": cta_action,
                "turns_to_lead": turns_to_lead,
                "delivery_status": delivery_status,
                "client_id": cid,
            }
        )
    return jsonify({"client_id": cid, "items": items})


@app.get("/api/costs")
def api_costs():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    d0, d1 = _utc_day_bounds(0)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  details->>'call_type' AS call_type,
                  count(*) AS calls,
                  COALESCE(sum((details->>'prompt_tokens')::numeric),0)::bigint AS prompt_tokens,
                  COALESCE(sum((details->>'completion_tokens')::numeric),0)::bigint AS completion_tokens,
                  COALESCE(sum((details->>'estimated_usd')::numeric),0)::float AS usd
                FROM bot_events
                WHERE client_id=%s AND event_type='llm_usage'
                  AND occurred_at >= %s AND occurred_at < %s
                GROUP BY details->>'call_type'
                ORDER BY usd DESC, calls DESC
                """,
                (cid, d0, d1),
            )
            rows = cur.fetchall()
    items = []
    total = 0.0
    for call_type, calls, pt, ct, usd in rows:
        val = float(usd or 0.0)
        total += val
        items.append(
            {
                "call_type": call_type or "unknown",
                "calls": int(calls or 0),
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "estimated_usd": round(val, 6),
            }
        )
    return jsonify({"client_id": cid, "today_utc": d0.date().isoformat(), "estimated_usd_total": round(total, 6), "items": items})


@app.get("/api/events")
def api_events():
    denied = _guard()
    if denied:
        return denied
    conn, err = _require_db()
    if err:
        return err
    cid = _client_id()
    limit = _to_int(request.args.get("limit"), 200, 1, 1000)
    event_type = (request.args.get("event_type") or "").strip()
    sid = (request.args.get("sid") or "").strip()
    request_id = (request.args.get("request_id") or "").strip()
    with conn:
        with conn.cursor() as cur:
            where_parts = ["client_id=%s"]
            params: list[object] = [cid]
            if event_type:
                where_parts.append("event_type=%s")
                params.append(event_type)
            if sid:
                where_parts.append("sid=%s")
                params.append(sid)
            if request_id:
                where_parts.append("request_id=%s")
                params.append(request_id)
            params.append(limit)
            query = f"""
                SELECT occurred_at, event_type, request_id, sid, status, details
                FROM bot_events
                WHERE {' AND '.join(where_parts)}
                ORDER BY occurred_at DESC
                LIMIT %s
            """
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
    items = []
    for ts, event_type, request_id, sid, status, details in rows:
        items.append(
            {
                "ts": ts.isoformat(),
                "event_type": event_type,
                "request_id": request_id,
                "sid": sid,
                "status": status,
                "details": details or {},
            }
        )
    return jsonify({"client_id": cid, "items": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)

