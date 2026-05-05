# logging_setup.py
import json
import logging
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config import estimate_llm_usage_usd

LOG_DIR = os.getenv("BOT_LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, os.getenv("BOT_LOG_FILE", "app.jsonl"))

SENSITIVE_KEYS = ("api_key", "apikey", "token", "secret", "authorization", "password")
_PHONE_DIGIT_MIN = 10

BOT_EVENTS_SCHEMA_VERSION = int(os.getenv("BOT_EVENTS_SCHEMA_VERSION", "1"))


def _mask_phone_like(value):
    s = str(value or "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < _PHONE_DIGIT_MIN:
        return value
    if len(digits) >= 11:
        return f"+{digits[0]}******{digits[-2:]}"
    return "***"


def _sanitize(d):
    if not isinstance(d, dict):
        return d
    clean = {}
    for k, v in d.items():
        kl = k.lower() if isinstance(k, str) else ""
        if isinstance(k, str) and any(s in kl for s in SENSITIVE_KEYS):
            clean[k] = "***"
        elif isinstance(k, str) and ("phone" in kl or "tel" in kl):
            clean[k] = _mask_phone_like(v)
        elif isinstance(k, str) and "situation" in kl:
            txt = str(v or "")
            clean[k] = (txt[:80] + "…") if len(txt) > 80 else txt
        elif isinstance(v, dict):
            clean[k] = _sanitize(v)
        elif isinstance(v, list):
            clean[k] = [_sanitize(x) if isinstance(x, dict) else x for x in v]
        else:
            clean[k] = v
    return clean


class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)


def get_logger(name="bot"):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = JsonLineFormatter()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def request_context_defaults() -> dict:
    """Поля HTTP-запроса для склейки пайплайна (без обязательного Flask вне контекста)."""
    try:
        from flask import has_request_context, request

        if has_request_context() and getattr(request, "ctx", None):
            ctx = request.ctx
            sid = ctx.get("sid")
            out = {
                "request_id": ctx.get("request_id"),
                "sid": sid,
                "session_id": sid,
                "client_id": ctx.get("client_id"),
                "path": ctx.get("path"),
            }
            return {k: v for k, v in out.items() if v is not None}
    except Exception:
        pass
    return {}


def make_request_context(cookie_sid=None):
    """Контекст запроса: request_id + sid из cookie до разбора body."""
    cookie = (cookie_sid or "").strip() or None
    return {
        "request_id": str(uuid.uuid4()),
        "sid": cookie,
        "session_id": cookie,
        "client_id": None,
        "app_version": os.getenv("APP_VERSION", "dev"),
        "env": os.getenv("APP_ENV", "local"),
    }


def emit_bot_event(
    logger,
    event_name: str,
    *,
    status=None,
    details: dict | None = None,
    **overrides: object,
):
    """Продуктовое событие для дашборда и Postgres-импорта (единый контракт)."""
    row = {
        "kind": "bot_event",
        "schema_version": BOT_EVENTS_SCHEMA_VERSION,
        "event": event_name,
    }
    row.update(request_context_defaults())
    for k, v in overrides.items():
        if v is not None:
            row[k] = v
    if status is not None:
        row["status"] = status
    row["details"] = dict(details or {})
    logger.info("bot_event", extra={"extra_data": _sanitize(row)})


def log_json(logger, message, **fields):
    """Как раньше, плюс подстановка request_id / sid / client_id / path из Flask ctx."""
    inj = request_context_defaults()
    for k, v in inj.items():
        if k == "session_id":
            continue
        fields.setdefault(k, v)
    logger.info(message, extra={"extra_data": _sanitize(fields)})


def usage_dict_from_completion(resp) -> dict | None:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    pt = getattr(u, "prompt_tokens", None)
    ct = getattr(u, "completion_tokens", None)
    tt = getattr(u, "total_tokens", None)
    out = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }
    est = estimate_llm_usage_usd(prompt_tokens=pt, completion_tokens=ct)
    if est is not None:
        out["estimated_usd"] = est
    return out


def log_llm_usage(
    logger,
    resp,
    *,
    call_type: str,
    model: str | None = None,
    extra_details: dict | None = None,
):
    """После успешного chat.completions.create (non-stream)."""
    u = usage_dict_from_completion(resp)
    if not u:
        return
    det = {"call_type": call_type, "model": model or getattr(resp, "model", None)}
    if extra_details:
        det.update(extra_details)
    det.update(u)
    emit_bot_event(logger, "llm_usage", details=det)


def log_llm_stream_usage(
    logger,
    usage_obj,
    *,
    call_type: str,
    model: str | None,
    extra_details: dict | None = None,
):
    """После stream с include_usage (или финальный chunk.usage)."""
    if usage_obj is None:
        return
    pt = getattr(usage_obj, "prompt_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None)
    tt = getattr(usage_obj, "total_tokens", None)
    det = {
        "call_type": call_type,
        "model": model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }
    est = estimate_llm_usage_usd(prompt_tokens=pt, completion_tokens=ct)
    if est is not None:
        det["estimated_usd"] = est
    if extra_details:
        det.update(extra_details)
    emit_bot_event(logger, "llm_usage", details=det)


def log_llm_error(logger, *, call_type: str, err: str, model: str | None = None):
    emit_bot_event(
        logger,
        "llm_error",
        status="error",
        details={
            "call_type": call_type,
            "error": (err or "")[:500],
            "model": model,
        },
    )
