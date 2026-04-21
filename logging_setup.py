# logging_setup.py
import json, logging, os, sys, uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.getenv("BOT_LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, os.getenv("BOT_LOG_FILE", "app.jsonl"))

SENSITIVE_KEYS = ("api_key","apikey","token","secret","authorization","password")
_PHONE_DIGIT_MIN = 10


def _mask_phone_like(value):
    s = str(value or "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < _PHONE_DIGIT_MIN:
        return value
    if len(digits) >= 11:
        return f"+{digits[0]}******{digits[-2:]}"
    return "***"

def _sanitize(d):
    if not isinstance(d, dict): return d
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
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def make_request_context(session_id=None):
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "app_version": os.getenv("APP_VERSION", "dev"),
        "env": os.getenv("APP_ENV", "local"),
    }

def log_json(logger, message, **fields):
    logger.info(message, extra={"extra_data": _sanitize(fields)})