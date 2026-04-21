"""Приём лида: валидация, e-mail отправка и fallback на диск."""
import json
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from uuid import uuid4

from logging_setup import get_logger, log_json
from session import normalize_phone

logger = get_logger("bot")


def _send_lead_email(rec: dict) -> tuple[bool, str | None]:
    smtp_host = (os.getenv("LEAD_SMTP_HOST") or "").strip()
    smtp_port = int((os.getenv("LEAD_SMTP_PORT") or "587").strip())
    smtp_user = (os.getenv("LEAD_SMTP_USER") or "").strip()
    smtp_pass = os.getenv("LEAD_SMTP_PASS") or ""
    smtp_from = (os.getenv("LEAD_EMAIL_FROM") or smtp_user).strip()
    smtp_to = (os.getenv("LEAD_EMAIL_TO") or "").strip()
    smtp_ssl = (os.getenv("LEAD_SMTP_SSL") or "0").strip().lower() in {"1", "true", "yes"}
    smtp_starttls = (os.getenv("LEAD_SMTP_STARTTLS") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not (smtp_host and smtp_from and smtp_to):
        return False, "email_config_missing"

    msg = EmailMessage()
    msg["Subject"] = f"[Lead] {rec.get('name') or 'Без имени'}"
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content(
        "\n".join(
            [
                f"Время (UTC): {rec.get('ts')}",
                f"Имя: {rec.get('name') or '-'}",
                f"Телефон: {rec.get('phone') or '-'}",
                f"Ситуация: {rec.get('situation_note') or '-'}",
                f"SID: {rec.get('sid') or '-'}",
                f"Client ID: {rec.get('client_id') or '-'}",
                f"Intent: {rec.get('intent') or '-'}",
            ]
        ),
        charset="utf-8",
    )

    try:
        if smtp_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
                if smtp_user:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                if smtp_starttls:
                    server.starttls()
                if smtp_user:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        return True, None
    except Exception as e:
        log_json(logger, "lead_email_send_failed", err=str(e)[:300], client_id=rec.get("client_id"))
        return False, "email_send_failed"


def handle_lead(data: dict) -> tuple[dict, int]:
    name = (data.get("name") or "").strip()
    phone = normalize_phone((data.get("phone") or "").strip() or "")
    intent = (data.get("intent") or "").strip()
    sid = (data.get("sid") or "").strip()
    client_id = (data.get("client_id") or "").strip()
    situation_note = (data.get("situation_note") or "").strip()

    if not phone:
        return {"ok": False, "error_code": "bad_phone", "delivery": None}, 400

    os.makedirs("leads", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    fname = f"{ts}_{uuid4().hex[:6]}.json"
    rec = {
        "ts": ts,
        "name": name,
        "phone": phone,
        "intent": intent,
        "sid": sid,
        "client_id": client_id,
        "situation_note": situation_note,
    }

    sent, send_err = _send_lead_email(rec)
    if sent:
        return {"ok": True, "error_code": None, "delivery": "email"}, 200

    try:
        with open(os.path.join("leads", fname), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        log_json(
            logger,
            "lead_saved_file_fallback",
            client_id=client_id,
            sid=sid,
            error_code=send_err or "email_send_failed",
        )
        return {
            "ok": True,
            "error_code": send_err or "email_send_failed",
            "delivery": "file_fallback",
        }, 200
    except Exception as e:
        log_json(logger, "lead_fallback_write_failed", err=str(e)[:300], client_id=client_id, sid=sid)
        return {"ok": False, "error_code": "fallback_write_failed", "delivery": None}, 500
