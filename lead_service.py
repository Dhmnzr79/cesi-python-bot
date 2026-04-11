"""Приём лида: валидация и запись на диск."""
import json
import os
from datetime import datetime
from uuid import uuid4


def handle_lead(data: dict) -> tuple[dict, int]:
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    intent = (data.get("intent") or "").strip()

    if not phone or len(phone) < 6:
        return {"ok": False, "error": "invalid_phone"}, 400

    os.makedirs("leads", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    fname = f"{ts}_{uuid4().hex[:6]}.json"
    rec = {"ts": ts, "name": name, "phone": phone, "intent": intent}

    with open(os.path.join("leads", fname), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)

    return {"ok": True}, 200
