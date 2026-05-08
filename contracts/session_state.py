from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


HistoryRole = Literal["user", "assistant", "system"]
LeadState = Literal["collecting_name", "collecting_phone", "active"]


class SessionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: HistoryRole
    text: str
    ts: str


class SessionState(BaseModel):
    """Session state contract (read by all layers). See `docs/ARCHITECTURE V5.md` §1.7."""

    model_config = ConfigDict(extra="forbid")

    sid: str = Field(..., min_length=1)
    client_id: str = Field(..., min_length=1)
    history: list[SessionMessage]
    current_doc_id: str | None = None
    # На рантайме сейчас хранится как `last_catalog_service_id` в `session.mem`.
    last_service_id: str | None = None
    covered_h3: list[str]
    topic_turn_count: int = Field(..., ge=0)
    lead_state: LeadState | None = None
    shown_boosters: list[str]

