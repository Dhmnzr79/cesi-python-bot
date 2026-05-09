from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal[
    "catalog_facts",
    "catalog_md",
    "price_card",
    "price_ref",
    "price_lookup_clarify",
    "price_concern",
    "doctor",
    "contacts",
    "none",
]

MatchMethod = Literal[
    "catalog_containment",
    "session_fallback",
    "doctors_lookup",
    "concern_default",
    "none",
]


class SourceRouteResult(BaseModel):
    """A3 output contract. See `docs/ARCHITECTURE V5.md` §1.3 and §A3."""

    model_config = ConfigDict(extra="forbid")

    source: SourceType
    service_id: str | None = None
    ref: str | None = None
    concern_ref: str | None = None
    payload: dict[str, Any] | None = None
    match_score: float = Field(..., ge=0.0, le=1.0)
    match_method: MatchMethod = "none"

