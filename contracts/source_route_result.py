from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal[
    "catalog_facts",
    "catalog_md",
    "price_card",
    "doctor",
    "contacts",
    "none",
]


class SourceRouteResult(BaseModel):
    """A3 output contract. See `docs/ARCHITECTURE V5.md` §1.3."""

    model_config = ConfigDict(extra="forbid")

    source: SourceType
    service_id: str | None = None
    ref: str | None = None
    payload: dict[str, Any] | None = None
    match_score: float = Field(..., ge=0.0, le=1.0)

