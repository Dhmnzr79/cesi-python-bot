from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArbiterDecision(BaseModel):
    """A5 output contract. See `docs/ARCHITECTURE V5.md` §1.5."""

    model_config = ConfigDict(extra="forbid")

    selected_ref: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    alternative: str | None = None

