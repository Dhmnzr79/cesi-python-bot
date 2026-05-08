from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerifierVerdict(BaseModel):
    """A7 output contract. See `docs/ARCHITECTURE V5.md` §1.6."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    hallucinated_facts: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

