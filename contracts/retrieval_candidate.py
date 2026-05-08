from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DocType = Literal["faq", "service", "info", "pricing", "doctor", "contacts"]


class RetrievalCandidate(BaseModel):
    """A4 candidate contract. See `docs/ARCHITECTURE V5.md` §1.4."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(..., min_length=1)
    doc_type: DocType
    subtype: str | None = None
    topic: str = Field(..., min_length=1)
    snippet: str = Field(..., max_length=500)
    retrieval_score: float
    alias_hit: bool
    in_scope: bool

