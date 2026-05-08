from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RouteIntent = Literal["content", "price_lookup", "price_concern", "unknown"]
ServiceTopic = Literal["implantation", "prosthetics", "clinic", "doctors", "unknown"]
QueryMode = Literal["overview", "specific", "comparison", "process"]


class DecisionFrameConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: float = Field(..., ge=0.0, le=1.0)
    topic: float = Field(..., ge=0.0, le=1.0)
    service: float = Field(..., ge=0.0, le=1.0)
    query_mode: float = Field(..., ge=0.0, le=1.0)


class DecisionFrame(BaseModel):
    """Resolver output contract. See `docs/ARCHITECTURE V5.md` §1.1."""

    model_config = ConfigDict(extra="forbid")

    route_intent: RouteIntent
    service_topic: ServiceTopic
    service_id: str | None = None
    query_mode: QueryMode
    confidence: DecisionFrameConfidence
    needs_clarification: bool

