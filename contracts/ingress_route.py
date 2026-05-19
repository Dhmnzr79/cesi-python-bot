from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


IngressRoute = Literal[
    "normal",
    "hard_stop_non_target",
    "manual_contact",
    "not_offered_policy",
    "service_not_offered",
]

PolicyKey = Literal["no_pediatric_dentistry", "no_oms", "no_dms"]

IngressSource = Literal[
    "rule",
    "llm",
    "catalog_ground_truth",
    "doctor_ground_truth",
    "offered_ground_truth",
    "fallback",
    "skipped",
]


class IngressRouteResult(BaseModel):
    """Early ingress classifier output (before Resolver / retrieval)."""

    model_config = ConfigDict(extra="forbid")

    route: IngressRoute
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1, max_length=128)
    policy_key: PolicyKey | None = None
    requested_service: str | None = Field(default=None, max_length=120)
    source: IngressSource = "llm"
    is_urgent: bool = False
