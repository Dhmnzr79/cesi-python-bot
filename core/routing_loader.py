from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class IngressMinConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_stop_non_target: float = Field(..., ge=0.0, le=1.0)
    manual_contact: float = Field(..., ge=0.0, le=1.0)
    service_not_offered: float = Field(..., ge=0.0, le=1.0)


class IngressThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: IngressMinConfidence


class ResolverMinConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: float = Field(..., ge=0.0, le=1.0)
    topic: float = Field(..., ge=0.0, le=1.0)
    service: Literal["ignored"]
    query_mode: float = Field(..., ge=0.0, le=1.0)


class ResolverThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: ResolverMinConfidence


class ArbiterThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(..., ge=0.0, le=1.0)


class VerifierThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(..., ge=0.0, le=1.0)
    timeout_sec: float = Field(..., ge=1.0, le=120.0)
    max_concurrent_shadow: int = Field(..., ge=1, le=32)
    shadow_backlog_max: int = Field(..., ge=0, le=256)


class RetrievalThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_topic_min_confidence: float = Field(..., ge=0.0, le=1.0)
    low_score_threshold: float = Field(..., ge=0.0, le=1.0)
    alias_scope_guard_min: float = Field(..., ge=0.0, le=1.0)


class CatalogMatchThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    containment_min: float = Field(..., ge=0.0, le=1.0)


class AliasThresholds(BaseModel):
    """PR #1.10 alias pipeline thresholds (see IMPLEMENTATION_PLAN PR #1.10)."""

    model_config = ConfigDict(extra="forbid")

    strong_effective_min: float = Field(..., ge=0.0, le=1.0)
    soft_assist_min: float = Field(..., ge=0.0, le=1.0)
    near_exact_score: float = Field(..., ge=0.0, le=1.0)
    near_exact_length_ratio_min: float = Field(..., ge=0.0, le=1.0)
    embedding_high_min: float = Field(..., ge=0.0, le=1.0)
    embedding_strong_cosine_min: float = Field(..., ge=0.0, le=1.0)
    embedding_medium_min: float = Field(..., ge=0.0, le=1.0)
    embedding_medium_max: float = Field(..., ge=0.0, le=1.0)
    embedding_medium_score_cap: float = Field(..., ge=0.0, le=1.0)
    rescue_max_query_chars: int = Field(..., ge=1, le=256)
    rescue_max_core_tokens: int = Field(..., ge=1, le=32)
    rescue_min_sim: float = Field(..., ge=0.0, le=1.0)
    rescue_margin_min: float = Field(..., ge=0.0, le=1.0)
    rescue_effective_cap: float = Field(..., ge=0.0, le=1.0)
    scope_guard_min: float = Field(..., ge=0.0, le=1.0)
    embed_matrix_top_chunks: int = Field(..., ge=8, le=512)


class Thresholds(BaseModel):
    """Validated representation of `core/routing.yaml` (see ARCHITECTURE V5.md §D2)."""

    model_config = ConfigDict(extra="forbid")

    ingress: IngressThresholds
    resolver: ResolverThresholds
    arbiter: ArbiterThresholds
    verifier: VerifierThresholds
    retrieval: RetrievalThresholds
    catalog_match: CatalogMatchThresholds
    alias: AliasThresholds


_LOCK = threading.Lock()
_CACHED: Thresholds | None = None
_CACHED_MTIME: float | None = None


def _routing_yaml_path() -> str:
    return os.path.join(os.path.dirname(__file__), "routing.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"routing thresholds file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"routing thresholds must be a YAML mapping at top-level: {path}")
    return data


def load_thresholds(*, force_reload: bool = False) -> Thresholds:
    """Load and validate thresholds from `core/routing.yaml` with a simple mtime cache."""
    global _CACHED, _CACHED_MTIME
    path = _routing_yaml_path()
    mtime = None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    with _LOCK:
        if not force_reload and _CACHED is not None and _CACHED_MTIME == mtime:
            return _CACHED
        raw = _load_yaml(path)
        try:
            parsed = Thresholds.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"invalid routing thresholds schema in {path}: {e}") from e
        _CACHED = parsed
        _CACHED_MTIME = mtime
        return parsed


@dataclass(frozen=True)
class _ThresholdsProxy:
    """Proxy to keep `THRESHOLDS` as a module-level singleton-like value."""

    def __getattr__(self, item: str):
        return getattr(load_thresholds(), item)


THRESHOLDS = _ThresholdsProxy()

