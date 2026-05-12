from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AskOrchestrationResult(BaseModel):
    """
    Результат общего оркестратора /ask и /ask/stream (PR #1.2.5).
    Различается только способ финальной выдачи: JSON vs SSE.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["unknown_client", "reset_session", "service_reply", "chunk"]

    q: str = ""
    sid: str = ""
    client_id: str = ""

    http_status: int = Field(default=200, ge=100, le=599)

    decision_frame: dict[str, Any] | None = None

    client_error: dict[str, Any] | None = None

    service_payload: dict[str, Any] | None = None
    service_doc_id: str | None = None
    service_track_user: bool = True
    service_route: str | None = None

    chosen_chunk: dict[str, Any] | None = None
    llm_question: str | None = None
    log_event: str = "Answer generated"
    chunk_route: str = "retrieval_chunk"
    # Детерминированный хвост ответа (например цена из каталога), не через инструкции к LLM.
    generator_append_text: str | None = None
