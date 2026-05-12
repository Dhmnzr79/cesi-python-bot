"""A7 Verifier — shadow/eval only (PR #1.9).

Один LLM-вызов с `VerifierVerdict`, только при детерминированном high-risk триггере по тексту ответа.
Не меняет ответ пользователю и не блокирует выдачу: LLM выполняется в фоне после фиксации ответа.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import ValidationError

from contracts.verifier_verdict import VerifierVerdict
from core.routing_loader import THRESHOLDS
from llm import client
from logging_setup import emit_bot_event, get_logger, log_llm_error, log_llm_usage

logger = get_logger("bot")

_MODEL = (os.getenv("MODEL_VERIFIER") or "").strip() or "gpt-5.4-nano"

_SHADOW_BOOT_LOCK = threading.Lock()
_shadow_slot_sem: threading.Semaphore | None = None
_SHADOW_EXECUTOR: ThreadPoolExecutor | None = None


def _shadow_runtime() -> tuple[threading.Semaphore, ThreadPoolExecutor]:
    """Пул воркеров + семафор ёмкости workers+backlog (ограничение очереди submit)."""
    global _shadow_slot_sem, _SHADOW_EXECUTOR
    with _SHADOW_BOOT_LOCK:
        if _SHADOW_EXECUTOR is None:
            w = max(1, min(int(THRESHOLDS.verifier.max_concurrent_shadow), 32))
            b = max(0, min(int(THRESHOLDS.verifier.shadow_backlog_max), 256))
            cap = w + b
            _shadow_slot_sem = threading.Semaphore(cap)
            _SHADOW_EXECUTOR = ThreadPoolExecutor(max_workers=w, thread_name_prefix="verifier_shadow")
        assert _shadow_slot_sem is not None
        return _shadow_slot_sem, _SHADOW_EXECUTOR

VerifierRunStatus = Literal["skipped", "ok", "error"]

VERIFIER_SYSTEM_PROMPT = """Ты слой проверки (Verifier) для ответа стоматологического ассистента.

На входе: фрагмент источника (единственный разрешённый контекст фактов) и готовый ответ бота.
Твоя задача — оценить, все ли конкретные утверждения в ответе опираются на этот фрагмент.

Правила:
- Если в ответе есть цифры, сроки, цены, проценты, гарантии или категоричные медицинские выводы, которых нет в источнике — это не grounded.
- «Мягкие» формулировки без новых фактов (перефраз, эмпатия) — не считаются нарушением.
- hallucinated_facts: краткие строки-описания того, что выглядит не из источника (или пустой список).
- confidence: твоя уверенность в оценке от 0 до 1.

Верни строго один JSON-объект с полями: grounded (boolean), hallucinated_facts (массив строк), confidence (число 0..1).
"""


def _norm(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def collect_high_risk_signals(answer: str) -> list[str]:
    """Детерминированный high-risk триггер по финальному тексту ответа (без LLM)."""
    if not (answer or "").strip():
        return []
    raw = answer
    n = _norm(raw)
    found: set[str] = set()

    if re.search(r"\d", raw):
        found.add("digit")

    if re.search(r"\d+(?:[.,]\d+)?\s*%", raw) or "процент" in n:
        found.add("percent")

    if re.search(r"(₽|руб\.?\b|\bр\.\s*\d|\bрубл)", n) or re.search(r"\b(тыс|млн)\.?\s*руб", n):
        found.add("currency_rub")

    if re.search(
        r"\b(оплат|рассроч|поэтапн|кредит|цен[аеы]|стоим|скидк|акци[яи])\w*",
        n,
    ):
        found.add("payment_or_price_word")

    if re.search(
        r"\b(\d{1,4}\s*[-–]?\s*\d{0,4}\s*(лет|год|года|месяц|месяца|месяцев|недел\w*|день|дня|дней|суток|сутки)|"
        r"через\s+\d{1,3}\s*(дн|дня|дней|час|часа|часов|мин|недел)|"
        r"за\s+\d{1,3}\s*(дн|дня|дней|час|часа|часов))\b",
        n,
    ):
        found.add("duration_or_delay")

    if re.search(
        r"\b(сегодня|завтра|сразу|в\s+течение\s+\d{1,3}|однодневн|моментально|"
        r"за\s+один\s+день|через\s+день)\b",
        n,
    ):
        found.add("time_promise")

    if re.search(r"\b(гарант|гарантий)\w*", n):
        found.add("warranty")

    if re.search(
        r"\b(абсолютно|гарантированно|100\s*%|точно\s+(можно|нельзя)|"
        r"никаких\s+осложн|без\s+осложн|полностью\s+исключ)",
        n,
    ):
        found.add("absolute_claim")

    if re.search(r"\bпротивопоказ", n):
        found.add("contraindication")

    return sorted(found)


def build_turn_trace_prefix(*, answer: str, source_ref: str, source_text: str) -> dict[str, Any]:
    """Поля для `turn_complete` / request.ctx до фонового LLM."""
    reasons = collect_high_risk_signals(answer)
    triggered = len(reasons) > 0
    base: dict[str, Any] = {
        "verifier_triggered": triggered,
        "verifier_trigger_reason": list(reasons),
        "verifier_generator_source_ref": (source_ref or "").strip() or None,
    }
    if not triggered:
        base.update(
            {
                "verifier_status": "skipped",
                "verifier_shadow_async": False,
                "verifier_grounded": None,
                "verifier_confidence": None,
                "verifier_hallucinated_facts": None,
            }
        )
    else:
        base.update(
            {
                "verifier_status": None,
                "verifier_shadow_async": True,
                "verifier_grounded": None,
                "verifier_confidence": None,
                "verifier_hallucinated_facts": None,
            }
        )
    _ = source_text  # caller may pass combined chunk+append body for future trace hints
    return base


def _verifier_user_payload(*, source_ref: str, source_snippet: str, answer: str) -> str:
    return json.dumps(
        {
            "source_ref": source_ref,
            "source_snippet": (source_snippet or "")[:8000],
            "bot_answer": (answer or "")[:8000],
        },
        ensure_ascii=False,
    )


def verify_answer_structured(
    *,
    answer: str,
    source_snippet: str,
    source_ref: str,
    call_type: str = "v5_verifier",
) -> tuple[VerifierVerdict | None, VerifierRunStatus, str | None]:
    """Синхронный LLM-вызов Verifier. Для runtime вызывайте из фонового потока."""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, "skipped", "no_openai_api_key"

    timeout_sec = float(THRESHOLDS.verifier.timeout_sec)
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0,
            max_completion_tokens=400,
            response_format={"type": "json_object"},
            timeout=timeout_sec,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": _verifier_user_payload(source_ref=source_ref, source_snippet=source_snippet, answer=answer)},
            ],
        )
        log_llm_usage(logger, resp, call_type=call_type, model=_MODEL)
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        verdict = VerifierVerdict.model_validate(obj)
        return verdict, "ok", None
    except TimeoutError as e:
        log_llm_error(logger, call_type=call_type, err=str(e), model=_MODEL)
        return None, "skipped", f"timeout:{e!s}"[:200]
    except ValidationError as e:
        try:
            logger.warning(
                "verifier_validation_failed",
                extra={
                    "extra_data": {
                        "call_type": call_type,
                        "model": _MODEL,
                        "raw_output": (raw or "")[:2000],
                        "error": str(e)[:2000],
                    }
                },
            )
        except Exception:
            pass
        return None, "error", f"validation:{str(e)[:400]}"
    except json.JSONDecodeError as e:
        log_llm_error(logger, call_type=call_type, err=str(e), model=_MODEL)
        return None, "error", f"json_decode:{str(e)[:200]}"
    except Exception as e:
        err = str(e)
        low = err.lower()
        if "timeout" in low or "timed out" in low:
            log_llm_error(logger, call_type=call_type, err=err[:500], model=_MODEL)
            return None, "skipped", f"timeout:{err[:200]}"
        log_llm_error(logger, call_type=call_type, err=err[:500], model=_MODEL)
        return None, "error", err[:500]


def _verdict_shadow_document(
    *,
    verdict: VerifierVerdict | None,
    verifier_status: VerifierRunStatus,
    outcome_detail: str | None,
    trace_prefix: dict[str, Any],
) -> dict[str, Any]:
    """JSON для колонки v5_turn_traces.verifier_verdict (shadow, не блокирующий контракт ответа пользователю)."""
    return {
        "shadow_only": True,
        "verifier_status": verifier_status,
        "verifier_trigger_reason": trace_prefix.get("verifier_trigger_reason"),
        "verifier_generator_source_ref": trace_prefix.get("verifier_generator_source_ref"),
        "verifier_source_has_deterministic_append": trace_prefix.get("verifier_source_has_deterministic_append"),
        "verifier_outcome_detail": outcome_detail,
        "verdict": verdict.model_dump() if verdict else None,
    }


def _commit_verifier_shadow_outcome(
    *,
    logger_,
    request_id: str | None,
    sid: str,
    client_id: str | None,
    route: str | None,
    answer_preview: str,
    trace_prefix: dict[str, Any],
    verdict: VerifierVerdict | None,
    verifier_status: VerifierRunStatus,
    outcome_detail: str | None,
) -> None:
    """Всегда пишет verifier_shadow + upsert PG (если есть turn_id); нужен для полноты denominator."""
    try:
        _emit_verifier_shadow(
            logger_=logger_,
            request_id=request_id,
            sid=sid,
            client_id=client_id,
            route=route,
            answer_preview=answer_preview,
            trace_prefix=trace_prefix,
            verdict=verdict,
            verifier_status=verifier_status,
            outcome_detail=outcome_detail,
        )
    except Exception as e:
        try:
            logger.warning(
                "verifier_shadow_emit_failed",
                extra={"extra_data": {"err": str(e)[:400], "request_id": request_id}},
            )
        except Exception:
            pass
    if not request_id:
        return
    try:
        from pg_sink import enqueue_v5_verifier_shadow

        enqueue_v5_verifier_shadow(
            {
                "turn_id": str(request_id),
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sid": sid,
                "client_id": client_id,
                "request_id": str(request_id),
                "verifier_verdict": _verdict_shadow_document(
                    verdict=verdict,
                    verifier_status=verifier_status,
                    outcome_detail=outcome_detail,
                    trace_prefix=trace_prefix,
                ),
            }
        )
    except Exception as e:
        try:
            logger.warning(
                "verifier_shadow_pg_enqueue_failed",
                extra={"extra_data": {"err": str(e)[:400], "request_id": request_id}},
            )
        except Exception:
            pass


def _emit_verifier_shadow(
    *,
    logger_,
    request_id: str | None,
    sid: str,
    client_id: str | None,
    route: str | None,
    answer_preview: str,
    trace_prefix: dict[str, Any],
    verdict: VerifierVerdict | None,
    verifier_status: VerifierRunStatus,
    outcome_detail: str | None,
) -> None:
    det: dict[str, Any] = {
        "request_id": request_id,
        "sid": sid,
        "client_id": client_id,
        "route": route,
        "answer_preview": answer_preview[:240],
        "verifier_triggered": trace_prefix.get("verifier_triggered"),
        "verifier_trigger_reason": trace_prefix.get("verifier_trigger_reason"),
        "verifier_generator_source_ref": trace_prefix.get("verifier_generator_source_ref"),
        "verifier_status": verifier_status,
        "verifier_outcome_detail": outcome_detail,
        "verifier_grounded": verdict.grounded if verdict else None,
        "verifier_confidence": verdict.confidence if verdict else None,
        "verifier_hallucinated_facts": list(verdict.hallucinated_facts) if verdict else None,
    }
    st = "ok" if verifier_status == "ok" else ("error" if verifier_status == "error" else "ok")
    emit_bot_event(logger_, "verifier_shadow", status=st, details=det)


def schedule_verifier_shadow_if_needed(
    *,
    answer: str,
    source_text: str,
    source_ref: str,
    sid: str,
    client_id: str | None,
    route: str | None,
    logger_,
    trace_prefix: dict[str, Any] | None = None,
) -> None:
    """После финального answer: синхронно логирует skip; при high-risk — фоновый LLM без блокировки HTTP.

    High-risk: семафор ограничивает число задач «в полёте» (воркеры + backlog из routing.yaml); при переполнении —
    `verifier_shadow` + PG с `queue_saturated`. Любой сбой submit/воркера также фиксируется в telemetry (denominator).
    """
    trace_prefix = trace_prefix or build_turn_trace_prefix(
        answer=answer, source_ref=source_ref, source_text=source_text
    )
    request_id: str | None = None
    try:
        from flask import has_request_context, request

        if has_request_context():
            request_id = (getattr(request, "ctx", None) or {}).get("request_id")
    except Exception:
        pass

    preview = (answer or "").strip()

    if not trace_prefix.get("verifier_triggered"):
        _emit_verifier_shadow(
            logger_=logger_,
            request_id=request_id,
            sid=sid,
            client_id=client_id,
            route=route,
            answer_preview=preview,
            trace_prefix=trace_prefix,
            verdict=None,
            verifier_status="skipped",
            outcome_detail="trigger_not_met",
        )
        return

    sem, executor = _shadow_runtime()
    if not sem.acquire(blocking=False):
        _commit_verifier_shadow_outcome(
            logger_=logger_,
            request_id=request_id,
            sid=sid,
            client_id=client_id,
            route=route,
            answer_preview=preview,
            trace_prefix=trace_prefix,
            verdict=None,
            verifier_status="error",
            outcome_detail="queue_saturated",
        )
        return

    def _guarded_worker() -> None:
        try:
            verdict, st, err = verify_answer_structured(
                answer=answer,
                source_snippet=source_text,
                source_ref=source_ref,
                call_type="v5_verifier",
            )
            detail = None
            if st == "skipped":
                detail = err or "skipped"
            elif st == "error":
                detail = err or "error"
            _commit_verifier_shadow_outcome(
                logger_=logger_,
                request_id=request_id,
                sid=sid,
                client_id=client_id,
                route=route,
                answer_preview=preview,
                trace_prefix=trace_prefix,
                verdict=verdict,
                verifier_status=st,
                outcome_detail=detail,
            )
        except Exception as e:
            try:
                logger.warning(
                    "verifier_shadow_worker_failed",
                    extra={"extra_data": {"err": str(e)[:500], "request_id": request_id}},
                )
            except Exception:
                pass
            _commit_verifier_shadow_outcome(
                logger_=logger_,
                request_id=request_id,
                sid=sid,
                client_id=client_id,
                route=route,
                answer_preview=preview,
                trace_prefix=trace_prefix,
                verdict=None,
                verifier_status="error",
                outcome_detail=f"worker_exception:{str(e)[:400]}",
            )
        finally:
            sem.release()

    try:
        executor.submit(_guarded_worker)
    except Exception as e:
        try:
            sem.release()
        except Exception:
            pass
        try:
            logger.warning(
                "verifier_shadow_submit_failed",
                extra={"extra_data": {"err": str(e)[:400], "request_id": request_id}},
            )
        except Exception:
            pass
        _commit_verifier_shadow_outcome(
            logger_=logger_,
            request_id=request_id,
            sid=sid,
            client_id=client_id,
            route=route,
            answer_preview=preview,
            trace_prefix=trace_prefix,
            verdict=None,
            verifier_status="error",
            outcome_detail=f"submit_failed:{str(e)[:400]}",
        )
