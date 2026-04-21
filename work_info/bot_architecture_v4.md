# Bot Architecture v4 — demo production (single-client)

Документ фиксирует текущее состояние бота после hardening под быстрый demo-prod запуск для одного клиента.
Фокус: стабильный `/ask`, безопасный `/lead`, предсказуемый UX, минимум операционных рисков.

---

## 1) Цель версии

- Запуск demo-версии для одного клиента без тяжелых платформенных задач (CRM/n8n/multitenant/admin).
- Закрытие критичных дыр перед продом: debug-доступ, lead-delivery, единая валидация телефона, fail-soft LLM, PII в логах.
- Сохранение текущей продуктовой логики CTA/video/situation/followup.

---

## 2) Основные роуты

### `POST /ask`
- Основной контентный и сценарный endpoint.
- Валидация `client_id` через `resolve_client_id`.
- Ветки:
  - flow-handlers (lead/situation/back/yes),
  - ref-routing,
  - retrieval + chunk response.

### `POST /lead`
- Валидация JSON (`400 bad_json`).
- Валидация `client_id` как в `/ask` (`403 unknown_client`).
- Возврат унифицированного технического контракта:
  - `ok: bool`
  - `error_code: str | null`
  - `delivery: "email" | "file_fallback" | null`

### Debug роуты
- `/_debug/ping`
- `/__debug/retrieval`

Поведение:
- при `APP_ENV=prod` -> `404`
- вне `prod` -> обязательный `X-Debug-Token`

---

## 3) Модули и ответственность (as-built)

### `app.py`
- HTTP-граница, валидация `client_id`, маршрутизация, финализация ответа.
- Debug-роуты с prod-блокировкой.
- `/lead` интегрирован с единым lead-контрактом.

### `query_selector.py`
- Выбор чанка: dual retrieval, merge, alias assist, low-score guard, selective rerank.

### `retriever.py`
- Semantic retrieval + alias channels (raw/lemma/trigram).
- Alias optimization: индекс alias-термов при загрузке корпуса + fallback на полный перебор.
- `llm_rerank`:
  - JSON-only контракт `{"choice": 1}`,
  - строгая валидация,
  - fallback на top-1 с `fallback_reason`.

### `llm.py`
- Query rewrite для retrieval (JSON mode + валидация).
- Генерация ответа (JSON mode).
- Timeout + fail-soft fallback для answer generation и rewrite.

### `policy.py`
- Детерминированное управление UI-элементами: followups, refs, video, situation, CTA.
- Порог CTA на уровне темы через `cta_from_turn` (frontmatter).

### `flow_handlers.py`
- Сценарные ветки lead/situation.
- Правило `yes` после followup:
  - 1 кнопка -> redirect по `ref`,
  - 2+ кнопки -> уточнение выбора, без автопрыжка в первую.

### `session.py`
- SQLite session state.
- Единая нормализация телефона `normalize_phone()` в формат `+7XXXXXXXXXX`.
- `extract_phone()` использует `normalize_phone()`.

### `lead_service.py`
- Основная доставка лида: email.
- При ошибке email: file fallback (`leads/*.json`) + лог техпричины.
- Возврат минимального результата (`ok/error_code/delivery`).

### `logging_setup.py`
- JSONL логирование.
- Санитизация секретов + маскирование PII:
  - phone/tel поля маскируются,
  - `situation*` подрезаются.

### `meta_loader.py`
- Загрузка frontmatter.
- Поддержка `cta_from_turn` (дефолт `0`).

### `chunk_responder.py`
- Контентный пайплайн: chunk -> LLM -> UX payload -> policy -> session side-effects.

### `static/debug/ask-inspector.html` (dev)
- Отдельная страница: сырой JSON ответа `/ask`, раздельно `quick_replies` и `meta.followups`, CTA/ситуация/видео, блок `GET /__debug/retrieval` (с `X-Debug-Token`).
- Не входит в UI виджета; папку `static/debug/` можно удалить целиком.

---

## 4) Контент и frontmatter

Источники: `md/*.md`.

Ключевые поля frontmatter, реально используемые сейчас:
- `doc_id`, `doc_type`, `topic`, `subtopic`
- `aliases`
- `suggest_h3`, `suggest_refs`
- `cta_text`, `cta_action`, `cta_from_turn`
- `situation_allowed`
- `video_key`
- `empathy_enabled`, `empathy_tag`

Примечание:
- `situation_note` не прокидывается в retrieval/LLM (осознанно), используется в lead-контуре.

---

## 5) `/ask` — рабочий пайплайн

1. Принять запрос и валидировать `client_id`.
2. Обработать flow-ветки (lead/situation/back/yes/followup redirect).
3. Если есть `ref` -> ответ из конкретного чанка.
4. Иначе:
   - select chunk (`query_selector`),
   - low-score fallback при необходимости,
   - контентный ответ через `respond_from_chunk`.
5. `respond_from_chunk`:
   - LLM answer (fail-soft),
   - policy,
   - session side-effects,
   - унифицированный JSON.

---

## 6) Policy: followup/video/situation/CTA

- При наличии `video_key` в первом content-turn followup-слоты сжимаются (часто по одному за шаг).
- `quick_replies` refs показываются ограниченно (не более 1 в выдаче).
- CTA показывается по порогу `cta_from_turn` для конкретного документа.
- CTA скрывается при активном lead/situation pending и при явном booking-intent.

---

## 7) Lead контур (demo-safe)

Единый результат `handle_lead()`:
- `ok=true, delivery="email"`: email доставлен.
- `ok=true, delivery="file_fallback"`: email не сработал, fallback сохранен.
- `ok=false, delivery=null`: лид не обработан.

Технические коды (`error_code`) используются для логов/дебага, не как пользовательский текст.

---

## 8) Runtime и deploy правила

### Single-worker правило
- SQLite session storage -> запуск только 1 воркер.
- Production entrypoint:
  - `gunicorn -w 1 -b 0.0.0.0:8000 app:app`
- Зафиксировано в:
  - `Dockerfile`
  - `start.sh`

### Что обязательно в окружении
- `OPENAI_API_KEY`
- SMTP переменные для лидов (`LEAD_SMTP_HOST`, `LEAD_SMTP_PORT`, `LEAD_EMAIL_FROM`, `LEAD_EMAIL_TO`, и при необходимости auth)
- `APP_ENV=prod` для блокировки debug-роутов
- `DEBUG_TOKEN` для непроизводственных сред

---

## 9) Безопасность и observability

- Debug endpoints закрыты в `prod`.
- `client_id` валидируется и в `/ask`, и в `/lead`.
- Логи структурированы (JSONL), чувствительные поля маскируются.
- LLM-сбой не роняет ответ пользователю (fail-soft fallback).

---

## 10) Что намеренно не делаем в этой версии

- CRM/n8n/webhook orchestration
- Полная multitenant-изоляция
- Большой рефакторинг `app.py`
- Передача `situation_note` в LLM

Это оставлено на следующий этап после demo-запуска.

---

## 11) Краткий итог

Текущая v4 архитектура — это рабочий RAG + сценарный backend, адаптированный под быстрый прод-демо запуск:
- точность: dual retrieval + alias + selective rerank + low-score guard;
- UX: детерминированная policy, корректная обработка коротких `yes`;
- эксплуатация: safe debug policy, lead email+fallback, single-worker runtime, PII-aware logging.
