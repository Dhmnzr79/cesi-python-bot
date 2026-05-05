# Dashboard & Logging Plan (v1)

Документ фиксирует целевую архитектуру дашборда и события бота, чтобы не строить UI поверх "сырого хвоста JSONL".

---

## 1) Цель

Дашборд должен отвечать на бизнес-вопросы:

- Бот сегодня работает нормально?
- Где и почему он ошибается?
- Какие вопросы повторяются и требуют доработки базы знаний?
- Сколько лидов принес бот?
- Сколько стоит работа бота?
- Какие диалоги требуют внимания человека?

---

## 2) Принцип

Разделяем слои:

- Технический слой: события (`bot_event`) и низкоуровневые логи.
- Продуктовый слой: сущности и агрегаты (диалог, проблема, стоимость, лид).

JSONL остается источником отладки и резервом. Для аналитики и дашборда данные хранятся в PostgreSQL.

---

## 3) Этапы внедрения

## Этап 1 (MVP, без усложнений)

1. Пишем события в PostgreSQL одновременно с JSONL.
2. Минимальные таблицы:
   - `bot_events` (append-only, JSONB details)
   - `leads` (структурированные лиды)
3. Добавляем событие `turn_complete` с redacted-полями (без сырого текста):
   - `user_text_redacted`
   - `user_preview_redacted`
   - `bot_text_redacted` (или `bot_preview_redacted` в урезанном режиме)
4. Дашборд MVP:
   - Обзор
   - Диалоги (лента/карточки)
   - Проблемы (fallback/error/аномалии)
   - Лиды
   - Стоимость
   - Event Explorer (тех-экран)

## Этап 2 (после накопления данных)

1. Materialized views для тяжелых агрегатов.
2. Опционально таблица `llm_calls` для детальной cost-аналитики.
3. Политика хранения и ротации исторических данных.

---

## 4) Контракт событий (минимум)

Все продуктовые события:

- `kind = "bot_event"`
- `event_type` (тип события, единое имя для payload и PostgreSQL)
- `schema_version`
- `ts`
- `request_id`
- `sid`
- `client_id`
- `path`
- `status` (`ok`/`error`/null)
- `details` (JSON-объект)

Минимальный набор событий:

- `user_turn_completed`
- `bot_reply_completed`
- `turn_complete` (полные тексты)
- `lead_submitted`
- `llm_usage`
- `llm_error`
- `retrieval_selected`
- `retrieval_fallback`
- `cta_shown`

---

## 5) Событие `turn_complete` (обязательное для карточки диалога)

Назначение: "один завершенный ход = одна запись" для экрана диалога.

Обязательные `details`:

- `turn_number`
- `user_text_redacted` (текст пользователя после redaction, с лимитом длины)
- `user_preview_redacted` (короткий redacted preview)
- `bot_text_redacted` (redacted текст ответа, с лимитом длины)
- `intent`
- `doc_id` (или null)
- `route` (фиксированный enum)
- `low_score` (bool)
- `lead_flow` (bool)
- `handoff_filter` (bool)
- `answer_chars`
- `latency_ms`

Опциональные `details`:

- `selected_ref`
- `retrieval_score` (chosen score)
- `retrieval_score_original`
- `fallback_reason`
- `cta_action`

Для stream (`/ask/stream`) событие пишется только после финальной сборки ответа, не по дельтам.

Важно: `route` формируется в одном месте обработки запроса (локальная переменная контекста), а не собирается постфактум из нескольких `log_json`-событий.

Важно: сырые полные тексты (`user_text`, `bot_text`) в `bot_event` для MVP не пишем.
Важно: redaction применяется до отправки в storage (`bot_event`) и не зависит только от logger-sanitizer.

---

## 6) `route` enum (фиксированный список)

Рекомендуемый список:

- `retrieval_chunk`
- `retrieval_no_candidates`
- `low_score_fallback`
- `price_lookup`
- `price_concern` (если нужно отдельно анализировать возражения по цене)
- `catalog_facts`
- `contacts_chunk`
- `flow_redirect_ref`
- `lead_flow`
- `booking_flow`
- `handoff_filter`
- `noise_short_circuit`
- `duplicate_short_circuit`
- `rate_limited`
- `offtopic`
- `error`

Единый enum нужен для стабильных SQL-группировок в дашборде.

Примечание для MVP: допустимо свернуть `price_lookup` и `price_concern` в единый `price_route`, если детализация пока не нужна.

---

## 7) PostgreSQL: минимальная схема для старта

### `bot_events`

- `id` (bigserial pk)
- `occurred_at` (timestamptz, default now)
- `kind` (text)
- `event_type` (text)
- `schema_version` (int)
- `request_id` (text)
- `sid` (text)
- `client_id` (text)
- `path` (text)
- `status` (text null)
- `details` (jsonb)

Индексы:

- `(occurred_at desc)`
- `(client_id, occurred_at desc)`
- `(sid, occurred_at asc)`
- `(event_type, occurred_at desc)`
- `(request_id)`
- GIN по `details` при необходимости

### `leads`

- `id` (bigserial pk)
- `captured_at` (timestamptz)
- `request_id` (text)
- `sid` (text)
- `client_id` (text)
- `name` (text)
- `phone` (text)
- `topic` (text)
- `cta_action` (text)
- `turns_to_lead` (int)
- `delivery_status` (text)

---

## 8) Производительность и надежность

- Запись в БД не должна блокировать ответ пользователю.
- Для MVP допускается fire-and-forget, но ошибки записи должны логироваться.
- JSONL остается fallback-источником при проблемах БД.
- Технический флаг запуска sink: `BOT_PG_DSN` (если пустой, запись только в JSONL).
- При временных ошибках БД sink переотправляет событие ограниченное число раз (`BOT_PG_MAX_RETRY`).
- Если драйвер PostgreSQL недоступен, sink отключается и бот продолжает работу на JSONL без очереди.

---

## 9) Безопасность и PII

Текущий режим доступа: только владелец и коллега.

Минимум:

- маскировать телефон в UI (например, `+7 XXX XXX 89`)
- не показывать лишние PII на общих экранах

Для будущего расширения:

- хранить `client_id` в каждой таблице
- фильтровать все запросы дашборда по `client_id`

---

## 10) Что НЕ делать на старте

- Не строить тяжелые агрегаты отдельными апдейтами "на каждый event".
- Не делать "BI-ради BI" до появления стабильного потока данных.
- Не смешивать экран Event Explorer с главной страницей.

---

## 11) История бага по токенам (`resolved`)

В `logging_setup.py` текущая санитизация проверяет чувствительные ключи по подстроке:

- `any(s in key_lower for s in SENSITIVE_KEYS)`

Из-за этого поля:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

маскируются как `***`, потому что содержат подстроку `token`.

Следствие: `llm_usage` теряет реальное количество токенов, и стоимость в дашборде получается некорректной.

Минимальное исправление:

- перейти с подстрочного матча на точное совпадение чувствительных ключей, либо
- добавить явный allowlist для `prompt_tokens/completion_tokens/total_tokens`.

Статус: fixed в коде (`logging_setup.py`): проверка чувствительных ключей переведена на точное совпадение.

---

## 12) Definition of Done (MVP)

MVP считается готовым, если:

1. События пишутся и в JSONL, и в PostgreSQL.
2. Есть `turn_complete` с полными текстами.
3. Главная страница показывает метрики "сегодня" без парсинга JSONL.
4. Карточка диалога показывает реплики user/bot и маршрут (`route`).
5. Экран "Проблемы" показывает fallback/error/подозрительные кейсы.
6. Экран "Лиды" показывает статусы доставки и связь с диалогом.
