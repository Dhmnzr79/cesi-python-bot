# Bot Architecture v4 — текущее состояние

Документ описывает актуальную архитектуру после серии hardening-итераций перед demo-prod запуском.

По дашборду и логированию отдельный рабочий документ: `docs/DASHBOARD_LOGGING_PLAN.md`.

---

## 1) Цель версии

- Запуск demo-версии для одного клиента без платформенных задач (CRM/n8n/multitenant/admin).
- Точный, предсказуемый routing: LLM-классификатор вместо разрозненных regex-веток.
- Стабильный `/ask`, безопасный `/lead`, минимум операционных рисков.

---

## 2) Основные роуты

### `POST /ask`
Основной контентный и сценарный endpoint. Полный пайплайн описан в разделе 5.

### `POST /lead`
- Валидация JSON и `client_id`.
- Единый технический контракт ответа: `ok`, `error_code`, `delivery`.

### Debug роуты
- `/_debug/ping`, `/__debug/retrieval`
- При `APP_ENV=prod` → `404`, иначе требуют `X-Debug-Token`.
- Мини-дашборд последних `bot_event`: `GET /dashboard` (HTML) и `GET /dashboard/events` (JSON). В `prod` нужен заголовок/параметр `X-Dashboard-Token` / `?token=` — значение из `DASHBOARD_TOKEN` или, если не задан, `DEBUG_TOKEN`. Запросы к `/dashboard*` не логируются как `http_request`, чтобы не засорять JSONL автопереобновлениями.

---

## 3) Модули и ответственность

### `app.py`
HTTP-граница, валидация `client_id`, маршрутизация по intent, финализация ответа.

### `llm.py`
- `classify_intent(q)` — LLM-классификатор намерения: `contacts / price_lookup / price_concern / content`. Fallback → `content`.
- `rewrite_query_for_retrieval()` — переформулировка вопроса для семантического поиска (анафора, контекст диалога). Модель: `gpt-5.4-nano`, без temperature.
- `generate_answer_with_empathy()` — генерация ответа из чанка с учётом эмпатии. Модель: `gpt-5.4-mini`.
- `classify_booking_wants_appointment()` — бинарный классификатор намерения записаться. Модель: `BOOKING_INTENT_LLM_MODEL` (по умолчанию `gpt-5.4-nano`).

### `query_selector.py`
- `select_chunk_for_question()` — основной retrieval: dual-query merge → alias assist → reranker → возврат чанка.
- `select_price_service_route()` — ценовой routing: матч услуги из каталога/сессии + intent_override.
- `select_catalog_content_route()` — facts-карточка для услуг без MD-страницы (КТ, отбеливание).

### `retriever.py`
- Semantic retrieval + три alias-канала (raw / lemma / trigram).
- `llm_rerank(q, cands)` — LLM выбирает лучший чанк из топ-3. Модель: `gpt-5.4-mini`.
- `get_chunk_by_ref(ref)` — поиск чанка по `filename.md#anchor`. Принимает оба формата: с `.md` и без.

### `policy.py`
Детерминированное управление UI-элементами: followups, refs, video, situation, CTA. Порог CTA через `cta_from_turn` в frontmatter.

### `flow_handlers.py`
Сценарные ветки: lead, situation, back, yes. Booking-intent перехватывается здесь — до `classify_intent`.

### `chunk_responder.py`
Контентный пайплайн: chunk → LLM answer → policy → session side-effects → JSON.

### `session.py`
SQLite session state. Хранит `current_doc_id`, `last_catalog_service_id`, историю диалога, topic state, `client_id` (заполняется при `_bind_chat_ctx` на `/ask` и `/lead`).

### `meta_loader.py`
Загрузка frontmatter MD-файлов. `doc_id` — имя файла без `.md` если не задан явно.

### `lead_service.py`
Email-доставка лида. При ошибке — file fallback (`leads/*.json`).

### `logging_setup.py`
JSONL-логирование. Маскирование PII (phone), санитизация секретов. Продуктовые события с `kind="bot_event"` и `schema_version` (см. эмиттер `emit_bot_event`): `user_turn_completed`, `bot_reply_completed`, `cta_shown`, `lead_submitted`, `retrieval_selected`, `retrieval_fallback`, `llm_usage`, `llm_error`; в HTTP-контексте в лог подставляются `request_id`, `sid`, `client_id`, `path`. Оценка USD в `llm_usage` — опционально через `BOT_LLM_USD_PER_1M_PROMPT` / `BOT_LLM_USD_PER_1M_COMPLETION` в `.env`.

---

## 4) Контент: MD-файлы и service_catalog.json

### MD-файлы (`md/*.md`)
Ключевые поля frontmatter:
- `doc_id`, `topic`, `subtopic`
- `aliases` — фразы для alias-retrieval (три канала: raw/lemma/trigram)
- `suggest_h3`, `suggest_refs` — followup-ссылки
- `cta_text`, `cta_action`, `cta_from_turn`
- `empathy_enabled`, `situation_allowed`, `video_key`

Формат H3-якорей для suggest: `### Заголовок {#anchor-id}`, для alias внутри раздела: `<!-- aliases: [...] -->`.

### `clients/default/service_catalog.json`
Каталог услуг. Ключевые поля:
- `aliases` — фразы для матча услуги в ценовом routing
- `md_entry_ref` — ссылка на MD-страницу (null = facts-карточка)
- `price_key` — ключ в `prices.json`
- `concern_ref` — ref в формате `filename.md#anchor` для routing `price_concern` → content-чанк с объяснением стоимости
- `response_mode: "card"` — принудительная facts-карточка

### Формат ref
Везде используется формат `filename.md#anchor` (например, `implantation__faq__cost.md#korotko`).
`get_chunk_by_ref` принимает оба варианта — с `.md` и без, добавляет расширение автоматически.

---

## 5) `/ask` — пайплайн

```
Запрос
  ↓
Валидация client_id
  ↓
handle_flows() — booking / lead / situation / back / yes / followup-redirect
  ↓ (если не перехвачено)
ref из тела запроса → get_chunk_by_ref → respond_from_chunk
  ↓ (если нет ref)
_is_short_contextual? → get_chunk_by_ref(current_doc_id#korotko) → respond_from_chunk
  ↓ (если не короткая реплика)
classify_intent(q) → contacts | price_lookup | price_concern | content
  ↓
contacts    → retrieve(topk=4) + pick_contacts_chunk → respond_from_chunk
             (fallback: продолжить в retrieval если chunk не найден)

price_lookup/
price_concern → select_price_service_route(intent_override=intent)
               ├── price_concern + concern_ref → get_chunk_by_ref → respond_from_chunk
               ├── price_concern без concern_ref → build_price_concern_payload
               ├── price_lookup + prices_json → build_price_lookup_payload
               └── no match → build_price_clarify_payload

content     → select_catalog_content_route
             ├── facts (md_entry_ref=null) → build_service_facts_card_payload
             └── no match → select_chunk_for_question (retrieval)
  ↓
select_chunk_for_question
  ├── dual retrieval (primary + rewrite query) → merge → topk=8
  ├── prefer_overview_if_broad
  ├── low_score guard (< 0.33)
  │   └── soft_alias_assist если alias_score >= 0.72
  ├── contacts/prices детерминированный pick
  └── reranker: top_score ∈ [0.33, 0.75) AND score_gap < 0.15
      └── llm_rerank(top-3) → выбор лучшего чанка
  ↓
respond_from_chunk → LLM answer → policy → session → JSON
```

---

## 6) Retrieval и reranker

### Dual retrieval
Два запроса: исходный (`q`) и переформулированный (`q_rewrite`). Результаты мержатся, дедупликация по `(file, h2_id, h3_id)`.

### Alias-каналы
Три канала параллельно: raw (точное совпадение), lemma (лемматизация), trigram (нечёткое). Alias soft assist: если top_score < LOW_SCORE_THRESHOLD, но alias_score >= 0.72 — возвращает alias-чанк вместо low_score fallback.

### Reranker (always-on с gap-guard)
Включается когда:
- `top_score >= 0.33` (выше low_score порога)
- `top_score < 0.75` (не очевидный победитель)
- `score_gap < 0.15` (разрыв между #1 и #2 мал — есть неопределённость)
- не literal-point query

Модель: `gpt-5.4-mini`. Выбирает из топ-3 кандидатов.

---

## 7) Intent routing

`classify_intent()` вызывается после flow_handlers и _is_short_contextual. Booking обрабатывается раньше в flow_handlers — в classify_intent не попадает.

| Intent | Условие | Действие |
|--------|---------|----------|
| `contacts` | адрес, телефон, график | pick_contacts_chunk из retrieve(topk=4) |
| `price_lookup` | сколько стоит, цена | select_price_service_route → цена из prices.json |
| `price_concern` | дорого, не потяну | concern_ref → cost.md чанк, иначе generic payload |
| `content` | всё остальное | select_catalog_content_route или retrieval |

---

## 8) Модели

| Задача | Модель |
|--------|--------|
| Эмбеддинги | `text-embedding-3-large` |
| Генерация ответа | `gpt-5.4-mini` |
| Reranker | `gpt-5.4-mini` |
| Intent classifier | `gpt-5.4-mini` |
| Query rewrite | `gpt-5.4-nano` |
| Booking classifier | `gpt-5.4-nano` |
| Lead name classify | `gpt-5.4-mini` |

---

## 9) Session state

Ключевые поля в SQLite:
- `client_id` — последний известный `client_id` с `/ask` и `/lead` (для дашборда)
- `current_doc_id` — doc_id последнего отвеченного чанка (имя файла без `.md`)
- `last_catalog_service_id` — последняя услуга из каталога
- `hist` — история диалога (последние N реплик для rewrite)
- `topic_state[doc_id]` — doc_turn_count, covered_h3_ids, cta_shown, video_shown

---

## 10) Lead контур

`handle_lead()` возвращает:
- `ok=true, delivery="email"` — доставлено
- `ok=true, delivery="file_fallback"` — email упал, сохранено в `leads/*.json`
- `ok=false, delivery=null` — не обработано

---

## 11) Runtime

- **Single-worker**: SQLite → только 1 воркер gunicorn
- **Entrypoint**: `gunicorn -w 1 -b 0.0.0.0:8000 app:app`
- **Обязательные env**: `OPENAI_API_KEY`, SMTP-переменные, `APP_ENV=prod`

---

## 12) Что намеренно не делаем в v1.0

- CRM / n8n / webhook интеграции
- Полная multitenant-изоляция (один клиент `default`)
- Большой рефакторинг `app.py`
- Стриминг на бэкенде
