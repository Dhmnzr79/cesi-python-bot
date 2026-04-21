# Bot Architecture — актуальный срез (as-is)

Документ описывает текущую архитектуру бота в репозитории: как система реально работает сейчас, без исторических слоев и устаревших проблем.

---

## 1. Назначение системы

Бот отвечает на вопросы пользователей по контентной базе клиники (markdown + frontmatter), поддерживает сценарные переходы (CTA, ситуация, lead-flow), ведет сессию и формирует единый JSON-ответ для фронта.

Основной endpoint: `POST /ask`.

---

## 2. Модули и ответственность

### `app.py`
- Flask-роуты: `/ask`, `/lead`, `/_debug/ping`, `/__debug/retrieval`, `/static/...`.
- Точка входа запроса, валидация `client_id`, маршрутизация по веткам.
- Финализация ответа (`finalize_ask`), безопасная сериализация JSON, логирование HTTP.

### `query_selector.py`
- Оркестратор выбора чанка для ответа.
- Запускает rewrite (если включен), retrieval по исходному и переписанному запросу.
- Применяет broad-query корректировку, alias-приоритет, low-score защиту, selective rerank.

### `retriever.py`
- Загрузка корпуса и эмбеддингов, косинусный semantic search.
- Нормализация retrieval-запроса.
- Alias scoring (raw/lemma/trigram), определение broad query, предпочтение overview.
- `llm_rerank` (точечный LLM выбор между top-кандидатами).

### `llm.py`
- Переписывание query для retrieval с учетом истории диалога.
- Генерация финального текстового ответа с контекстом чанка.
- Эмпатия по триггерам и first-in-topic логике.
- JSON mode для генерации ответа (`{"answer": ...}`).

### `chunk_responder.py`
- Склейка основного контентного пути:
  1) чанк -> 2) LLM-ответ -> 3) сборка payload -> 4) policy -> 5) side-effects в session.
- Отметки `video_shown`, `situation_offered`, `cta_shown`, deferred refs и т.д.

### `policy.py`
- Детерминированные правила показа элементов UI:
  - followups,
  - quick refs,
  - video,
  - situation,
  - CTA.
- Интенты `contacts`, `price`, `booking`.

### `flow_handlers.py`
- Сценарные ветки вне контентного retrieval-пути:
  - lead-flow (имя -> телефон),
  - situation pending / back,
  - yes-routing после CTA и situation.

### `session.py`
- Единое состояние сессии в SQLite (`data/bot.db` по `SQLITE_PATH`).
- История диалога, профиль, topic_state, last bot action, сценарные флаги.

### `ux_builder.py`
- Единая сборка структуры ответа `/ask`.
- Формирование followups/quick refs/cta из metadata.
- UI-лимиты (не больше 2 followups и 1 quick ref).

### `meta_loader.py`
- Читает YAML frontmatter из `md/*.md`.
- Отдает doc metadata (`doc_type`, `suggest_h3`, `suggest_refs`, `video_key`, `situation_allowed`, `cta_*`, empathy flags).

### `build_index.py`
- Индексация markdown-контента в `data/corpus.jsonl` + `data/embeddings.npy`.
- Чанкование по H2/H3.
- Добавление aliases в embedding-текст.

### `lead_service.py`
- Прием и базовая валидация лида.
- Сохранение лида локально в `leads/*.json`.

### `logging_setup.py`
- JSONL-логирование в `logs/app.jsonl`.
- Санитизация чувствительных полей.

---

## 3. Контентный слой (markdown + YAML)

Источник знаний — `md/*.md`:
- текст секций в H2/H3;
- идентификаторы якорей (`{#...}`) для точного ref-routing;
- frontmatter-поля для навигации и сценариев.

Ключевые поля frontmatter:
- `doc_id`, `topic`, `subtopic`, `doc_type`;
- `aliases`;
- `suggest_h3`, `suggest_refs`;
- `cta_text`, `cta_action`;
- `situation_allowed`;
- `video_key`;
- `empathy_enabled`, `empathy_tag`.

Дополнительно поддерживаются локальные алиасы внутри блока через HTML-комментарии `<!-- aliases: [...] -->`.

---

## 4. Пайплайн обработки вопроса (`/ask`)

1. Принять JSON, валидировать `client_id`, получить `sid`.
2. Обработать быстрые сценарные ветки (reset, lead/situation/back/yes).
3. Если пришел `ref` -> взять чанк по ref и ответить по нему.
4. Если обычный `q`:
   - выбрать чанк через `query_selector.select_chunk_for_question()`;
   - при `low_score` вернуть безопасный fallback;
   - при успешном выборе вызвать `respond_from_chunk()`.
5. `respond_from_chunk()`:
   - получить meta документа;
   - сгенерировать ответ LLM;
   - собрать payload (`ux_builder`);
   - применить policy;
   - записать session side-effects;
   - вернуть унифицированный JSON.

---

## 5. Механика точности (retrieval quality)

### 5.1 Query rewrite
- Используется для retrieval, но не подменяет пользовательский вопрос в UI.
- Учитывает хвост истории (`hist`).
- Валидации rewrite:
  - reject маркеры утечки промпта,
  - overlap check с исходным вопросом,
  - fallback к исходному вопросу при невалидном rewrite.

### 5.2 Двойной retrieval и merge
- Поиск по `q_user` и по `q_rewrite_effective`.
- Объединение кандидатов с дедупликацией и сохранением лучшего score.

### 5.3 Broad query -> overview preference
- Короткие общие вопросы без “узких” терминов считаются broad.
- При совпадении top-файла приоритет смещается к overview-секции.

### 5.4 Alias layer: словоформы и опечатки
- Alias score для чанка = `max(raw, lemma, trigram)`.
- `lemma` канал использует `pymorphy3` (если установлен).
- `trigram` канал поднимает recall на опечатках и близких формах.
- Strong alias может выбирать чанк напрямую, soft alias может спасти low-score кейс.

### 5.5 Selective rerank
- Rerank вызывается только в узкой зоне неоднозначности (по score и gap).
- Если условия не выполнены, бот не тратит вызов rerank и берет semantic top.

### 5.6 Low-score guard
- При слишком низком top similarity бот не отвечает “из ничего”.
- Возвращается безопасный fallback с просьбой уточнить вопрос и CTA.

---

## 6. Сценарии и state-machine

### 6.1 Lead-flow
- Входы:
  - явный booking intent,
  - CTA action = `lead`,
  - “да” после предложенного CTA.
- Шаги:
  - `collecting_name`,
  - `collecting_phone`,
  - `submitted`.

### 6.2 Situation-flow
- Показ кнопки ситуации управляется policy + `situation_allowed`.
- `situation_pending` переключает экран в режим ввода ситуации.
- После получения ситуации:
  - note сохраняется,
  - бот переводит пользователя в lead-flow (шаг имени).
- Поддержан `situation_action=back` с восстановлением последнего контентного экрана.

### 6.3 Короткие ответы ("да")
- Трактовка опирается на `last_bot_action` из session:
  - `offered_cta` + yes -> lead-flow,
  - `offered_situation` + yes -> situation pending.

---

## 7. Память и состояние сессии

Хранение: SQLite (`sessions`), ключ — `sid`.

Основные поля:
- `hist`, `profile`, `session_turn_count`;
- `current_doc_id`, `topic_state`;
- `last_bot_action`, `last_offer_type`, `last_presented_buttons`;
- `situation_pending`, `situation_note`;
- `lead_intent`;
- `last_content_ui_payload` (для restore после back).

Topic-state по `doc_id`:
- `doc_turn_count`,
- `covered_h3_ids`,
- `video_shown` / `video_pending`,
- `situation_offered`,
- `suggest_ref_used`,
- `refs_deferred`,
- `cta_shown`.

---

## 8. Policy и UI-слоты (текущее поведение)

Policy принимает готовый payload и решает итоговую видимость:
- какие `followups` оставить;
- показывать ли `video`;
- показывать ли `situation`;
- показывать ли `quick_replies` (refs);
- показывать ли `cta`.

Базовые принципы:
- спецрежимы (lead/situation pending) имеют приоритет;
- не показывать уже исчерпанные элементы;
- учитывать `doc_turn_count` и состояние темы;
- не показывать CTA слишком рано (до нужного шага темы) и при активных конфликтующих режимах.

UI-нормализация дополнительно ограничивает:
- followups: максимум 2,
- quick refs: максимум 1.

---

## 9. API-контракт ответа `/ask` (текущий)

Базовая форма:

```json
{
  "answer": "...",
  "quick_replies": [{ "label": "...", "ref": "file.md#anchor" }],
  "cta": { "text": "...", "action": "lead" },
  "video": { "key": "..." },
  "situation": { "show": true, "mode": "normal|pending" },
  "offer": null,
  "meta": {
    "file": "...",
    "h2_id": "...",
    "h3_id": "...",
    "score": 0.0,
    "followups": [{ "label": "...", "ref": "..." }],
    "sid": "...",
    "client_id": "...",
    "policy_decision": { "...": "..." }
  }
}
```

Для фронта критичны:
- `answer`,
- `quick_replies`,
- `cta`,
- `situation.show` + `situation.mode`,
- `meta.followups`.

---

## 10. Видео: статус

- На backend видео контролируется через `video_key` + policy.
- На текущем этапе видео канал считать тестовым/вспомогательным (UI-заглушка допустима).
- Основной фокус качества — retrieval точность и сценарные переходы.

---

## 11. Мультиклиентность: текущее и перспектива

### Сейчас
- `client_id` валидируется на входе.
- В retrieval есть фильтрация кандидатов по `client_id`.
- Система функциональна для текущего single-client/ограниченного режима.

### Перспектива
- Полная tenant-изоляция:
  - контент: `md/{client_id}/...`,
  - индексы: `data/{client_id}/...`,
  - отдельные эксплуатационные контуры и аналитика.
- Это roadmap-задача масштабирования, не блокер текущей точности одного клиента.

---

## 12. Операционный контур

- Логи: JSONL (`logs/app.jsonl`), структурированные события.
- Отладка retrieval: `/__debug/retrieval`.
- Dev-инспектор JSON/retrieval: `static/debug/ask-inspector.html` (временно; папку `static/debug/` можно удалить).

---

## 13. Итог

Текущая архитектура — рабочий RAG + сценарный backend с детерминированной обработкой ключевых переходов и многоступенчатой защитой точности (rewrite, aliases, broad-query preference, selective rerank, low-score guard).

На текущем этапе для продукта приоритет:
1. стабильная точность попадания в релевантные чанки;
2. предсказуемое поведение сценариев (CTA / ситуация / lead-flow);
3. чистая интеграция фронта с текущим контрактом `/ask`.

