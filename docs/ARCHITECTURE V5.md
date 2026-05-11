# v5 — Архитектура стоматологического бота

**Статус:** canonical reference. Этот документ — единственный источник истины по архитектуре v5. Любые расхождения между этим документом и кодом — баг кода, не документа.

**Девиз архитектуры:** *LLM-assisted deterministic dental agent.*

---

## 0. Принципы (сквозные, нерушимые)

1. **LLM используется только на 4 не-перекрывающихся уровнях:** Resolver, Arbiter, Generator, Verifier. Каждый — отдельный prompt, отдельный structured output schema, отдельный eval.
2. **Источник истины — детерминированный** (catalog, prices.json, выбранный chunk). LLM выбирает источник или проверяет ответ, но не «придумывает» факт.
3. **Никаких client-specific `if`** в коде. Различия между клиниками живут в `clients/{id}/`.
4. **Build-time vs runtime LLM разделены явно.** Auto-frontmatter — build. Resolver/Arbiter/Verifier/Generator — runtime.
5. **Триггеры вызова LLM — счётные или детерминированные** (количество кандидатов, presence-check на ответе). Никаких порогов «когда я думаю, что спорно».
6. **Generator получает ровно один источник** (массив длиной 1). Никакой агрегации между retrieval / catalog / alias.
7. **Каждый LLM-вызов имеет timeout и fallback в детерминированное правило.** Превышение или ошибка — не пропускают запрос, а активируют fallback своего слоя.
8. **Никаких числовых scoring-весов** в роутинге. Все «scoring» решения — либо детерминированные signals (containment, alias hit), либо LLM-arbiter.
9. **Все пороги вынесены в `core/routing.yaml`** (не в `config.py` константами). См. §D2.

---

## 1. Глоссарий контрактов

Контракты — Pydantic models в `contracts/`. Имплементация любого слоя начинается с проверки контракта, не с кода.

### 1.1 `DecisionFrame` (выход Resolver)

```yaml
route_intent:   content | price_lookup | price_concern | unknown
service_topic:  implantation | prosthetics | clinic | doctors | unknown
service_id:     str | null              # подсказка для catalog, не команда
query_mode:     overview | specific | comparison | process
confidence:
  intent:       float  # 0..1
  topic:        float
  service:      float  # для null = 0
  query_mode:   float
needs_clarification: bool
```

**Заметка про eligibility-вопросы.** Запросы про eligibility пациента (диабет, возраст, состояние здоровья и т.п.)
маркируются как `query_mode: specific`. Маршрут к релевантному контенту (например, `contraindications.md`) обеспечивается
retrieval по содержимому, а не отдельным классом в Resolver.

**Важно:** `route_intent` НЕ содержит `booking`, `contacts`, `handoff`, `lead_followup` — эти случаи отсекаются на A1 hard gates ДО вызова Resolver.

### 1.2 `GateTrace` (выход каждого hard gate)

```yaml
gate:        str              # имя gate (booking | contacts | rate_limit | ...)
passed:      bool             # true = pipeline shortcut'ится здесь
route:       str | null       # куда направить, если passed
payload:     dict | null      # готовый response, если есть
confidence:  float
source:      regex | rule | catalog | llm
reason:      str
```

### 1.3 `SourceRouteResult` (выход A3)

```yaml
source:       catalog_facts | catalog_md | price_card | price_ref | price_lookup_clarify
              | price_concern | doctor | contacts | none
service_id:   str | null
ref:          str | null       # для catalog_md = doc_id с приоритетом для retrieval
concern_ref:  str | null       # для price_concern: ref на md-чанк из каталога (аналог price_ref)
payload:      dict | null      # готовый payload для catalog_facts / price-карты / doctor
match_score:  float
match_method: catalog_containment | session_fallback | doctors_lookup | concern_default | none
```

Имплементация: при появлении ветки `concern_ref` в рантайме — расширить Pydantic `SourceRouteResult` в `contracts/source_route_result.py` в том же PR, что подключает чтение `concern_ref` из каталога.

### 1.4 `RetrievalCandidate` (элемент массива из A4)

```yaml
ref:              str         # doc_id#h3_id
doc_type:         str         # faq | service | info | pricing | doctor | contacts
subtype:          str | null
topic:            str
snippet:          str         # ≤500 chars
retrieval_score:  float
alias_hit:        bool        # exact alias popal или нет
in_scope:         bool        # совпал ли с DecisionFrame.service_topic
```

### 1.5 `ArbiterDecision` (выход A5)

```yaml
selected_ref:  str             # ref выбранного источника
confidence:    float
reason:        str             # human-readable
alternative:   str | null      # второй кандидат для логов
```

### 1.6 `VerifierVerdict` (выход A7)

```yaml
grounded:           bool
hallucinated_facts: [str]     # фразы из ответа, которых нет в источнике
confidence:         float
```

### 1.7 `SessionState` (читается всеми слоями)

```yaml
sid:                str
client_id:          str
history:            [{role, text, ts}]   # последние N turns
current_doc_id:     str | null
last_service_id:    str | null
covered_h3:         [str]     # H3 которые уже показывались в текущем topic
topic_turn_count:   int
lead_state:         null | collecting_name | collecting_phone | active
shown_boosters:     [str]     # id booster'ов, которые уже мелькали
```

---

## 2. Core Pipeline (A)

Линейный поток `/ask`. Восемь шагов, каждый — отдельный модуль.

```
turn → A1 hard_gates → A2 resolver → A3 source_routing
     → A4 scoped_retrieval → A5 arbiter → A6 generator
     → A7 verifier → A8 policy/ux → response
```

---

### A1. Hard Gates

**Цель.** Детерминированно отсечь то, где LLM не нужна и опасна.

**Состав (порядок проверки):**

1. `client_id` allowlist — отсутствует → 403.
2. Rate-limit, anti-spam (noise/burst/duplicate) — превышен → 429.
3. **Active lead-flow** — продолжается, минуя resolver/retrieval.
4. **Ref-button / quick-reply** — `get_chunk_by_ref`, минуя resolver.
5. `BOOKING_INTENT_RE` (regex + опциональный nano-LLM подтверждение) → lead-flow.
6. `CONTACTS_RE` → contacts payload (детерм. шаблон из `clinic__info__contacts.md`).
7. **Handoff filter** (LLM-gate, остаётся как есть) — off-topic / abuse / prompt-injection → handoff response.
8. **Catalog hard match** — exact phrase containment по aliases каталога. См. A3.1.

**Контракт:** каждый gate возвращает `GateTrace`. Если `passed=true` — pipeline shortcut'ится здесь, дальше не идёт.

**Меняем:** существующие `flow_handlers.py` + `policy.py` regex-константы остаются. Добавляется логирование `GateTrace` в pg_sink.

**Fallback:** ошибка LLM в booking/handoff confirmation → используется только regex-результат, без подтверждения.

**Зависимости:** нет.

**Готово когда:** все P0-сценарии (booking, contacts, ref-button, rate-limit) проходят без вызова Resolver/Retrieval.

---

### A2. Resolver / DecisionFrame

**Цель.** Один LLM-вызов, который понимает запрос. Заменяет `classify_intent`, поглощает часть `query_selector`.

**Новый модуль:** `resolver.py`. Модель — nano-class. Structured output (Pydantic).

**Контракт выхода:** `DecisionFrame` (см. §1.1).

**Принципы:**
- 4 query_mode, не больше. Эмоции, темы (боль/гарантия/материалы) — НЕ query_mode, они живут в контенте/тегах.
- `service_id` — подсказка для catalog, **не команда**. Catalog match через containment всегда сильнее.
- Confidence per-field — таблично, см. §D2.

**Кеширование:** TTL 60s по ключу `(q_norm, last_3_turns_hash, client_id)`.

## Known Issues (Resolver baseline, accuracy 90.9%)

После завершения Phase 1 PR #1.1:
- accuracy на golden = 40/44 = 90.9%
- 4 fail-кейса:
  - тонкая граница overview ↔ specific (1 кейс)
  - topic=\"clinic\" путается с \"unknown\" на узких темах (гарантия/оплата) (2 кейса)
  - intent price_lookup vs content для пограничного кейса «бесплатная консультация» (1 кейс)

Все 4 fail — низкого риска для пользователя при включённой safety-net (PR #1.2):
- topic=unknown триггерит wider retrieval
- intent=content не блокирует ответ на price-related запрос

Если production telemetry покажет каскадные плохие ответы из-за этих кейсов — пересмотреть в отдельном PR.\nДо этого — не оптимизировать.

**Меняем:** `classify_intent` ([llm.py:971](llm.py:971)) → DEPRECATED. Часть `query_selector.py` с regex-классификацией price ([:279](query_selector.py:279)) остаётся как fallback rule (см. ниже).

**Fallback:** LLM timeout / malformed JSON → `DecisionFrame{route_intent: unknown, confidence: all 0, needs_clarification: true}`. Дальше pipeline идёт по «unknown» ветке (wider retrieval).

**Зависимости:** A1 пройден.

**Готово когда:** Resolver eval ≥ 90% accuracy на размеченном golden set по каждому полю (intent / topic / query_mode отдельно).

---

### A3. Deterministic Source Routing

**Цель.** Если источник истины существует и определён — отвечать из него, не звать retrieval/arbiter.

**Контракт выхода:** `SourceRouteResult` (см. §1.3).

#### A3.1 Catalog Match

Три ветки по результату `match_service_from_catalog`:

| Условие | Действие |
|---|---|
| `match (containment ≥ 0.88)` И есть `facts` или `price_key` | **Hard route** в catalog/price. Минуем A4/A5, идём в A6 с `source=catalog_facts` или `price_card`. |
| `match (containment ≥ 0.88)` И есть только `md_entry_ref` (нет facts) | **Soft hint:** идём в A4 с приоритетом этого `md_entry_ref`. Дальше arbiter. |
| `no match` или `match < 0.88` | Обычный pipeline: A4 → A5. |

**Меняем:** `match_service_from_catalog` ([query_selector.py:297](query_selector.py:297)) — упростить:
- Убрать magic-band scoring.
- Оставить два сигнала: **exact phrase containment** и **lemma-subset**.
- Добавить wrapper stripping (`«вы делаете…»`, `«можно у вас…»`, `«есть ли…»`) ДО матчинга.
- Использовать также rewritten query (если есть) — берём `max(score_raw, score_rewritten)`.

#### A3.2 Price Lookup

Маршрутизация после матча услуги в каталоге (и/или объединения с fallback из сессии) — пять детерминированных веток:

| Ветка | Условие | Действие |
|---|---|---|
| **A3.2.1** | В каталоге/`prices.json` есть **`price_item`** для услуги | `build_price_lookup_payload` → ответ из шаблона с **конкретной цифрой** (A6), без LLM-генерации числа |
| **A3.2.2** | `price_item` нет, в каталоге задан **`price_ref`** (специализированный md) | `get_chunk_by_ref(price_ref)` → **LLM** генерирует ответ только из этого чанка |
| **A3.2.3** | `price_item` нет и **`price_ref` нет** | **`DEFAULT_PRICE_FALLBACK_REF`** = `clinic__info__payment_terms.md#korotko` → `get_chunk_by_ref` + в промпт для генератора добавляется **price-aware инструкция**: сначала признать отсутствие точной цены на услугу, затем кратко пересказать условия оплаты из материала; **не выдумывать цифры** |
| **A3.2.4** | `route_intent = price_concern` и в каталоге есть **`concern_ref`** | `get_chunk_by_ref(concern_ref)` → LLM из чанка (см. `concern_ref` в §1.3) |
| **A3.2.5** | Сервис в каталоге не найден (или нужно уточнение перед ценой) | `build_price_clarify_payload` — минимальный детерминированный шаблон без галлюцинации цен |

**Меняем:** существующий `select_price_service_route` + точка вызова в оркестраторе; реализация **A3.2.3** может предшествовать полному модулю `source_routing.py` (см. уже закрытый PR #1.2.7) — при вводе A3 свести с единым `SourceRouteResult`.

**Примечание:** `session.last_service_id` используется как fallback для коротких multi-turn («А сколько стоит?») до полного контрактного закрепления в PR Phase 1 (см. `KNOWN_DEBT.md`).

#### A3.3 Doctors Lookup

**Новый модуль:** `doctors_lookup.py`. Простой index по `doctors__doctor__*.md` (без RAG):
- match по имени врача (alias / lemma).
- match по специализации (`implantation` → all doctors с `specialty: implantation`).

Запросы «кто врач по X», «расскажите про доктора Y» идут сюда, минуя retrieval.

**Fallback:** не нашли — pipeline идёт в обычный retrieval (A4) с topic=doctors.

**Зависимости:** A2 (для подсказки service_id и topic).

**Готово когда:** все do_you_do запросы и прямые price-запросы НЕ доходят до retrieval. Все запросы про конкретного врача — через A3.3.

---

### A4. Scoped Retrieval

**Цель.** Векторный поиск по корпусу с обязательной topic-фильтрацией.

**Меняем:** `retriever.py` — добавить параметр `scope_topic` в `retrieve()` ([retriever.py:812](retriever.py:812)). Источник: `DecisionFrame.service_topic` при `confidence.topic ≥ resolver_min_confidence.topic` (см. §D2). Иначе — wider search.

**Упрощения:**
- 12-band alias-scorer ([retriever.py:532](retriever.py:532)) — **выкинуть**. Заменить на: exact alias hit (boolean signal) + cosine similarity по эмбеддингам.
- LLM rerank ([retriever.py:901](retriever.py:901)) — оставить, но возвращать `{choice, confidence, none}`. Низкая confidence → не выбирать насильно.
- Кеш retrieval по `(q_norm, scope_topic, client_id)` — оставить, TTL 120s.

**Контракт:** возвращает массив `RetrievalCandidate` (см. §1.4), top-K (по умолчанию K=5).

**Fallback:** scope дал 0 кандидатов → один retry с `scope_topic=null` (wider). Если и там 0 → A5 получает пустой массив → guided.

**Зависимости:** A2 (для scope_topic), A3 если был soft hint (приоритет `md_entry_ref`).

**Готово когда:** retrieval не возвращает кандидатов из чужого topic'а при уверенном scope. Eval по cross-topic (≥10 кейсов) — 100% in-scope.

---

### A5. LLM Arbiter

**Цель.** Выбрать один источник, когда есть несколько content-кандидатов.

**Новый модуль:** `arbiter.py`. Модель — mini-class. Structured output.

**Триггер вызова (явный счётный критерий):** arbiter зовётся ровно когда `|candidates| ≥ 2` из разных источников (retrieval + alias_leader + catalog soft hint).

**НЕ зовётся когда:**
- Hard gate сработал (A1).
- Catalog hard route (A3.1, ветка 1).
- Только один кандидат → shortcut в A6.
- Ноль кандидатов → guided.
- Active lead-flow.

**Контракт входа:**
```yaml
question: str
decision_frame: DecisionFrame
candidates: [RetrievalCandidate]   # с doc_type/subtype, чтобы arbiter видел тип
```

**Контракт выхода:** `ArbiterDecision` (см. §1.5).

**Принципы:**
- Arbiter возвращает **ref**, не текст. Не «улучшает» ответ. Только выбор источника.
- Видит `doc_type/subtype` каждого кандидата (FAQ specific vs service overview).

**Меняем:** все 7 if-rules в [content_arbiter.py:297-590](content_arbiter.py:297) → DEPRECATED. Остаётся только тонкий триггерный layer «считай кандидатов → если ≥2 зови arbiter, иначе shortcut».

**Кеширование:** не кешируется (вход слишком вариативен).

**Fallback:**
- LLM timeout / malformed → выбираем кандидата с максимальным `retrieval_score`, ставим `confidence: 0`, помечаем `reason: "arbiter_fallback"`.
- `confidence < arbiter_min_confidence` (см. §D2) → guided, не выбираем насильно.

**Зависимости:** A4.

**Готово когда:** на ambiguous-eval (диабет+имплантация, дёшево+услуга, и т.д.) source pick accuracy ≥ 85%.

---

### A6. Generator (single-source)

**Цель.** Сгенерировать ответ строго по выбранному источнику.

**Меняем:** `chunk_responder.py` + `llm.py:build_messages_for_gpt` — принимать **массив длиной 1**: `sources=[selected_chunk]`. Никакой агрегации между retrieval / catalog / alias.

**Принципы:**
- Persona/tone — из `clients/{id}/tone.yaml` (см. §D1), не хардкод.
- Никаких «улучшений» фактов: если в источнике нет — в ответе нет.
- Цены/числа/гарантии/сроки — подставляются из структурированных полей (catalog/prices) через шаблон, не выводятся LLM свободным текстом.
- Источник `catalog_facts` / `price_card` / `contacts` / `doctor` идут в шаблонный generator (без LLM), `catalog_md` / `retrieval` — в LLM-generator.

**Кеширование:** не кешируется.

**Fallback:**
- `selected_chunk == null` → honest fallback: «Не нашёл точного ответа, передам администратору / уточните на консультации».
- LLM timeout → тот же honest fallback.

**Зависимости:** A3 или A5 (источник выбран).

**Готово когда:** faithfulness eval ≥ 95% (нет фактов вне источника).

---

### A7. Verifier (high-risk only)

**Цель.** Перепроверить ответ на наличие фактов вне источника, когда цена ошибки высокая.

**Новый модуль:** `verifier.py`. Модель — nano-class.

**Триггер вызова — детерминированный, по тексту ответа:**

```python
trigger = (
    contains_number(answer)             # «3 года», «98%», «от 35 000 ₽»
    or contains_modal(answer)           # можно / нельзя / противопоказан / подходит
    or contains_time_promise(answer)    # сегодня / сразу / за 1 день
    or contains_warranty_claim(answer)  # гарантируем / гарантия
)
```

Если ни одного → verifier пропускается, экономим латентность и стоимость.

**Контракт выхода:** `VerifierVerdict` (см. §1.6).

**Действие при `grounded=false`:**
- Один retry generator'а с инструкцией «убери факты вне источника».
- Если и там `grounded=false` → деградация в honest fallback («Лучше уточню у администратора»).

**Кеширование:** не кешируется.

**Fallback:** LLM timeout → пропускаем verifier (pass-through). В лог пишем `verifier_skipped: timeout`.

**Зависимости:** A6.

**Готово когда:** hallucination rate на high-risk eval < 1%.

---

### A8. Policy / UX

**Цель.** Собрать финальный payload: ответ + кнопки + видео + CTA + situation.

**Остаётся детерминированным.** Никакой LLM в slot-логике.

**Источники UI-элементов:**
- **Followups (внутри документа):** `suggest_h3` из frontmatter — local navigation.
- **Cross-document refs, CTA, situation, video:** через Booster Engine (см. §B).
- Никаких `suggest_refs` из md (выпилены в B3).

**Меняем:** [policy.py](policy.py) — числа выносим в `clients/{id}/policy.yaml` (max_slots, video_first_turn, cta_from_turn defaults). Логика slot scheduling не меняется.

**Зависимости:** A6/A7 (ответ), B (boosters).

**Готово когда:** UI-payload собирается без обращения к raw frontmatter полям `suggest_refs` / `cta_text` / `video_key`.

---

## 3. Boosters & UX (B)

Замена ручных `suggest_refs` в md на декларативный движок.

### B1. Booster Registry

**Новый файл:** `clients/{id}/boosters.yaml`. Глобальный реестр конверсионных элементов клиента.

**Минимальная схема booster'а:**

```yaml
- id: pain_anesthesia              # уникальный
  label: "Как обезболивают"
  slot: followup                   # cta | followup | situation | media
  action_or_ref: "implantation__faq__pain.md#kakuyu-anesteziyu"
  topics: [implantation]
  query_modes: [specific]
  booster_tags: [pain]             # активируется на md с этим тегом
  min_topic_turn: 0
  priority: high                   # high | medium | low

- id: video_pain_explainer
  slot: media
  action_or_ref: "youtube://kdjf83xx"
  topics: [implantation]
  query_modes: [specific]
  booster_tags: [pain, fear]
  min_topic_turn: 0
  priority: high

- id: consultation_cta
  slot: cta
  action_or_ref: "lead"
  topics: [implantation, prosthetics]
  query_modes: [overview, qualification, comparison]
  min_topic_turn: 1
  priority: medium
```

**Принципы:**
- Без `emotion`/`stage` — потребует отдельных классификаторов, отложить.
- Без числовых `priority: 80` — ровно 3 уровня.
- `slot` — фиксированный enum.

**Зависимости:** A2 (query_mode/topic из DecisionFrame), A8.

---

### B2. Booster Engine

**Новый модуль:** `booster_engine.py`. Чисто детерминированный матчинг + опциональный LLM pick из shortlist.

**Pipeline:**

1. **Filter** candidates: пересечение `topics` ∩ DecisionFrame.topic, `query_modes` ∩ DecisionFrame.query_mode, `booster_tags` ∩ md.booster_tags, `min_topic_turn ≤ session.topic_turn_count`.
2. **Exclude:** уже показанные в этой сессии (`session.shown_boosters`), ref на текущий H3, конфликты slot.
3. **Rank:** `priority` (high > medium > low) → `min_topic_turn` (меньший раньше) → порядок в registry.
4. **Slot mutex:** один CTA, одна situation, одно медиа за turn.
5. **(Опционально) LLM pick:** если в shortlist >2 кандидата на followup-slot — mini-LLM выбирает 1-2 по критерию «снижают тревогу, ведут к консультации». Промпт явный, не subjective.

**Зависимости:** B1, A2, A8.

**Готово когда:** `suggest_refs` в frontmatter постепенно опустошены, кросс-навигация работает через registry.

---

### B3. Frontmatter миграция

**В md остаётся:**
- `aliases` — синонимы для матчинга.
- `suggest_h3` — локальная навигация внутри документа (НЕ через booster engine).
- `booster_tags` — метки для матчинга бустеров.
- `doc_type`, `subtype`, `topic`, `subtopic`, `empathy_tag`, `situation_allowed` — служебные.

**Уходит:**
- `suggest_refs` → boosters с `slot: followup`.
- `cta_text`, `cta_action`, `cta_from_turn` → boosters с `slot: cta` + `clients/{id}/policy.yaml` для дефолтов.
- `video_key` → boosters с `slot: media` + `booster_tags`.

---

## 4. Content Tooling (C, build-time)

Снижение ручного труда при добавлении md. **Не runtime** — выполняется один раз при добавлении/изменении контента.

### C1. Auto-Frontmatter Draft

**Новый CLI:** `tools/draft_frontmatter.py`. На входе — md без frontmatter. На выходе — draft frontmatter.

**LLM генерирует:**
- `doc_type`, `subtype`, `topic`, `subtopic` — по содержимому.
- `aliases` — 15-20 кандидатов.
- `suggest_h3` — по H3-структуре файла.
- `empathy_tag`, `situation_allowed` — по теме.
- `booster_tags` — по содержимому.

**НЕ генерирует:** ничего связанного с инфра (видео keys), брендом (cta_text), ценами.

---

### C2. Approve UI (минимальный)

**В `admin_dashboard/`:** страница «новый md» со списком auto-draft полей и approve/edit. Куратор кликает, не пишет с нуля.

**Принцип:** LLM покрывает 70% рутины, человек принимает 100% решений.

---

### C3. Corpus Linter

**Новый CLI:** `tools/lint_corpus.py`. Проверки:
- md без `aliases` или без секции `#korotko`.
- `service_id` без `md_entry_ref` И без `facts`.
- duplicate aliases между документами.
- orphan documents (никто не ссылается, не в каталоге).
- широкие aliases (одно слово < 3 символов).
- booster с `action_or_ref`, указывающим на несуществующий md/H3.

**Запускается в CI** на каждый PR в `clients/{id}/md/` или `boosters.yaml`.

---

## 5. Multi-Client Foundation (D)

Один core-код на всех клиентов. Различия — в данных, не в логике.

### D1. Client Pack Contract

**Структура `clients/{id}/`:**

```
client.yaml         # id, name, активные topics, контакты-meta
tone.yaml           # persona, style, forbidden_phrases
policy.yaml         # slot params, cta_from_turn defaults, situation_allowed_topics, routing_profile
service_catalog.json
prices.json
doctors.json
boosters.yaml
md/                 # md-файлы клиента
eval/               # client-specific eval кейсы (≥30)
```

**Минимальный пример `client.yaml`:**

```yaml
id: cesi_kamchatka
name: "ЦЭСИ"
city: "Петропавловск-Камчатский"
active_topics: [implantation, prosthetics, doctors, clinic]
contacts:
  phone: "+7 ..."
  whatsapp: "+7 ..."
  address: "..."
  hours: "..."
```

**Минимальный пример `tone.yaml`:**

```yaml
persona_name: "Анна"
role: "консультант клиники"
style: "спокойно, экспертно, без давления"
forbidden_phrases:
  - "гарантируем результат на 100%"
  - "это абсолютно безопасно"
disclaimers:
  consultation_free: true
  consultation_phrase: "На консультации врач разберёт ваш случай детально, она бесплатная."
```

**Минимальный пример `policy.yaml`:**

```yaml
slots:
  max_per_turn: 2
  video_first_turn_only: true
cta_from_turn:
  default: 1
  expensive_services: 2     # для service_id из списка
expensive_services: [all_on_4, all_on_6]
situation_allowed_topics: [implantation, prosthetics]
routing_profile: balanced   # см. D3
```

**Меняем:**
- Persona / клиника-name из [llm.py:291-301](llm.py:291) `BASE_SYSTEM` → `tone.yaml`.
- Числа из [config.py](config.py) (`cta_from_turn`, `max_slots`) → `policy.yaml` с дефолтами.
- Нигде в коде не должно быть `if client_id == "..."`.

---

### D2. Confidence Thresholds (`core/routing.yaml`)

Все пороги — в одном файле, один глобальный default. **Никаких per-client тюнингов.**

```yaml
resolver:
  min_confidence:
    intent:     0.7    # ниже → clarify / unknown ветка
    topic:      0.6    # ниже → wider retrieval (без scope)
    service:    ignored # catalog match через containment сильнее
    query_mode: 0.5    # ниже → defaults to 'specific'

arbiter:
  min_confidence: 0.6  # ниже → guided fallback

verifier:
  min_confidence: 0.7  # ниже → retry generator или honest fallback

retrieval:
  scope_topic_min_confidence: 0.6  # для включения scope-фильтра
  low_score_threshold: 0.33        # ниже → пустой массив кандидатов
  alias_scope_guard_min: 0.85      # при alias ≥ этого — не режем корпус по Resolver topic (конфликт-гард)

catalog_match:
  containment_min: 0.88            # для hard route
```

**Принцип:** Если порог нужно поменять — это PR в `core/routing.yaml`, не правка в коде. Любая константа в `*.py`, кроме самого `routing.yaml` loader'а, — баг.

---

### D3. Routing Profiles (отложено до 2-го клиента)

**Не делать** до появления второго клиента.

Когда появится — ввести 2-3 профиля в `policy.yaml`:

```yaml
routing_profile: balanced | conservative | sales_oriented
```

Каждый профиль — фиксированный набор параметров (cta_from_turn, clarify_threshold, booster slot limits). До тех пор `routing_profile: balanced` — единственный, неявный.

---

## 6. Observability & Eval (E)

Без этого блока v5 — чёрный ящик, который не дебажится.

### E1. Trace-Level Logging

**Меняем:** `pg_sink.py` — расширить event-уровень до trace-уровня. Каждый turn пишет в одну строку:

```yaml
turn_id, sid, client_id, ts
gate_traces: [GateTrace]
decision_frame: DecisionFrame
source_routing: SourceRouteResult
retrieval_candidates: [RetrievalCandidate]
arbiter_decision: ArbiterDecision | null
generator_input: {source_ref, prompt_hash}
verifier_verdict: VerifierVerdict | null
final_payload: {answer, ui_elements}
latency_ms: {gates, resolver, retrieval, arbiter, generator, verifier, total}
errors: [str]
```

**Новый view в `admin_dashboard/`:** «turn replay» — пройти по шагам одного turn'а с возможностью повторить любой шаг.

---

### E2. Per-Client Eval

**В каждом `clients/{id}/eval/`:** ≥ 30 кейсов на специфику клиента (его услуги, его цены, его врачи, его очепятки).

**CI:** прогон всех client eval перед merge. Fail если accuracy упала >2% по любому клиенту.

---

### E3. Per-Layer Eval

Каждый LLM-вызов в pipeline имеет свою метрику и свой golden set:

| Слой | Метрика | Golden set |
|---|---|---|
| Resolver | Match по полям DecisionFrame (intent / topic / query_mode отдельно) | `evals/v5/resolver_golden.json` (≥50) |
| Arbiter | Source pick accuracy на ambiguous кейсах | `evals/v5/arbiter_golden.json` (≥30) |
| Verifier | False-positive / false-negative groundedness | `evals/v5/verifier_golden.json` (≥20) |
| Generator | Faithfulness (% фактов в ответе, найденных в источнике) | `evals/v5/generator_golden.json` (≥30) |

**Без этого** общая «accuracy» не диагностируется — непонятно, кто промазал.

---

### E4. Content-Gap Dashboard

**В `admin_dashboard/`:** топ-N запросов с `route=guided` или `confidence < threshold`. Сгруппированы по теме. Каждый — кандидат в новый md или новый alias / booster.

**Польза:** видно дыры в покрытии без ручной разметки логов.

---

### E5. Hallucination Dashboard

**Метрики из verifier (A7):**
- % ответов с `grounded=false` (по дням, по клиентам, по топикам).
- Топ-фразы галлюцинаций.
- Регрессионный график (день / неделя / месяц).

---

## 7. Карта зависимостей

```
A1 (gates) ──┐
A2 (resolver) ──┼─→ A3 (source routing) ──┐
                │                          ├─→ A6 (generator) ──→ A7 (verifier) ──→ A8 (policy)
                └─→ A4 (scoped retrieval) ──→ A5 (arbiter) ──┘                                ↑
                                                                                              │
B1 (booster registry) ──→ B2 (engine) ───────────────────────────────────────────────────────┘
B3 (frontmatter migration) ──↑

C1 (auto-frontmatter) ──→ C2 (approve UI)         build-time, не блокирует runtime
C3 (corpus linter) ──→ CI

D1 (client pack)         ─── горизонтальный invariant для A, B
D2 (routing.yaml)        ─── единственное место всех порогов
D3 (profiles)            ─── отложено до 2-го клиента

E1-E5 (observability)    ─── поверх всего, обязательное условие готовности
```

---

## 8. Сводная таблица acceptance criteria

| Слой | Критерий готовности |
|---|---|
| A1 | P0 сценарии (booking, contacts, ref-button, rate-limit) проходят без вызова LLM-resolver/retrieval. |
| A2 | Resolver eval ≥ 90% accuracy на golden set по каждому полю. |
| A3 | Все do_you_do запросы и прямые price-запросы НЕ доходят до retrieval. Doctor lookup работает по 10 кейсам. |
| A4 | Retrieval не возвращает кандидатов из чужого topic'а при уверенном scope (cross-topic eval = 100% in-scope). |
| A5 | Source pick accuracy ≥ 85% на ambiguous-eval. |
| A6 | Faithfulness eval ≥ 95% (нет фактов вне источника). |
| A7 | Hallucination rate на high-risk eval < 1%. |
| A8 | UI-payload собирается без обращения к raw frontmatter полям `suggest_refs` / `cta_text` / `video_key`. |
| B | `suggest_refs` в frontmatter полностью выпилены. Кросс-навигация только через booster engine. |
| C | Новый md можно добавить за 5-7 минут (vs 30 минут руками). Linter в CI. |
| D | Persona вынесена. Все пороги в `core/routing.yaml`. Нет `if client_id == ...` в коде. |
| E | Каждый turn логируется на trace-уровне. Per-layer eval в CI. Hallucination dashboard работает. |

---

## 9. DEPRECATED (выпиливается в ходе v5)

| Что | Заменено на | PR |
|---|---|---|
| `llm.py:classify_intent` | `resolver.py` | TBD |
| `content_arbiter.py:select_content_route` (7 if-rules) | `arbiter.py` + триггер `|candidates|≥2` | TBD |
| `retriever.py:_alias_hit_score_raw_for_chunk` (12-band) | exact alias hit + cosine sim | TBD |
| `query_selector.py:_match_score` magic-band scoring | containment + lemma-subset | TBD |
| `ALIAS_STRONG_THRESHOLD`, `ALIAS_SOFT_THRESHOLD`, `LOW_SCORE_THRESHOLD` константы в `config.py` | `core/routing.yaml` | TBD |
| `BASE_SYSTEM` хардкод клиники в `llm.py:291-301` | `clients/{id}/tone.yaml` | TBD |
| `cta_from_turn` per-md в frontmatter | `clients/{id}/policy.yaml` defaults + boosters | TBD |
| `suggest_refs` в md | `clients/{id}/boosters.yaml` | TBD |

После Phase 3 (см. реализационную дорожку в `IMPLEMENTATION_PLAN.md`) этот раздел должен быть пустым.
