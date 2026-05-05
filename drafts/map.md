# Финальный план: routing/retrieval accuracy без усложнения бота

Цель:
повысить точность ответов на типичные вопросы пациента и убрать конфликт catalog vs retrieval.

Главный принцип:
conversion-first, simple-by-design.

Бот должен:
- точно отвечать на частые вопросы;
- мягко вести к заявке;
- не усложняться ради редких нецелевых кейсов;
- не превращаться в multi-agent/multi-intent систему без необходимости.

---

# Сейчас НЕ делаем

- не делаем большой дашборд;
- не делаем полный eval на 80-150 вопросов;
- не делаем отдельную safety/complaint-архитектуру;
- не объединяем handoff_filter и classify_intent;
- не делаем meta-LLM “выбери лучший кандидат”;
- не добавляем keyword-костыли под отдельные фразы;
- не делаем новый broad_query intent;
- не добавляем второй LLM-классификатор внутри intent=content;
- не чистим весь legacy/compat в этом же этапе.

---

# P0. Минимальный routing smoke eval

Перед изменением routing сделать маленький smoke-набор на 20-30 вопросов.

Это не большой eval-проект, а защита от регрессий.

Формат кейса:
- q;
- expected_route;
- expected_doc_id или expected_topic;
- forbidden_doc_id, если есть;
- короткий критерий правильного ответа.

Примеры вопросов:
- “какая приживаемость имплантов?”
- “риск отторжения импланта”
- “больно ли ставить имплант?”
- “сколько стоит имплант?”
- “расскажите про классическую имплантацию”
- “сколько длится приживление?”
- “если кости мало?”
- “у меня нет зуба”
- “какой врач ставит импланты?”
- “где вы находитесь?”
- “хочу записаться”
- “болит зуб”
- “коронки на жевательные зубы”
- “импланты”
- “дорого”
- “а долго?”
- “а больно?”
- “а если не приживется?”

Важно:
запустить smoke-набор до и после P0-изменений.

---

# P0. Исправить конфликт catalog md_first vs retrieval

Проблема:
при intent=content catalog может перехватить вопрос раньше retrieval.

Пример:
вопрос “какая приживаемость имплантов?”
retrieval находит FAQ про приживаемость,
но catalog видит широкий alias “имплантация” и может увести в service overview.

Что нужно:
- catalog md_first не должен безусловно отвечать до retrieval;
- catalog md_first должен быть кандидатом или fallback;
- если retrieval нашел точный FAQ/chunk с хорошей уверенностью, он должен побеждать;
- price/facts routing не ломать.

Ожидаемое поведение:
- “какая приживаемость имплантов” → implantation__faq__osseointegration
- “риск отторжения импланта” → implantation__faq__osseointegration
- “расскажите про классическую имплантацию” → implantation__service__classic
- “сколько стоит имплантация” → price route

---

# P0. Deterministic content arbiter

Область:
только intent=content.

Не трогать:
- price routing;
- contacts;
- lead-flow;
- handoff;
- /lead.

Арбитр должен быть детерминированным.

Запрещено:
- meta-LLM “выбери лучший кандидат”;
- новый LLM-классификатор внутри content;
- keyword-исключения под конкретные фразы.

Форма:
одна функция, например:

select_content_route(q, sid, client_id) -> ContentRouteResult

Эта функция должна использоваться и в /ask, и в /ask/stream.

ContentRouteResult:
- kind: chunk | service | guided | fallback
- selected_route
- selected_chunk или payload
- selected_doc_id
- reason
- confidence/debug_meta
- candidates/rejected_candidates

Важно:
арбитр должен уменьшить хаос в app.py, а не добавить новую пачку if-веток в оба endpoint-а.

---

# P0. Candidate collection для content-вопросов

Для intent=content собрать кандидатов:

1. retrieval_candidate
   - результат select_chunk_for_question;
   - top chunk;
   - top_score;
   - doc_id;
   - h2/h3;
   - doc_type/subtype/topic.

2. catalog_candidate
   - результат select_catalog_content_route;
   - mode;
   - matched_service_id;
   - match_score;
   - md_entry_ref;
   - service title.

3. alias_candidate
   - alias_leader;
   - alias_score;
   - alias source/type, если можно определить.

4. session_context_candidate
   - current_doc_id;
   - last_catalog_service_id;
   - текущая тема, если есть.

Важно:
- select_chunk_for_question уже делает retrieval + (при необходимости) rerank;
- candidate collector и arbiter не должны запускать второй поиск;
- один проход retrieval/rerank, один проход catalog, один проход alias/session context;
- результат арбитра используется и в `/ask`, и в `/ask/stream`.

---

# P0. Selection rules

Не сравнивать catalog_score и retrieval_score напрямую как числа.
Сейчас это разные шкалы.

Арбитр сравнивает:
- тип кандидата;
- специфичность;
- confidence zone;
- doc relationship.

Правила:

1. FAQ/specific retrieval wins over broad service catalog

Если retrieval_candidate указывает на FAQ/specific chunk,
а catalog_candidate указывает на широкий service overview,
выбираем retrieval.

Пример:
“приживаемость имплантов” → FAQ, не classic service overview.

2. Same doc: specific H3 wins over overview

Если catalog и retrieval указывают на один документ,
но retrieval нашел более точный H3,
выбираем retrieval chunk.

3. Strong catalog wins only when retrieval is weak

Catalog md_first может победить, если:
- retrieval слабый;
- нет specific FAQ alias;
- catalog match уверенный;
- вопрос похож на service overview.

Пример:
“расскажите про классическую имплантацию” → service overview.

4. Weak all candidates → guided UX / clarify

Если все кандидаты слабые:
- не падать автоматически в catalog;
- дать короткий guided ответ с кнопками.

Пример:
“импланты”

Возможный ответ:
“Могу подсказать по стоимости, срокам, боли, приживаемости или записи. Что для вас важнее?”

Кнопки:
- Стоимость
- Больно ли
- Сроки
- Подходит ли мне
- Записаться

5. LLM rerank остается узким механизмом

Не использовать LLM как главный арбитр.

LLM rerank можно оставить только в текущем узком сценарии:
- спор внутри retrieval candidates;
- score в средней зоне;
- score_gap маленький;
- как сейчас.

---

# P0. Alias specificity

Широкие alias услуг:
- “имплант”
- “имплантация”
- “зубы”
- “протезирование”

не должны перебивать точные FAQ alias:
- “приживаемость имплантов”
- “риск отторжения”
- “больно ли ставить имплант”
- “сколько длится приживление”

Что нужно:
- учитывать специфичность alias в арбитраже;
- broad service alias считать слабее FAQ-specific alias;
- не делать ручные исключения под конкретные фразы;
- решение должно переноситься на другие услуги и клиники.

Возможный подход:
- alias из 1 общего слова → broad_service;
- alias из 2-4 смысловых слов → specific;
- FAQ/document alias выигрывает у service overview при близкой уверенности.

Важно:
- не добавлять ручные keyword-исключения под отдельные фразы;
- правила должны переноситься на другие услуги и клиники.

---

# P0. Price injection rule

Сейчас при catalog md_first может подмешиваться price_line в LLM-промпт.

После арбитра зафиксировать правило:

- service overview/catalog route → price_line подмешивается как сейчас, если политика это разрешает;
- FAQ/retrieval route → price_line не подмешивается, если вопрос не price-related.

Пример:
“приживаемость имплантов” → FAQ без цены.
“расскажите про классическую имплантацию” → service overview, цена может быть уместна.
“сколько стоит имплантация” → price route.

---

# P0. Arbiter telemetry

Логирование competing candidates — часть P0, не отдельная фича.

В debug_meta / bot_event писать:
- selected candidate;
- selected_route;
- reason;
- confidence;
- catalog_candidate;
- retrieval_candidate;
- alias_candidate;
- rejected_candidates;
- price_line_applied true/false.

Зачем:
чтобы на реальных вопросах было видно, почему арбитр выбрал именно этот chunk.

---

# P0. Типы чанков и broad service overview (контракт)

Тип чанка определяется сначала по документу, а не по anchor-метке:

- `__faq__*` или `doc_type=faq` → faq_specific (включая `#korotko`)
- `__info__*` или `doc_type=info` → info_specific (включая `#korotko`)
- `__pricing__*` или `doc_type=pricing` → pricing_specific
- `doctors__doctor__*` или `doc_type=doctor` → doctor_specific
- `__service__*` или `doc_type=service`:
  - `#korotko` → service_overview (главный краткий обзор услуги);
  - любой другой h3 → service_section (конкретный аспект услуги).

Примеры:
- `implantation__faq__osseointegration.md#korotko` = точный FAQ про приживаемость;
- `implantation__service__classic.md#korotko` = обзор услуги.

Broad service overview — это частный случай service_overview:
- catalog_candidate с mode=`md_first`;
- md_entry_ref указывает на service-док;
- итоговый chunk = `__service__...#korotko`.

Правило арбитра:
- broad service overview подходит для “расскажите про классическую имплантацию”,
  “что такое all-on-4”;
- но не должен перебивать:
  - faq_specific;
  - info_specific;
  - более точный service_section внутри той же услуги.

---

# P1. Query rewrite через текущую тему

_is_short_contextual сейчас достаточно узкий.
Не расширять его без необходимости.

Проблема:
query rewrite сейчас опирается на историю последних реплик, но не получает явно текущую тему.

Что нужно:
добавить в rewrite-контекст человекочитаемую текущую тему:
- h2 heading текущего doc/chunk;
- title/frontmatter, если есть;
- last_catalog_service title, если есть.

Не передавать в LLM только технический doc_id вида:
implantation__faq__osseointegration

Лучше:
“Текущая обсуждаемая тема: Приживаемость имплантов”

Зачем:
короткие follow-up вопросы будут лучше переформулироваться:
- “а долго?”
- “а риски?”
- “а если не приживется?”
- “а кости?”
- “а сколько?”

Цель:
улучшить follow-up без новых if-веток.

---

# P1. Guided UX для широких вопросов

Не делать сложный multi-intent.

Если пользователь пишет слишком широко:
- “импланты”
- “расскажите”
- “что лучше?”
- “что делать?”
- “хочу зубы”

бот должен дать короткий полезный overview и предложить направления.

Кнопки:
- Стоимость
- Больно ли
- Сроки
- Подходит ли мне
- Записаться

Комментарий:
если это раздувает P0, можно вынести в следующий PR после content arbiter.

---

# P1. Минимальный safety / complaint без усложнения

Не строим отдельную safety-архитектуру.

Нормальные страхи пациента:
- “боюсь боли”
- “опасно?”
- “а если не приживется?”
- “дорого”
- “у меня сложный случай”

должны идти в обычный sales/RAG-flow.

Только явно нецелевые/срочные/конфликтные сценарии отправлять в универсальный handoff:
- сильное кровотечение;
- высокая температура;
- гной;
- претензия/жалоба директору;
- юридический конфликт;
- спам/маты/троллинг.

Один универсальный handoff-текст.
В логах можно писать reason, но пользователю не нужен отдельный сложный сценарий.

Сейчас отдельные classify_safety/classify_complaint подключать не надо,
если нет реальных логов, что это частая проблема.

---

# P2. Унифицировать text matching

Проблема:
сейчас query_selector и retriever имеют разные:
- normalization;
- stop-words;
- token overlap;
- lemma logic;
- alias scoring.

Это может создавать конфликт:
catalog matching и corpus alias matching оценивают один и тот же вопрос по-разному.

Что нужно:
- двигаться к общему модулю text_match.py;
- вынести туда базовую нормализацию;
- общий набор stop-words;
- token/core extraction;
- lemma/token overlap.

Важно:
не обязательно делать полный refactor в P0.
Но при реализации arbiter не добавлять третий новый нормализатор.

---

# P2. Hardening ref-routing

Если клиент прислал ref в /ask:

1. Валидировать доступность chunk для client_id.
Это уже частично есть через get_chunk_by_ref, но стоит явно проверить.

2. Логировать:
- ref_requested;
- ref_resolved;
- ref_rejected.

3. Убедиться, что ref не обходит handoff gate.

4. Желательно:
ref должен быть из набора, который бот сам отдал в quick_replies/followups этой сессии.

То есть whitelist по last_presented_buttons.

Комментарий:
это не P0 accuracy-задача, но хороший hardening.

---

# P2. Убрать legacy/compat после стабилизации

После routing-fix и короткой проверки:
- убрать inspect.signature backwards compat;
- убрать _apply_response_policy_compat дубли;
- убрать _get_last_content_ui_payload_compat;
- убрать _mark_suggest_ref_used_compat;
- сократить legacy log_json-события или отделить их от product analytics;
- оставить bot_event как канон для dashboard/analytics.

Комментарий:
не смешивать с P0, чтобы не получить слишком большой diff.

---

# P3. Dashboard и PostgreSQL source of truth

Пока не делаем в текущем этапе.

В будущем:
- PostgreSQL должен стать источником для dashboard;
- JSONL остается fallback/audit;
- нужен backfill/import из JSONL в PG;
- Event Explorer оставить как developer mode;
- основной dashboard должен показывать качество, а не JSON.

Сигналы качества:
- low_score/no_candidates;
- проблемные route;
- повторные вопросы;
- частые темы;
- вопросы, приведшие к лидам;
- вопросы, где нужна доработка md/alias.

---

# P3. Полный eval-набор

Минимальный smoke eval — P0.
Полный eval — позже.

Набор:
80-150 бытовых фраз реальных пациентов.

Группы:
- цена;
- боль;
- сроки;
- приживаемость;
- противопоказания;
- нет зуба;
- врачи;
- контакты;
- запись;
- широкие вопросы “импланты”;
- короткие уточнения “а если кости мало?”, “а долго?”.

Для каждого кейса:
- expected_route;
- expected_doc_id или expected_topic;
- forbidden_doc_id, если есть;
- критерий правильного ответа.

---

# P3. Позже объединить handoff_filter и classify_intent

Перспективная оптимизация:
заменить два LLM-вызова одним route classifier:

{lead, content, price_lookup, price_concern, contacts, offtopic, handoff}

Но сейчас не делать.

Причина:
это меняет поведение первого слоя routing.
Нужно делать только после eval-набора.

---

# Правило для всех изменений

Любое усложнение должно проходить фильтр:

1. Помогает ли это типичному пациенту?
2. Повышает ли точность или конверсию?
3. Не делает ли перенос на другую клинику сложнее?
4. Можно ли решить это контентом/alias/eval вместо новой ветки кода?

Приоритет:
сначала точность типичных вопросов и путь к заявке,
потом аналитика,
и только потом редкие edge cases.
