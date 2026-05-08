# v5 — План реализации

**Парный документ к** `docs/ARCHITECTURE V5.md`. Архитектура отвечает на вопрос «что делаем», этот документ — «в каком порядке и какими PR».

**Девиз плана:** *Каждый PR — независимый минимально-стабильный шаг. Никакого «v5 одним коммитом».*

---

## 0. Принципы порядка работы

1. **Контракты before код.** Pydantic-схемы в `contracts/` пишутся раньше любого слоя, который их использует.
2. **Eval before код.** Golden set для слоя пишется ДО имплементации этого слоя. Без eval — нет права мерджить.
3. **Один PR = один блок дорожной карты.** Никаких «Resolver + Arbiter + topic-scope в одном PR».
4. **Shadow → Safety-net → On.** Новый LLM-слой проходит 3 стадии: пишется в логи (shadow) → подключается с откатом на старую логику при низкой confidence (safety-net) → старая логика помечается DEPRECATED.
5. **DEPRECATED → удалено за 1-2 PR.** Если строка живёт в `DEPRECATED.md` дольше двух PR — удаляется насильно. Без deadline старый код живёт вечно и порождает наслоение.
6. **Ничего не мерджится без зелёного eval.** PR закрывает acceptance criteria из `ARCHITECTURE V5.md §8` или не закрывает — третьего не дано.
7. **Параллельные треки B/C/D/E** — мерджатся независимо от Core (A), но имеют свою последовательность внутри.

---

## 1. Карта фаз

```
Phase 0 — Foundation Artifacts (no runtime changes)
   ├── PR #0.1  Архитектурные документы + .cursorrules
   ├── PR #0.2  Контракты (contracts/)
   ├── PR #0.3  core/routing.yaml + loader
   ├── PR #0.4  evals/v5/ скелет + runner
   └── PR #0.5  pg_sink trace schema (без записи)

Phase 1 — Core Pipeline (runtime, по слоям)
   ├── PR #1.1  Resolver (shadow)
   ├── PR #1.2  Resolver (safety-net on)
   ├── PR #1.3  Topic-scoped retrieval
   ├── PR #1.4  Source routing A3 (catalog hard route + 3 branches)
   ├── PR #1.5  Doctors lookup (A3.3)
   ├── PR #1.6  Arbiter (shadow)
   ├── PR #1.7  Arbiter (on) + удаление 7 if-rules
   ├── PR #1.8  Generator single-source
   ├── PR #1.9  Verifier с детерминированным триггером
   └── PR #1.10 12-band alias scorer → embedding similarity

Phase 2 — Cleanup
   ├── PR #2.1  Удаление DEPRECATED Phase 1
   └── PR #2.2  Чистка config.py (числа в routing.yaml)

Track B — Boosters (параллельно с Phase 1, но после A2/A8)
   ├── PR #B.1  Booster registry schema + loader
   ├── PR #B.2  Booster engine
   ├── PR #B.3  Migration: video_key → media boosters
   ├── PR #B.4  Migration: cta_* → cta boosters
   └── PR #B.5  Migration: suggest_refs → followup boosters

Track C — Content Tooling (параллельно, build-time)
   ├── PR #C.1  Corpus linter (sync)
   ├── PR #C.2  Auto-frontmatter draft CLI
   └── PR #C.3  Approve UI в admin_dashboard

Track D — Multi-client (параллельно, после Phase 1)
   ├── PR #D.1  client.yaml + tone.yaml + persona в Generator
   ├── PR #D.2  policy.yaml + вынос cta_from_turn / max_slots
   └── PR #D.3  Per-client eval directory + CI

Track E — Observability (параллельно, по мере появления слоёв)
   ├── PR #E.1  Trace logging on
   ├── PR #E.2  Per-layer eval в CI
   ├── PR #E.3  Turn replay view
   ├── PR #E.4  Content-gap dashboard
   └── PR #E.5  Hallucination dashboard
```

---

## 2. Phase 0 — Foundation Artifacts

**Цель фазы:** заложить артефакты, без которых имплементация будет наслаиваться.

**Принцип:** ни один из этих PR не меняет runtime-поведение бота. Бот в проде работает по v4 без изменений.

---

### PR #0.1 — Архитектурные документы + `.cursorrules`

**Файлы:**
- `docs/ARCHITECTURE V5.md` (уже есть)
- `docs/IMPLEMENTATION_PLAN.md` (этот документ)
- `.cursorrules` в корне репо
- `DEPRECATED.md` в корне репо (пустой шаблон)

**`.cursorrules` минимум:**
```
ARCHITECTURE: всегда читай docs/ARCHITECTURE V5.md перед изменениями.

INVARIANTS:
- Никаких `if client_id == ...` в коде. Различия — в clients/{id}/.
- LLM используется только на 4 уровнях: Resolver, Arbiter, Generator, Verifier.
- Никаких числовых scoring-весов. Все пороги — в core/routing.yaml.
- Каждый LLM-вызов имеет structured output schema (Pydantic).
- Generator получает массив длиной 1.
- Generator не «улучшает» факты. Если в источнике нет — в ответе нет.

WHEN UNCLEAR: спрашивай. Не добавляй новых полей в контракты без обновления
docs/ARCHITECTURE V5.md.
```

**Acceptance:** документы в репо, ссылаются друг на друга.

**Зависит от:** ничего.

---

### PR #0.2 — Контракты в `contracts/`

**Файлы (новые):**
- `contracts/__init__.py`
- `contracts/decision_frame.py` — Pydantic для DecisionFrame
- `contracts/gate_trace.py`
- `contracts/source_route_result.py`
- `contracts/retrieval_candidate.py`
- `contracts/arbiter_decision.py`
- `contracts/verifier_verdict.py`
- `contracts/session_state.py`
- `contracts/booster.py`

**Источник схем:** `ARCHITECTURE V5.md §1` — копируется 1-в-1.

**Acceptance:** `from contracts import *` работает. Каждая модель имеет docstring со ссылкой на раздел архитектуры.

**Зависит от:** PR #0.1.

---

### PR #0.3 — `core/routing.yaml` + loader

**Файлы (новые):**
- `core/routing.yaml` — все пороги из `ARCHITECTURE V5.md §D2`
- `core/routing_loader.py` — singleton-loader с кешем

**Acceptance:** `from core.routing_loader import THRESHOLDS; THRESHOLDS.resolver.min_confidence.intent` работает. Loader падает при отсутствии файла или невалидной схеме.

**Зависит от:** PR #0.2.

---

### PR #0.4 — `evals/v5/` скелет + runner

**Файлы (новые):**
- `evals/v5/resolver_golden.json` — 5-10 кейсов для старта (расширим в Phase 1)
- `evals/v5/arbiter_golden.json` — 3-5 кейсов
- `evals/v5/verifier_golden.json` — 3-5 кейсов
- `evals/v5/generator_golden.json` — 3-5 кейсов
- `evals/v5/run_layer_eval.py` — runner на ~200 строк
- `evals/v5/README.md` — формат кейсов и интерпретация

**Acceptance:** `python evals/v5/run_layer_eval.py --layer resolver` работает на пустых заглушках (всё fail, но runner запускается).

**Зависит от:** PR #0.2.

---

### PR #0.5 — `pg_sink` trace schema (без записи)

**Файлы:**
- `pg_sink.py` — добавить новые таблицы / колонки для trace-уровня (см. `ARCHITECTURE V5.md §E1`).
- Миграция БД.

**Acceptance:** таблицы созданы, но пока ничего не пишется (записи добавятся в каждом PR Phase 1).

**Зависит от:** PR #0.2.

---

## 3. Phase 1 — Core Pipeline

**Цель фазы:** реализовать новые слои A2-A7 с заменой старых компонентов.

**Принцип:** каждый LLM-слой проходит **shadow → safety-net → on**.

---

### PR #1.1 — Resolver (shadow)

**Цель:** LLM-Resolver работает в логи, не влияет на pipeline. Сравниваем с реальным `classify_intent` для калибровки.

**Файлы:**
- `resolver.py` (новый) — один LLM-вызов, structured output `DecisionFrame`.
- `app.py` — добавить вызов `resolver.resolve(...)` ПОСЛЕ `classify_intent`, результат пишется в `pg_sink.decision_frame`.
- `evals/v5/resolver_golden.json` — расширить до 50 кейсов.

**НЕ ТРОГАТЬ:** `classify_intent`, `query_selector`, `content_arbiter`. Pipeline идёт по v4.

**Acceptance:**
- `evals/v5/run_layer_eval.py --layer resolver` ≥ 90% по каждому полю DecisionFrame.
- В trace-логе виден реальный DecisionFrame.

**Зависит от:** PR #0.2, #0.4, #0.5.

**DEPRECATED:** ничего пока.

---

### PR #1.2 — Resolver safety-net on

**Цель:** Resolver влияет на маршрут, но при низкой confidence откатывается на старую логику.

**Файлы:**
- `app.py` — переключение на использование `DecisionFrame.route_intent` вместо результата `classify_intent`.
- Safety-net правило: если `confidence.intent < THRESHOLDS.resolver.min_confidence.intent` → fallback на старый `classify_intent` для этого turn'а.

**Контракт (без изменений):**
- **Safety-net по intent/topic:** решение берётся из Resolver только при достаточной уверенности (пороги — из `core/routing.yaml` через `THRESHOLDS`). Иначе — fallback на v4 логику для этого turn'а.
- **Rollback (обязательно):** env-флаг `RESOLVER_OFF=1` → бот работает строго по v4 (как до PR #1.1/#1.2).
- **Mapping v4→v5:** старый результат `classify_intent` маппится на новый `DecisionFrame.route_intent` (как описано в контракте safety-net).
- **DEPRECATED:** добавить в `DEPRECATED.md` (Active table) запись `llm.py:classify_intent` (помечается, не удаляется до PR #2.1).
- **Trace:** в turn-trace логируются маркеры `resolver_used` и `safety_net_used` (какой путь реально сработал).

**Acceptance:**
- Eval `accuracy_full.json` (текущий) не падает >2%.
- `RESOLVER_OFF=1` → бот работает по v4.
- Trace показывает `resolver_used | safety_net_used`.

**Что мониторить после deploy (shadow→on):**
- `% turns с safety_net_intent` < 15% (15–25% — ок, но watch; >25% → порог слишком жёсткий, смягчить в `routing.yaml`).
- `% turns с safety_net_topic` < 20% (>30% → пересмотреть минимум `confidence.topic`).
- `% turns с query_mode default to specific` < 30% (safe default, не критично).

**Зависит от:** PR #1.1.

**DEPRECATED:** `llm.py:classify_intent` — помечается, но НЕ удаляется (нужен для safety-net).

---

### PR #1.3 — Topic-scoped retrieval

**Цель:** retriever принимает `scope_topic` и фильтрует кандидатов.

**Файлы:**
- `retriever.py` — добавить параметр `scope_topic` в `retrieve()`.
- `app.py` — передавать `DecisionFrame.service_topic` если `confidence.topic ≥ threshold`.

**НЕ ТРОГАТЬ:** 12-band alias scorer (это PR #1.10), LLM rerank.

**Acceptance:**
- Cross-topic eval: 100% in-scope при уверенном топике.
- Общий eval не падает.

**Зависит от:** PR #1.2.

**DEPRECATED:** ничего.

---

### PR #1.4 — Source routing A3 (catalog hard route)

**Цель:** реализовать 3 ветки catalog match из `ARCHITECTURE V5.md §A3.1`. Закрывает класс багов «КТ».

**Файлы:**
- `source_routing.py` (новый) — оркестратор A3.
- `query_selector.py:match_service_from_catalog` — упростить: containment + lemma-subset, wrapper stripping.
- `app.py` — вставить A3 после Resolver и до A4.

**Логика веток:**
```
match (containment ≥ THRESHOLDS.catalog_match.containment_min) И facts/price_key
    → SourceRouteResult{source: catalog_facts | price_card}, минуем A4/A5
match И только md_entry_ref
    → SourceRouteResult{source: catalog_md, ref: md_entry_ref}, A4 с приоритетом
no match
    → SourceRouteResult{source: none}, обычный A4
```

**Acceptance:**
- Eval do_you_do (≥10 кейсов): все идут в catalog_facts, не в retrieval.
- Кейс «Вы делаете КТ зубов?» → catalog_facts.
- Wrapper stripping eval (5 кейсов): «вы делаете», «можно у вас», «есть ли» — все попадают в catalog.

**Зависит от:** PR #1.2.

**DEPRECATED:** часть `query_selector.py:select_catalog_content_route` — заменена `source_routing.py`.

---

### PR #1.5 — Doctors lookup (A3.3)

**Цель:** отдельный детерм. lookup для запросов про врачей.

**Файлы:**
- `doctors_lookup.py` (новый).
- `source_routing.py` — добавить ветку `topic=doctors`.

**Acceptance:**
- Запросы «кто врач по имплантации», «расскажите про доктора Боярышину», «врачи клиники» — обрабатываются без RAG.

**Зависит от:** PR #1.4.

---

### PR #1.6 — Arbiter (shadow)

**Цель:** LLM-Arbiter работает в логи, не влияет на выбор источника.

**Файлы:**
- `arbiter.py` (новый) — один LLM-вызов, structured output `ArbiterDecision`.
- `app.py` — после A4 вызывается arbiter если `|candidates| ≥ 2`, результат пишется в `pg_sink.arbiter_decision`. Реальный выбор по-прежнему делает старый `content_arbiter`.
- `evals/v5/arbiter_golden.json` — расширить до 30 кейсов.

**Acceptance:**
- Source pick accuracy ≥ 85% на golden set.
- В trace-логе видны и старый pick, и Arbiter pick.

**Зависит от:** PR #1.3.

---

### PR #1.7 — Arbiter on + удаление 7 if-rules

**Цель:** Arbiter принимает решение. Старые 7 if-rules удаляются.

**Файлы:**
- `app.py` — переключить на `arbiter.decide(...)` при `|candidates| ≥ 2`. Один кандидат → shortcut. Ноль → guided.
- `content_arbiter.py:select_content_route` — **удалить** (не пометить deprecated, а именно удалить, потому что Arbiter уже валидирован в #1.6).
- Fallback: arbiter timeout / `confidence < threshold` → guided.

**Acceptance:**
- Общий eval не падает >2%.
- Ambiguous-eval (диабет+имплантация и т.д.) ≥ 85% accuracy.

**Зависит от:** PR #1.6.

**DEPRECATED → REMOVED:** `content_arbiter.py:select_content_route` (7 if-rules).

---

### PR #1.8 — Generator single-source

**Цель:** Generator получает ровно один chunk, не агрегирует.

**Файлы:**
- `chunk_responder.py`, `llm.py:build_messages_for_gpt` — массив длиной 1.
- Цены/числа из `prices.json` подставляются через шаблон, не через LLM.

**Acceptance:**
- Faithfulness eval ≥ 95%.
- В логах виден `generator_input.source_ref` — всегда одна ссылка.

**Зависит от:** PR #1.7.

---

### PR #1.9 — Verifier с детерминированным триггером

**Цель:** проверка ответа на high-risk факты.

**Файлы:**
- `verifier.py` (новый) — LLM-вызов + детерминированный триггер по тексту ответа.
- `app.py` — вставить verifier после A6.
- `evals/v5/verifier_golden.json` — 20 кейсов с известными hallucinations.

**Триггер:**
```python
contains_number(answer) or contains_modal(answer)
or contains_time_promise(answer) or contains_warranty_claim(answer)
```

**Acceptance:**
- Hallucination rate < 1% на high-risk eval.
- Verifier пропускается если триггер не сработал (видно в логах).

**Зависит от:** PR #1.8.

---

### PR #1.10 — 12-band alias scorer → embedding similarity

**Цель:** убрать самый страшный кусок калибровки.

**Файлы:**
- `retriever.py:_alias_hit_score_raw_for_chunk` — **удалить**.
- Заменить на: exact alias hit (boolean signal) + cosine similarity по эмбеддингам алиасов.
- Алиас-индекс пересобирается build-time (один раз при индексации корпуса).

**Acceptance:**
- Общий eval не падает >2%.
- Алиас-кейсы из `accuracy_full.json` — все зелёные.

**Зависит от:** PR #1.3 (topic-scope включён, чтобы новый scorer не путался по корпусу).

**DEPRECATED → REMOVED:** `_alias_hit_score_raw_for_chunk`, `_lemma_alias_channel`, `_trigram_alias_channel`.

---

## 4. Phase 2 — Cleanup

**Цель фазы:** убрать накопленный DEPRECATED, чтобы не было наслоения.

---

### PR #2.1 — Удаление DEPRECATED Phase 1

**Файлы:**
- `llm.py:classify_intent` — **удалить**. Safety-net в `app.py` снимается.
- `content_arbiter.py` — удалить весь файл, если там не осталось живого кода.
- Все упоминания удаляемых функций в импортах.

**Acceptance:**
- `grep -r "classify_intent\|select_content_route\|_alias_hit_score_raw" .` — 0 результатов.
- Eval зелёный.

**Зависит от:** все PR Phase 1 в проде ≥ 1 неделя без регрессий.

---

### PR #2.2 — Чистка `config.py`

**Файлы:**
- `config.py` — удалить константы, переехавшие в `core/routing.yaml`:
  - `ALIAS_STRONG_THRESHOLD`, `ALIAS_SOFT_THRESHOLD`
  - `LOW_SCORE_THRESHOLD`
  - `PRICE_SERVICE_MATCH_STRONG`
  - `BROAD_QUERY_MAX_WORDS`
- Все импорты этих констант — заменить на `THRESHOLDS.*`.

**Acceptance:**
- В `config.py` остаются только инфра-константы (модели, таймауты, embedding name).
- Никаких числовых порогов точности в `*.py`.

**Зависит от:** PR #2.1.

---

## 5. Track B — Boosters

**Цель трека:** заменить ручные `suggest_refs` в md на декларативный движок.

**Стартует:** после PR #1.2 (есть DecisionFrame). Идёт параллельно с остальной Phase 1.

---

### PR #B.1 — Booster registry schema + loader

**Файлы:**
- `clients/{id}/boosters.yaml` (новый, на каждого клиента).
- `booster_loader.py` (новый).
- `contracts/booster.py` — уже создан в #0.2.

**Минимальный пример boosters.yaml:** см. `ARCHITECTURE V5.md §B1`.

**Acceptance:** `BoosterLoader.load(client_id)` возвращает список валидных Booster объектов.

**Зависит от:** PR #0.2.

---

### PR #B.2 — Booster engine

**Файлы:**
- `booster_engine.py` (новый).
- `policy.py` — заменить frontmatter-driven сборку UI на вызов `BoosterEngine.pick(...)`.
- В `clients/{id}/boosters.yaml` стартовый набор бустеров на основе текущих `suggest_refs` / `cta_text` / `video_key`.

**Acceptance:**
- UI-elements eval (≥15 кейсов): для каждого turn'а UI собирается через engine.
- Mutex-правила работают: один CTA, одна situation, одно видео за turn.

**Зависит от:** PR #B.1, PR #1.2.

---

### PR #B.3 — Migration: video_key → media boosters

**Файлы:**
- `clients/{id}/boosters.yaml` — добавить media-бустеры с `booster_tags` для каждого активного `video_key`.
- md-файлы — заменить `video_key: xxx` на `booster_tags: [...]` (где tag матчится с media-бустером).
- `policy.py` — убрать прямое чтение `video_key` из meta.

**Acceptance:**
- Видео показывается там же, где раньше (regression eval).
- `grep -r "video_key" md/` — 0 результатов.

**Зависит от:** PR #B.2.

---

### PR #B.4 — Migration: cta_* → cta boosters

**Файлы:**
- `clients/{id}/boosters.yaml` — глобальные CTA-бустеры с условиями активации (topic / query_mode / min_topic_turn).
- md-файлы — удалить `cta_text`, `cta_action`, `cta_from_turn`.
- `clients/{id}/policy.yaml` — `cta_from_turn` defaults.

**Acceptance:** CTA показывается так же, как раньше (regression).

**Зависит от:** PR #B.3, PR #D.2.

---

### PR #B.5 — Migration: suggest_refs → followup boosters

**Файлы:**
- `clients/{id}/boosters.yaml` — followup-бустеры на основе текущих `suggest_refs`.
- md-файлы — удалить `suggest_refs`, оставить `suggest_h3` (локальная навигация).

**Acceptance:**
- `grep -r "suggest_refs" md/` — 0 результатов.
- Cross-doc навигация работает через engine.

**Зависит от:** PR #B.4.

---

## 6. Track C — Content Tooling

**Цель трека:** снизить ручной труд при добавлении md.

**Стартует:** независимо от Phase 1, build-time.

---

### PR #C.1 — Corpus linter

**Файлы:**
- `tools/lint_corpus.py` (новый).
- `.github/workflows/corpus_lint.yml` — CI на PR в `clients/*/md/` или `boosters.yaml`.

**Проверки:** см. `ARCHITECTURE V5.md §C3`.

**Acceptance:** linter падает на синтетических тест-корпусах с известными ошибками.

**Зависит от:** PR #0.2.

---

### PR #C.2 — Auto-frontmatter draft CLI

**Файлы:**
- `tools/draft_frontmatter.py` (новый).
- `tools/README.md` — как пользоваться.

**Acceptance:** `python tools/draft_frontmatter.py --md path/to/new.md` выводит draft frontmatter в stdout.

**Зависит от:** PR #C.1.

---

### PR #C.3 — Approve UI

**Файлы:**
- `admin_dashboard/views/new_md.py` (или аналог в твоём фреймворке).
- Шаблон страницы.

**Acceptance:** куратор может: загрузить md → увидеть draft frontmatter → одобрить/отредактировать → сохранить файл с frontmatter в репо.

**Зависит от:** PR #C.2.

---

## 7. Track D — Multi-client

**Цель трека:** один core-код на всех клиентов, различия в данных.

**Стартует:** после PR #1.8 (Generator стабилен).

---

### PR #D.1 — `client.yaml` + `tone.yaml` + persona в Generator

**Файлы:**
- `clients/{id}/client.yaml`, `clients/{id}/tone.yaml` — для default клиента.
- `clients/loader.py` (новый) — singleton.
- `llm.py:BASE_SYSTEM` — переписан как функция `build_base_system(client_id)`, читающая из `tone.yaml`.

**Acceptance:**
- Бот для default клиента работает идентично текущему.
- Создание нового клиента сводится к копированию папки `clients/default/` и правке `client.yaml` + `tone.yaml`.

**Зависит от:** PR #1.8.

---

### PR #D.2 — `policy.yaml` + вынос cta_from_turn / max_slots

**Файлы:**
- `clients/{id}/policy.yaml`.
- `policy.py` — читает дефолты из `policy.yaml`, frontmatter-override остаётся.

**Acceptance:** все числа из `config.py`, относящиеся к UX/policy (max_slots, cta_from_turn default, situation_allowed_topics), переехали в `policy.yaml`.

**Зависит от:** PR #D.1.

---

### PR #D.3 — Per-client eval directory + CI

**Файлы:**
- `clients/{id}/eval/{cases}.json` — ≥30 кейсов на клиента.
- `.github/workflows/per_client_eval.yml` — прогон по всем клиентам перед merge.

**Acceptance:** CI падает, если accuracy на любом клиенте упала >2%.

**Зависит от:** PR #D.2, PR #0.4.

---

## 8. Track E — Observability

**Цель трека:** v5 не должен быть чёрным ящиком.

**Стартует:** PR #E.1 идёт сразу после Phase 0. Остальные — параллельно по мере появления слоёв.

---

### PR #E.1 — Trace logging on

**Файлы:**
- `pg_sink.py` — включить запись trace-уровня (схема уже создана в #0.5).
- Интеграционные точки в `app.py` для каждого слоя.

**Acceptance:** в БД виден полный trace одного turn'а: gates → resolver → routing → retrieval → arbiter → generator → verifier → policy.

**Зависит от:** PR #0.5.

---

### PR #E.2 — Per-layer eval в CI

**Файлы:**
- `.github/workflows/v5_eval.yml` — прогон `evals/v5/run_layer_eval.py --layer all` на каждый PR.
- Регрессионные пороги по слоям.

**Acceptance:** PR не мерджится, если eval любого слоя упал ниже acceptance criteria из `ARCHITECTURE V5.md §8`.

**Зависит от:** PR #0.4, PR #1.7 (нужны все runtime-слои).

---

### PR #E.3 — Turn replay view

**Файлы:**
- `admin_dashboard/views/turn_replay.py`.

**Acceptance:** по `turn_id` можно увидеть полный trace + повторить любой шаг (resolver / arbiter / verifier) с тем же входом.

**Зависит от:** PR #E.1.

---

### PR #E.4 — Content-gap dashboard

**Файлы:**
- `admin_dashboard/views/content_gaps.py`.

**Acceptance:** показывает топ-N запросов с `route=guided` или `confidence < threshold`, сгруппированных по топику.

**Зависит от:** PR #E.1.

---

### PR #E.5 — Hallucination dashboard

**Файлы:**
- `admin_dashboard/views/hallucinations.py`.

**Acceptance:** показывает % `grounded=false` по дням, топ-фразы, регрессионный график.

**Зависит от:** PR #1.9, PR #E.1.

---

## 9. Граф зависимостей PR (упрощённый)

```
#0.1 → #0.2 → #0.3
              ↓
            #0.4 → #0.5

Phase 1 chain:
#0.5 → #1.1 → #1.2 → #1.3 → #1.4 → #1.5
                              ↓
                            #1.6 → #1.7 → #1.8 → #1.9
                                                  ↓
                                                #1.10

Phase 2:
all Phase 1 → #2.1 → #2.2

Track B:
#1.2 → #B.1 → #B.2 → #B.3 → #B.4 → #B.5
                              ↑
                          #D.2 (для cta defaults)

Track C:
#0.2 → #C.1 → #C.2 → #C.3

Track D:
#1.8 → #D.1 → #D.2 → #D.3

Track E:
#0.5 → #E.1 → #E.3, #E.4
        ↓
       #1.9 → #E.5
       #1.7 → #E.2
```

---

## 10. Шаблон Cursor-сессии для одного PR

Чтобы избежать наслоения — каждый PR делается в одной сессии Cursor с явным контекстом.

```
@.cursorrules
@docs/ARCHITECTURE V5.md
@docs/IMPLEMENTATION_PLAN.md
@contracts/{relevant}.py
@core/routing.yaml

Задача: имплементировать PR #X.Y согласно плану.

Цель: <одна строка из IMPLEMENTATION_PLAN>
Файлы: <точный список>
Контракт входа/выхода: <ссылка на §1 ARCHITECTURE>
Acceptance: <criteria из §8 ARCHITECTURE>

НЕ ТРОГАТЬ:
- <явный список файлов>
- никакие DEPRECATED функции (см. DEPRECATED.md)

DEPRECATED после этого PR:
- <что помечается>
```

После имплементации — отдельная сессия на ревью:

```
@.cursorrules
@docs/ARCHITECTURE V5.md

Проверь diff PR #X.Y на соответствие:
1. Контрактам в §1.
2. Acceptance criteria в §8.
3. Принципам §0.

Найди все нарушения. Не правь, только список с file:line.
```

---

## 11. Чек-лист готовности к запуску имплементации

- [ ] PR #0.1 — `.cursorrules` и оба архитектурных документа в `main`.
- [ ] PR #0.2 — все контракты в `contracts/`.
- [ ] PR #0.3 — `core/routing.yaml` + loader в проде.
- [ ] PR #0.4 — runner и golden sets (хотя бы скелет).
- [ ] PR #0.5 — trace schema в БД.
- [ ] `DEPRECATED.md` создан и пуст.
- [ ] CI настроен на запуск `corpus_lint` и (минимально) `v5_eval`.

После этого — Phase 1 идёт линейно по PR #1.1 → #1.10. Параллельные треки B/C/D/E запускаются по мере появления зависимостей.

**Один PR — один merge — один откат при необходимости. Никакого «v5 в main одной кнопкой».**
