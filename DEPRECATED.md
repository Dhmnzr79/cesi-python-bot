# DEPRECATED Registry

**Назначение.** Реестр кода, помеченного на удаление. Задача — не дать старому коду жить вечно «на всякий случай» рядом с новым. Это главная защита от наслоения подходов.

**Парный документ:** `docs/IMPLEMENTATION_PLAN.md`.

---

## Правила работы с этим файлом

1. Когда новый PR заменяет старую функцию/модуль/константу — старая помечается здесь со ссылкой на заменяющий PR.
2. **Запись живёт максимум 2 PR** в этом реестре. После этого старый код удаляется насильно.
3. В коде помеченная функция получает комментарий-маркер:
   ```python
   # DEPRECATED — replaced by <new>, see DEPRECATED.md, removed in PR #X.Y
   ```
4. Никакие новые вызовы DEPRECATED-функций не добавляются. CI должен ловить это (см. PR #C.1 corpus_lint, расширение).
5. После удаления — запись переезжает в раздел «Removed» с датой PR.

---

## Active (помечено, но ещё в коде)

| Что | File:line | Заменено на | Помечено в PR | Удалить в PR |
|---|---|---|---|---|
| `llm.py:classify_intent` | `llm.py:971` | `resolver.py:resolve_with_fallback()` + запись trace в `pg_sink` из `app.py` | PR #1.2 | PR #2.1 |
| `query_selector.py:select_catalog_content_route` | `query_selector.py:select_catalog_content_route` | `source_routing.route_source` + `content_arbiter.collect_content_candidates` hint | PR #1.3 | PR #2.1 |
| `llm.py:classify_handoff_filter` | `llm.py:948` | `ingress_gate.classify_ingress()` | ingress PR | PR #2.1 |
| `ux_builder.py:offtopic_response` + `app.py` intent==offtopic | — | `ingress_gate` hard_stop / ingress LLM | ingress PR | PR #2.1 |

---

## Removed (история)

| Что | Удалено в PR | Дата merge |
|---|---|---|
| _(пусто)_ | | |

---

## Закрытый технический долг (не registry-row)

| Проблема | Закрыто в PR | Примечание |
|---|---|---|
| Расхождение промпта Resolver между eval (`resolve_decision_frame_shadow`) и runtime (`resolve_decision_frame`) | PR #1.2.6 | Единый `RESOLVER_SYSTEM_PROMPT` + общий `_call_resolver_llm`; golden eval вызывает `resolve_decision_frame`. |
| Дублирование pipeline между `/ask` и `/ask/stream` | PR #1.2.5 | Общий `_orchestrate_ask_turn`; эндпоинты — только JSON vs SSE dispatch. |
| Topic-leak в multi-turn retrieval | reverted in PR #1.3-revert | Было в раннем PR под номером #1.3: жёсткий `scope_topic` в `retrieve()`. Отключено до A3. В runtime scope остаётся в shadow/disable до **PR #1.4** (topic-scope reactivation **после** **PR #1.3** Source routing A3); в trace — `retrieval_scope_topic`. |
| Session bridge для `service_id` из истории | reverted PR #1.3.1-revert | Отдельная склейка `service_id` по session отменена; fallback остаётся через `session.last_service_id` в ценовом пути до A3 в полном объёме. |
| Хардкод-шаблон цены в `ux_builder.py` | PR #1.2.7 | Заменён на default `price_ref` → `clinic__info__payment_terms.md#korotko` + price-aware инструкция в `llm_question` (`app.py`). |
| Multi-turn service_id resolution через session | PR #1.3 | Закреплено в `SourceRouteResult.match_method="session_fallback"` в `source_routing.py` (детерминированно вместо неявного поведения `query_selector`). |
| Шаблонная склейка doctor cards (обрезка по символам, без LLM) | PR #1.3.6 | Заменено на LLM-компоновку из structured facts `doctors_lookup`. |

---

## Запланированное к DEPRECATED (по дорожной карте)

Из `docs/ARCHITECTURE V5.md §9`. Эти строки появятся в Active после соответствующих PR.

| Что | Будет помечено в | Будет удалено в |
|---|---|---|
| `content_arbiter.py:select_content_route` (7 if-rules) | — (удаляется сразу) | PR #1.7 |
| `retriever.py` 12-band alias scorer (`_alias_hit_score_raw_for_chunk`, `_lemma_alias_channel`, `_trigram_alias_channel`) | `alias_scorer_legacy_shadow.py` (telemetry only) | после стабилизации shadow |
| `query_selector.py:_match_score`, `_match_score_lemma` (magic-band) | PR #1.3 | PR #2.1 |
| `config.py:ALIAS_STRONG_THRESHOLD`, `ALIAS_SOFT_THRESHOLD` | PR #1.10 | PR #2.2 |
| `config.py:LOW_SCORE_THRESHOLD` | PR #1.4 | PR #2.2 |
| `config.py:PRICE_SERVICE_MATCH_STRONG` | PR #1.3 | PR #2.2 |
| `config.py:BROAD_QUERY_MAX_WORDS` | PR #1.3 | PR #2.2 |
| `llm.py:BASE_SYSTEM` (хардкод клиники) | PR #D.1 | PR #2.1 (или ранее) |
| `cta_text`, `cta_action`, `cta_from_turn` в md frontmatter | PR #B.4 | PR #B.4 |
| `video_key` в md frontmatter | PR #B.3 | PR #B.3 |
| `suggest_refs` в md frontmatter | PR #B.5 | PR #B.5 |

---

## CI guard (план)

В рамках PR #C.1 (corpus_lint) добавить проверку:
- `grep` по DEPRECATED-функциям из таблицы Active в новых diff'ах → fail PR.
- `grep` по DEPRECATED-полям frontmatter после соответствующих миграционных PR → fail.

Это автоматическая защита от случайного использования помеченного кода.
