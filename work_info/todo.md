# TODO — v1.0 demo-prod

## 🔴 Срочно — прямо сейчас

- [x] **1. max_tokens → max_completion_tokens** (`llm.py`)
- [x] **2. Facts через LLM** (`ux_builder.py` → `build_service_facts_card_payload()`)
- [x] **3. Контекстный ценовой вопрос** (`query_selector.py`) — `_service_from_session_context()`
- [x] **4. md_entry_ref: null для КТ и отбеливания** — проверено, уже было сделано

## 🟡 Важно — до прода

- [x] **5. concern_ref для имплантации** — добавлено в каталог + роутинг в `app.py`
- [x] **6. Короткие реплики** (`app.py`) — `_is_short_contextual()` + редирект в current_doc
- [x] **7. Aliases** (`service_catalog.json`) — переработаны: простые формулировки, убраны ценовые и вопросительные фразы, проверены пересечения
- [ ] **8. prices.json заполнить** — ждём цены от клиента

- [ ] **Лемматизация в ценовом матче** (`query_selector.py` + `retriever.py`)
  - `_match_score` без лемматизации — частично закрыто новыми aliases (короткие формы).
  - Полное решение: подключить `_lemma_alias_channel` из `retriever.py`.

## 🟢 После запуска

- [ ] **9. concern_ref как поле в каталоге** — уже сделано для имплантации, расширить на все услуги
- [ ] **10. JSON Schema валидация** (`service_catalog.json`, `prices.json`)
- [ ] **11. Aliases расширить из логов** — после накопления реальных запросов
- [ ] **12. Проверить все md_entry_ref в каталоге** — все 11 услуг
- [ ] Teaser над виджетом (widget.md §19)
- [ ] Видеокаталог на фронте (маппинг key → URL/title)
- [ ] Starter prompts — заполнить для клиента
