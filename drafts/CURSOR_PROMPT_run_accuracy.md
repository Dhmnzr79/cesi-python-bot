# Prompt для Cursor: сделать launch-eval runner

**Статус:** реализовано в `tools/run_accuracy.py`. Кейсы: `evals/accuracy_launch.json` (копия из `drafts/accuracy_launch.json` при синхронизации). В `finalize_ask` в `meta` клиенту добавляется `route` (как в логах `turn_complete`).

Ты работаешь над стоматологическим AI-ботом ЦЭСИ. Пользователь — маркетолог, не программист. Нужен простой пред-продовый runner точности, без сложного eval-фреймворка.

## Контекст

Есть файл:

- `evals/accuracy_launch.json`

Он содержит P0 launch-eval кейсы для проверки точности перед demo-prod запуском. Внутри есть:

- одиночные кейсы с `q`
- многоходовые кейсы с `turns`
- `expected.route` или `expected.route_any`
- `expected.doc_id` или `expected.doc_id_any`
- `expected.h3_id` или `expected.h3_id_any`
- `expected.matched_service_id`
- `expected.price_key`
- `expected.lead_step`
- `must_contain`
- `must_not_contain`
- `forbidden_doc_ids`
- `expected.fallback_reason_any` (список допустимых `fallback_reason`, напр. clarify по цене)
- в `ux`: `expect_lead_flow`, `expect_cta` (опционально)

API бота:

- `POST /ask`
- body: `{ "q": "...", "sid": "...", "client_id": "default" }`
- для follow-up кейсов нужно сохранять один и тот же `sid` между ходами.

## Задача

Сделай файл:

`tools/run_accuracy.py`

Он должен:

1. Читать `evals/accuracy_launch.json`.
2. Дёргать локальный бот по адресу из аргумента `--base-url`, по умолчанию `http://127.0.0.1:9000`.
3. Для каждого кейса создавать новый `sid`.
4. Для многоходовых кейсов `turns` использовать один и тот же `sid`.
5. Сравнивать фактический ответ с ожиданиями.
6. Печатать компактную таблицу по группам:

```text
GROUP                 PASS FAIL TOTAL  %
contacts              3    0    3      100
booking               3    0    3      100
...
TOTAL                 42   8    50     84
```

7. Печатать список fail-кейсов:

```text
FAIL price_tomography_01
Q: Сколько стоит КТ?
Expected: route=price_lookup matched_service_id=tomography price_key=tomography
Got: route=retrieval_chunk doc_id=implantation__pricing__implants h3_id=korotko
Reason: expected_route_mismatch, expected_service_mismatch
Answer: ...
```

8. Сохранять полный snapshot в:

`evals/snapshots/YYYYMMDD_HHMMSS_accuracy_launch.json`

В snapshot сохранить:

- metadata запуска
- каждый кейс
- каждый turn
- request body
- response payload
- extracted actual fields
- pass/fail
- fail reasons

## Как извлекать фактические поля

Из response JSON:

- `answer` = `payload["answer"]`
- `meta` = `payload.get("meta", {})`
- `route` = сначала `meta.get("route")`, если нет — попробовать `meta.get("selected_route")`, если нет — оставить null. Если сервер не возвращает route в payload, добавь в actual поле `route: null`, но не падай.
- `doc_id` = `meta.get("doc_id")`; если нет — взять `meta.get("file")` без `.md`; если нет — null.
- `h3_id` = `meta.get("h3_id")`
- `intent` = `meta.get("intent")`
- `matched_service_id` = `meta.get("matched_service_id")`
- `price_key` = `meta.get("price_key")`
- `lead_step` = `meta.get("lead_step")`
- `fallback_reason` = `meta.get("fallback_reason")`
- `cta_present` = `payload.get("cta") is not None`
- `quick_replies_count` = `len(payload.get("quick_replies") or [])`
- `lead_flow` = `bool(meta.get("lead_flow"))`

Важно: для кейсов с `route_any` при `route=null` runner даёт предупреждение `route_not_exposed`, но не валит кейс. Для строгого `expected.route` пустое значение — fail. Сейчас `meta.route` проставляется в `finalize_ask`.

## Проверки

Сделай функции:

- `matches_expected(actual, expected) -> list[str]` возвращает список причин fail.
- `check_text(answer, must_contain, must_not_contain) -> list[str]`
- `normalize_text(s)` lower, replace `ё` -> `е`, схлопнуть пробелы.

Правила:

- Если есть `expected.route`, actual.route должен совпасть.
- Если есть `expected.route_any`, actual.route должен быть одним из списка. Если actual.route is null — не fail, а warning `route_not_exposed`.
- Если есть `expected.doc_id`, actual.doc_id должен совпасть.
- Если есть `expected.doc_id_any`, actual.doc_id должен быть одним из списка.
- Если есть `expected.h3_id`, actual.h3_id должен совпасть.
- Если есть `expected.h3_id_any`, actual.h3_id должен быть одним из списка.
- Если есть `expected.matched_service_id`, actual.matched_service_id должен совпасть.
- Если есть `expected.price_key`, actual.price_key должен совпасть.
- Если есть `expected.lead_step`, actual.lead_step должен совпасть.
- Если есть `expected.fallback_reason_any`, actual.fallback_reason должен быть одним из списка.
- Если есть `forbidden_doc_ids`, actual.doc_id не должен быть в этом списке.
- Все `must_contain` должны встречаться в answer.
- Ни один `must_not_contain` не должен встречаться в answer.

## CLI

Поддержи аргументы:

```bash
python tools/run_accuracy.py --file evals/accuracy_launch.json --base-url http://127.0.0.1:9000 --client-id default
```

Дополнительно:

- `--group price_lookup` — прогнать только одну группу.
- `--id faq_pain_01` — прогнать один кейс.
- `--fail-only` — в консоли печатать только fails и summary.
- `--timeout 30`.

## Ограничения

- Не переписывай архитектуру бота.
- Не добавляй внешние зависимости кроме `requests`, если он уже доступен. Если requests нет — используй urllib.
- Не делай pytest. Это отдельный CLI-скрипт.
- Не усложняй.
- Код должен быть понятным и с комментариями.

## Важное замечание

`doc_id` в `meta` для ответов из чанка дублируется из имени файла (`build_ask_response`), чтобы eval и виджет видели стабильный идентификатор темы.
