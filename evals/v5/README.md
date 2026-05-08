# v5 per-layer evals

Этот каталог содержит per-layer golden sets и минимальный runner для v5 (см. `docs/ARCHITECTURE V5.md` §E3).

## Файлы

- `resolver_golden.json` — кейсы для Resolver (`DecisionFrame`)
- `arbiter_golden.json` — кейсы для Arbiter (`ArbiterDecision`)
- `verifier_golden.json` — кейсы для Verifier (`VerifierVerdict`)
- `generator_golden.json` — кейсы для Generator (faithfulness)
- `run_layer_eval.py` — запуск eval по слоям

## Запуск

```bash
python evals/v5/run_layer_eval.py --layer resolver
python evals/v5/run_layer_eval.py --layer all
```

Пока runtime-слои ещё не реализованы, runner будет помечать результаты как `SKIP` (это ожидаемо на Phase 0).

## E2E smoke (PR #E.0)

Цель: минимальный end-to-end smoke runner, который дёргает `/ask` целиком и проверяет:
- `meta.route` (если указан `expected_route`)
- `must_contain` — подстроки, которые **обязаны** быть в `answer` (case-insensitive)
- `must_not_contain` — подстроки, которых **не должно** быть в `answer` (case-insensitive)

### Файлы

- `e2e_smoke.json` — набор кейсов + baseline (root `baseline`)
- `run_e2e_smoke.py` — runner

### Формат кейса (кратко)

- `question`: текущий turn
- `history`: опционально, массив предыдущих turns вида `[{\"question\": \"...\"}]` (runner проигрывает их в `/ask` с той же `sid`)
- `expected_route`: опционально, строка
- `expected_route_any`: опционально, массив строк (ambiguous кейсы)
- `must_contain` / `must_not_contain`: опционально, массив строк (case-insensitive substring)

### Как запустить

1) Подними бота локально (по умолчанию ожидается `http://localhost:5000/ask`).

2) Запусти:

```bash
python evals/v5/run_e2e_smoke.py
```

Параметры через env:
- `BOT_URL` — URL endpoint `/ask` (default: `http://localhost:5000/ask`)
- `BOT_TIMEOUT_SEC` — timeout на запрос (default: 20)
- `CLIENT_ID` — client_id для запроса (default: `default`)
- `E2E_SMOKE_PATH` — путь к json (default: `evals/v5/e2e_smoke.json`)

### Как обновлять baseline

- Первый запуск: если `baseline=null`, runner выдаёт таблицу PASS/FAIL. После этого baseline фиксируется вручную: поставить `baseline=<passed>` в `e2e_smoke.json`.
- После улучшений: baseline обновлять **только** когда изменения осознанные и ожидаемые (не “подгонка” под текущий вывод).

### Что считать регрессией

- Любое падение ниже `baseline - 2` по количеству PASS (при фиксированном baseline).
- Сдвиг route в критичных кейсах (contacts/lead/price/handoff) даже если ответ «примерно похож».

### known_v4_failures

В корне `e2e_smoke.json` есть массив `known_v4_failures`: каждая запись связывает `case_id` с кратким описанием причины (типичный v4-баг) и полем `expected_fix_in_pr` (например `#1.3`, `#1.7`).

Это **намеренные регрессионные цели**, а не «сломанный» набор целиком: соответствующие кейсы в `cases[]` могут сейчас быть FAIL, но после merge указанного PR ожидается, что кейс начнёт **стабильно проходить**. Baseline по количеству PASS по-прежнему задаётся вручную; список `known_v4_failures` нужен для отслеживания прогресса и приоритетов без потери истории «почему этот сценарий важен».

