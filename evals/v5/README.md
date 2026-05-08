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

