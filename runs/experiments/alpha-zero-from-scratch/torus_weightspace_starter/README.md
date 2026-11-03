# Torus9 Weight-Space Programming: подготовленная инфраструктура

Начните с [ASTRA_HANDOFF.md](ASTRA_HANDOFF.md). Полная методика —
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), результаты инженерной проверки —
[PREPARATION_REPORT.md](PREPARATION_REPORT.md).

Сохранён существующий GraphNet: 8 residual-блоков × 120 каналов, 5CH,
396 009 параметров FP32. Конструктор теперь действительно обнуляет все параметры,
включая LayerNorm. Топологические buffers остаются каноническими индексами соседей.
Дополнительные ownership/score heads сохранены; для игры используются policy/WDL.

`design_weights.py` — пустой шаблон будущей программы Astra. Он не реализует
стратегию. Старые `artifacts/demo.pt` и `artifacts/zeros.pt` — нетронутые локальные
артефакты starter v0; новый строгий интерфейс работает с версионированным форматом v1.
`write_demo_weights` сохранён только как существующая тестовая фикстура.

Новые инструменты работают исключительно внутри эксперимента. Правила,
наблюдения, WDL utility и PUCT импортируются из проекта. Production loaders,
Arena, Orchestrator, training, checkpoint и уведомления не изменяются.

## Компоненты

- `model.py`: существующая архитектура, нулевая инициализация.
- `weights.py`: схема всех тензоров, NumPy I/O, строгая валидация, SHA-256, checkpoint I/O.
- `experiment_cli.py`: zero-pass, бюджеты и явные диагностические команды.
- `diagnostics.py`: positions, простые контрольные соперники и отдельный Arena adapter.
- `lock_reference.py`, `reference-lock.json`: проверенный локальный эталон и evidence.
- `positions.json`, `openings.json`: короткие легальные истории, получаемые через движок.
- `tests/`: изолированные синтетические тесты без production games.

## Научные источники

- [Оригинальная статья](https://rikiyat.github.io/blog/weight-space-programming/).
- [Experiment 6: исходные материалы](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch).
- [Методология и результаты](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/README.md).
- [Первоначальный prompt budget 0](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/runs/budget_0/message.txt).
- [Программная запись весов budget 0](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/runs/budget_0/tool_calls.jsonl).
- [Материалы budget 50](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch/runs/budget_50).

В Experiment 6 применяется policy-only MLP для обычного 9×9 Go; бюджет диагностики
задаётся числом positions/games запросов. Здесь сохранено это разделение, но модель,
топология и правила — из GoCube. Policy-only контрольные игры и сравнения с MCTS
имеют разные отчёты; их результаты нельзя объединять. Эвристики обычных краёв и
углов не переносятся на тор.
