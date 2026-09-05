# Cube 4×4×6 — Stage 0 logging retest

Дата: 2026-09-05

Проверенный commit реализации: `cf3687ef59ef4cb4c4e7152f76331e00a8d8c459`
Ветка: `codex/stage0-logging-fix-2026-09-05`

## Что реализовано

- Добавлен versioned JSON record `schema_version: 1`, один файл на принятую
  self-play партию.
- Cube 4×4×6 получает ID `C4-000001`, `C4-000002`, … . ID резервируется
  через persistent inter-process file lock только для принятой завершённой
  партии; поэтому workers не конфликтуют и speculative batch slots не создают
  пропусков.
- Record содержит run/iteration/game number, self-play checkpoint, UTC start/end
  time и duration, topology/size/rules/komi, полный effective parameter
  snapshot, ordered moves с явным `PASS`, final position, result/score/margin,
  terminal/no-result metadata и cleanup/endgame diagnostics.
- После iteration создаётся `iteration-manifest.json` с Game ID, record paths,
  SHA-256, checkpoint, parameters и aggregate metrics.
- TensorBoard и existing training tensors сохранены; learning parameters,
  rules, scoring и training logic не менялись.
- Google Sheets не используется.

Для терминала без score поле `final_score`/`final_score_margin` записывается как
`null`, а причина указывается в `field_notes`; для scored партии
`no_result_reason` равен `null` с пометкой «not applicable». Это соответствует
реально отсутствующим значениям и не подменяет их выдуманными данными.

## Изменённые файлы

- `alphazero/SelfPlayAgent.pyx` — worker-side move capture, accepted-game ID и
  record payload; generic non-GoCube path остаётся совместимым.
- `alphazero/envs/gocube/records.py` — allocator, JSON schema/writer, SHA-256 и
  iteration manifest.
- `alphazero/envs/gocube/train.py` — effective iteration context, record writing
  и manifest creation.
- `tests/test_gocube_selfplay_records.py` — allocator concurrency, schema,
  moves/PASS, final position, terminal metadata, hashes и manifest checks.

Незакоммиченный пользовательский `overnight30.py` в commit не включён.

## Тесты

Команда:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

Результат: `246 passed, 1 warning`.

Предупреждение — существующий `torch.load(weights_only=False)` FutureWarning,
не связанный с logging.

## Diagnostic Stage 0 run

Официальный baseline C4-T001 не запускался. Запущен только отдельный
диагностический run на 5 завершённых партиях:

```bash
OUT=/tmp/gocube-stage0-logging-retest-2026-09-05-v4
mkdir -p "$OUT"
cd "$OUT"
PYTHONPATH=/home/codex/projects/gocube-alphazero \
/home/codex/projects/gocube-alphazero/.venv/bin/python \
/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/train.py \
  --topology cube --size 4 --workers 2 --sims 1 --arena-sims 1 \
  --games-per-iteration 5 --iterations 1 --train-batch-size 1 \
  --train-steps-per-iteration 1 --fast-game-prob 0.25 \
  --endgame-sample-weight 1 --no-arena \
  --run-name stage0-cube4x4x6-logging-retest-20260905-v4
```

Результат: 5 games, 5 scored, 0 no-result; checkpoint self-play —
`stage0-cube4x4x6-logging-retest-20260905-v4@0`.

Iteration manifest:

```text
/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/iteration-manifest.json
```

### Records

| Game ID | Record | SHA-256 |
|---|---|---|
| `C4-000001` | `/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/C4-000001.json` | `8b4b0a80b935ab2be2cfed76d9df72cbac7eb73052a2cc909e9bb0a0807d5ce6` |
| `C4-000002` | `/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/C4-000002.json` | `4f2bf1af3b4bd6a43bc1e7f0d5effb914201fb20dee296d9407e4fd93f5b79f3` |
| `C4-000003` | `/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/C4-000003.json` | `7f77ad77048f8b00a54ab26387ff2b35122712a6bbf9413e464127f64d3c556a` |
| `C4-000004` | `/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/C4-000004.json` | `3f5e2495cd694b3ca91eafbaf68d7c4247e1553243d8d56d5c840df20e0d2f2b` |
| `C4-000005` | `/tmp/gocube-stage0-logging-retest-2026-09-05-v4/data/stage0-cube4x4x6-logging-retest-20260905-v4/records/iteration-0001/C4-000005.json` | `c66983aed63d05157f130551d31e164346ed391468be7254fd1ee5277cad5bfd` |

Проверено автоматически для всех пяти records:

- Game ID уникальны и идут без пропусков `C4-000001`…`C4-000005`.
- Каждый record существует, его `record_path` совпадает с manifest, а SHA-256
  совпадает с содержимым файла.
- В каждом есть 90 effective parameters, iteration 1, checkpoint `@0`, moves,
  final position, terminal/result metadata и cleanup diagnostics.
- Manifest содержит все пять records и aggregate metrics: games=5,
  black_wins=4, white_wins=1, draws=0, average game length=140.4.

Вручную открыты и проверены следующие три records:

- `C4-000001`: 115 moves, 11 PASS, scored, `black_win`;
- `C4-000002`: 125 moves, 2 PASS, scored, `black_win`;
- `C4-000003`: 129 moves, 3 PASS, scored, `black_win`.

Для всех трёх подтверждены contiguous move numbers, final position на всех 96
точках Cube 4, соответствие `terminal.kind` и `terminal_kind`, а также наличие
cleanup/endgame diagnostics. Поэтому из каждого record автоматически доступны
последовательность ходов, PASS, финальная позиция, результат и terminal metadata.

## Итог

```text
STAGE 0: PASS
```

Пять диагностических партий теперь однозначно связываются через
`Game ID → run → iteration → checkpoint → effective parameters → record → SHA-256`.
