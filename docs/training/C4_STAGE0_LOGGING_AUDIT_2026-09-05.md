# Cube 4×4×6 — Stage 0 logging audit

Дата проверки: 2026-09-05
Проверяемая ветка: `main`
Проверяемый commit: `7b206929cec357a6771a5c2410f9de170bacb399`

## Область и запуск

В терминологии текущего кода Cube 4×4×6 задаётся как topology `cube`, размер грани `4` (4×4 на каждой из 6 граней).

Текущий entrypoint: `alphazero/envs/gocube/train.py`. Отдельного config-файла нет: конфигурация задаётся CLI и затем сохраняется в checkpoint. Текущий код использует относительные каталоги `checkpoint/<run-name>`, `data/<run-name>` и `runs/<run-name>`.

Диагностический run: `stage0-cube4x4x6-20260905`.

Точная команда успешного диагностического запуска (из отдельного временного каталога):

```bash
PYTHONPATH=/home/codex/projects/gocube-alphazero \
/home/codex/projects/gocube-alphazero/.venv/bin/python \
/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/train.py \
  --topology cube --size 4 --workers 1 --sims 1 --arena-sims 1 \
  --games-per-iteration 5 --iterations 1 --train-batch-size 1 \
  --train-steps-per-iteration 1 --fast-game-prob 0.25 \
  --endgame-sample-weight 1 --no-arena \
  --run-name stage0-cube4x4x6-20260905
```

Команда была выполнена с рабочим каталогом `/tmp/gocube-stage0-2026-09-05-cube4` и stdout/stderr записаны в `/tmp/gocube-stage0-2026-09-05-cube4/console.log`. Запуск создал 5 полностью завершённых тестовых партий в iteration 1. Официальное обучение и baseline не запускались.

До этого был пробный запуск с ошибочным для данной задачи `--size 6` в `/tmp/gocube-stage0-2026-09-05`; его данные не использованы в выводах аудита. Были также две неуспешные попытки запуска системным `python` и без `PYTHONPATH`; они завершились до self-play.

## Созданные данные успешного run

Все пути ниже находятся вне репозитория и не добавлялись в Git:

```text
/tmp/gocube-stage0-2026-09-05-cube4/console.log
/tmp/gocube-stage0-2026-09-05-cube4/checkpoint/stage0-cube4x4x6-20260905/gocube-run.json
/tmp/gocube-stage0-2026-09-05-cube4/checkpoint/stage0-cube4x4x6-20260905/iteration-0000.pkl
/tmp/gocube-stage0-2026-09-05-cube4/checkpoint/stage0-cube4x4x6-20260905/iteration-0001.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-data.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-policy.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-value.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-score.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-ownership.pkl
/tmp/gocube-stage0-2026-09-05-cube4/data/stage0-cube4x4x6-20260905/iteration-0001-ownership-mask.pkl
/tmp/gocube-stage0-2026-09-05-cube4/runs/stage0-cube4x4x6-20260905/events.out.tfevents.1788629457.Legion
```

Наличие run manifest подтверждено. В checkpoint сохранены фактические run-level параметры: cube/4, Japanese, komi 7.5, KataGo rules V3, fingerprint, `sims=1`, `fast-game-prob=0.25`, workers=1 и параметры сети/optimizer.

Фактически получено: 5 games, 5 scored, 0 no-result, средняя длина 148.8 ходов, 575 базовых и 575 сохранённых samples, 865 regular и 285 fast decisions (realized fast fraction 24.78%).

## Game-level logging

Проверка выполнена по созданным файлам и по исходному коду `alphazero/envs/gocube/train.py`, `alphazero/SelfPlayAgent.pyx` и `alphazero/envs/gocube/game.py`.

| Поле | Статус | Что реально сохраняется |
|---|---|---|
| Уникальный постоянный Game ID | FAIL | ID партии не создаётся и не сохраняется. |
| iteration | FAIL | Есть только iteration-level scalar/checkpoint; в записи партии поля нет. |
| Номер партии внутри iteration | FAIL | В output нет per-game номера. |
| checkpoint/model | FAIL | Модель указана только в run-level checkpoint args, не привязана к партии. |
| Время партии | FAIL | Есть aggregate `performance/sample_time`, per-game времени нет. |
| Количество ходов | FAIL | Сохраняется только aggregate average game length; per-game turns не сохраняются. |
| Результат | FAIL | Есть aggregate `win_rate/player0`, `player1`, `draws`; результата каждой партии нет. |
| Final score / margin | FAIL | В TensorBoard нет per-game score/margin и отдельного game record нет. |
| Terminal kind | FAIL | Есть aggregate `terminal/scored_games`/ `no_result_games`; per-game значение отсутствует. |
| Причина no-result | FAIL | Есть только aggregate `terminal/cycle_no_result`; per-game reason отсутствует. |
| Cleanup moves / endgame diagnostics | FAIL | Aggregate counters сохраняются (`cleanup1_moves=260`, `cleanup2_moves=57`, captures=193, ko unblock=1), per-game breakdown отсутствует. |
| Фактические параметры именно этой партии | FAIL | Сохраняется общий checkpoint args, но не per-game snapshot. |
| Путь к полному машинному record партии | FAIL | Training self-play records не создаются. |
| Hash record | FAIL | Поле/файл hash для training record отсутствует. |

## Ручная проверка records

FAIL: среди output успешного run нет ни одного сохранённого полного record, поэтому проверить 3 records вручную невозможно (проверено: 0/3). В частности, нельзя восстановить из сохранённых training artifacts всю последовательность ходов, PASS, финальную позицию и per-game технические метаданные. Файлы `*-data.pkl`, `*-policy.pkl`, `*-value.pkl`, `*-score.pkl` и ownership tensors являются агрегированными sample tensors, а не replay records партий.

## Iteration-level logging

| Показатель | Статус | Результат/примечание |
|---|---|---|
| games count | PASS | TensorBoard/console: 5 games. |
| Black wins | PASS | `win_rate/player0 = 0.4` (2 wins). |
| White wins | PASS | `win_rate/player1 = 0.6` (3 wins). |
| draws | PASS | 0. |
| no-result | PASS | 0; `terminal/no_result_games` присутствует. |
| average game length | PASS | 148.8. |
| regular decisions | PASS | 865. |
| fast decisions | PASS | 285. |
| фактическая доля fast search | PASS | 24.78%. |
| saved samples | PASS | 575. |
| training history size | PASS | `history_iterations=1`. |
| optimizer steps | PASS | planned=1, actual=1. |
| examples seen | PASS | 1. |
| effective sample passes | PASS | 0.001739. |
| effective learning rate | PASS | 0.01. |
| policy loss | PASS | 4.720041. |
| value loss | PASS | 1.544092. |
| ownership loss | PASS | 0.706123. |
| score loss | PASS | 0.322854. |
| Arena result | NOT AVAILABLE | Arena отключена диагностическим `--no-arena`; код умеет писать arena win-rate, но в этом run значение не возникает. |
| self-play time | PASS | Есть `performance/sample_time` (3.405 s/game aggregate). Полного отдельного wall-clock поля нет. |
| training time | NOT AVAILABLE | Отдельный training duration metric не пишется; короткий запуск даёт только progress output. |
| Arena time | NOT AVAILABLE | Arena не запускалась. |
| average inference batch | PASS | Scalar существует, значение 0.0 в warmup-run; обычная model inference в коротком запуске не дала ненулевого значения. |
| CPU | NOT AVAILABLE | Не сохраняется. |
| RAM | NOT AVAILABLE | Не сохраняется. |
| GPU | NOT AVAILABLE | Не сохраняется как metric (checkpoint args содержит `cuda=true`). |
| VRAM | NOT AVAILABLE | Не сохраняется. |

Дополнительные iteration terminal diagnostics также записались: scored games=5, cleanup stages entered=5/5, cleanup1 moves=260, cleanup2 moves=57, cleanup captures=193, ko unblock actions=1, cycle no-result=0, training-valid fraction=1.0.

## Главные вопросы

1. **NO.** Нельзя однозначно связать каждую сыгранную партию одновременно с параметрами и машинным record: per-game ID/record отсутствуют, а параметры есть только на уровне run/checkpoint.
2. **NO.** После run можно автоматически собрать aggregate iteration metrics, но нельзя собрать данные всех партий как records без ручного console output; самих per-game records нет.
3. **NO.** По сохранённым tensors, checkpoint и aggregate TensorBoard нельзя точно восстановить ход каждой партии и полный experiment trail.

## Найденные проблемы и что потребуется отдельно

Нужно отдельное следующее задание на versioned per-game event/record schema: устойчивый Game ID, iteration/game index, model/checkpoint ID, per-game параметры, timestamps, moves включая PASS и финальную позицию, result/score, terminal/no-result fields, cleanup diagnostics, record path и hash. Нужен также машинный run/iteration manifest с длительностями, hardware metrics и явной связью каждого record с run/checkpoint. В рамках Stage 0 код не изменялся.

## Итог

**STAGE 0: FAIL**

Причина: партии действительно завершились и aggregate training/terminal telemetry сохраняется, но автоматические полные per-game records отсутствуют; поэтому отсутствуют однозначная per-game связь с параметрами и возможность точного воспроизведения эксперимента.

