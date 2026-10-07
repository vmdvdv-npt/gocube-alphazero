# Orchestrator V2: руководство по режимам и запуску

Это обязательная инструкция для агента и оператора. Перед запуском, возобновлением
или изменением параметров прочитайте **версию из того commit, который будет исполнять
job**, проверьте реальные артефакты и выполните `job --check`. Не переносите настройки
из примеров в production без проверки фактической lineage.

Файл хранится в Git вместе с реализацией. `AGENTS.md` требует его прочитать;
публичная точка запуска выводит путь и ссылку на GitHub. Команда `guide` показывает
текст. CI и `job` сверяют fingerprint проверенных интерфейсов с маркером в конце
файла: изменения поддерживаемых режимов требуют пересмотра инструкции. При
регистрации нового job сохраняется `operator-guide.json` с SHA256 инструкции.
Автоматическая проверка подтверждает согласованность версий; прочитать и правильно
применить инструкцию обязан запускающий агент.

## Единственная публичная граница запуска

Работайте из отдельного worktree, если требуется разрешённое изменение кода.
Не меняйте рабочее дерево, HEAD или immutable runtime действующего процесса.
Код и это руководство для настоящего запуска должны быть закоммичены;
локальная правка инструкции блокирует запуск до commit. Используйте Python окружения
проекта; ниже `.venv/bin/python` означает фактический доступный интерпретатор.

```bash
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint guide
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint job configs/operator/my-job.json --runs-root /absolute/path/runs --check
.venv/bin/python -m gocube_golden.orchestrator_v2.production_entrypoint job configs/operator/my-job.json --runs-root /absolute/path/runs
```

`--check` разрешает зарегистрированные checkpoint и replay, нормализует параметры
и показывает compiled plan. Не создаёт job, не обучает, не играет и не отправляет
уведомления. Проверяйте не только успешное завершение, но и полученные настройки.
Настоящий запуск регистрирует параметры, закрепляет commit, создаёт стандартный
контроллер с READY handshake и использует стандартные lease/supervision/outbox.
Нельзя заменять его самодельным launcher, вызовами training/selfplay engine или
ручным workflow. Lifecycle уведомления обязательны; отсутствие стандартного
Telegram config блокирует запуск. Не отправляйте тестовое сообщение без запроса.

Публичный JSON имеет `schema: "gocube-operator-job-v1"`. Сейчас поддерживается
**Torus9 5CH, komi 1.5**, совместимая обученная модель с Adam state, включая
завершённую адаптацию M137. Старые 6CH/adaptation checkpoints не являются
допустимым родителем ordinary training. Checkpoint задаётся селектором
`lineage/checkpoint_id`, а не путём, скриптом или командой. Resolver ищет единственный
зарегистрированный checkpoint в `torus9/active` или `torus9/archive` и проверяет
целостность/совместимость. Optimizer state продолжается, не сбрасывается.

## Какие публичные режимы поддерживаются

| Режим | Как выбрать | Self-play | Результат |
| --- | --- | --- | --- |
| Конечное ordinary training | `parent`, `training.iterations: N > 0` | Да | Новая отдельная lineage; периодические арены |
| Непрерывное training | `training.iterations: null` | Да | Durable run без заданного последнего поколения |
| Training, затем A/B | Конечное training и `ab_tests` | Да | Все тесты стартуют от общего результата основной части job |
| Только обычный A/B | `training.iterations: 0`, `ab_tests` с `A`/`B` | Да, в каждой ветке | Две независимые ветки и итоговая арена |
| Offline A/B | `training.iterations: 0`, `offline_replay`, `A`/`B` | **Нет** | Обучение на историческом replay, арена финалов |
| Offline A/B/C и больше | То же, `arms` с ≥2 именами | **Нет** | Все ветки и round-robin каждой пары |
| Только арены | `arenas`; training и ab_tests отсутствуют либо iterations=0 | **Нет** | Сравнение уже зарегистрированных checkpoint |
| Выбор победителя → training | `winner_selection`, без `parent`/A/B; iterations>0 либо null | После выбора — да | Арена выбирает родителя новой lineage |
| Явные дополнительные арены | `arenas` вместе с конечным training/A/B | По основному режиму | Дополнительные арены известных checkpoint после основных действий |

`iterations: null` нельзя сочетать с A/B: бесконечная часть не завершится.
Арены после неё также не выполнятся, пока основной шаг не завершён. Несколько
`ab_tests` независимы: каждый стартует от одного общего родителя, а не от победителя
предыдущего теста. Победитель A/B автоматически не продолжает production.

## Параметры и наследование

Верхний уровень: `schema`, `run_id`, `topology`, `parent`, `training`, `arena`,
`execution`, `ab_tests`, `arenas`, `self_play`, `winner_selection`. Неизвестные поля
отвергаются. `run_id`, id теста/арены и имена arms: буквы, цифры, `_`, `-`, `.`,
первый символ — буква или цифра. Один run_id нельзя перепрофилировать изменением JSON.
`topology` по умолчанию `torus9`; остальные значения этим публичным job не поддерживаются.

**Значения ниже — defaults интерфейса, а не настройки текущей production lineage.**
Обычные параметры обучения задайте явно после изучения родителя и истории.
Модель, оптимизатор, совместимость, training seed и остальные поддерживаемые
унаследованные настройки берутся из effective config родителя.

| `training` | Default | Ограничение / смысл |
| --- | --- | --- |
| `iterations` | 5 | Целое ≥0 либо null; 0 отключает основную часть training |
| `games_per_iteration` | 384 | Положительное целое; свежие self-play игры ordinary режима |
| `mcts_simulations` | 200 | Положительное целое; fixed self-play budget |
| `learning_rate` | 0.00005 | Конечное положительное число; не предполагает LR scaling при смене batch |
| `updates_per_iteration` | 160 | Положительное целое; число optimizer updates |
| `batch_size` | 64 | Любое положительное целое |
| `gradient_clip` | 1.0 | Конечное положительное число |
| `replay_generations` | 6 | Положительное целое; rolling replay ordinary режима |

Samples = batch_size × updates_per_iteration, с выборкой позиций с возвращением.
Изменение batch при фиксированных updates меняет объём обучения. Для честного
сравнения фиксируйте samples и явно меняйте updates.

`execution` сейчас фиксирован: `device: "cuda"`, `workers: 16`. Другие значения
отвергаются. Используются стандартные active games 4/worker и inference batch cap
64 / wait 1 ms; публичных переключателей этих лимитов нет.

`arena`: `every_iterations`=5, `games`=192, `mcts_simulations`=128,
необязательный boolean `tree_reuse`. Games — чётное число ≥64; budgets положительные.
Arena использует paired starts/color swap, noise off, temperature 0,
cpuct 1.25, FPU 0, watchdog 1000, komi 1.5; resignation/fast search отключены.
Это не поля свободной настройки operator JSON. `tree_reuse` включайте явно и
одинаково для сравниваемых сетей. Для A/B arena доступны только games,
mcts_simulations, tree_reuse; seed общий, публичного arm-specific arena seed нет.

`self_play` доступен ordinary training и обычным A/B веткам:

```json
{
  "search_mode": "pcr",
  "pcr": {"cheap_simulations": 100, "full_simulations": 400, "full_probability": 0.33},
  "master_seed": 2026100501,
  "tree_reuse": true
}
```

`search_mode` — `fixed` или `pcr`. Без PCR используется fixed budget из training;
объект `pcr` допустим только при явном `search_mode: "pcr"`. Cheap/full —
положительные целые, full>cheap; full_probability строго между 0 и 1.
Full-позиции имеют обычный root noise и learner targets; cheap-позиции не попадают
в learner replay и используют noise off. Raw артефакт сохраняет все ходы.
Master seed — целое ≥0, изменяет только self-play seed, не training seed.
Tree reuse выключен при отсутствии явного включения; у арены и self-play отдельные
переключатели. Offline режим не вызывает self-play независимо от этих полей.

## Примеры ordinary training и обычного A/B

Все селекторы в примерах — placeholders: замените их проверенными артефактами.
Каждый пример — отдельный JSON. Ни один пример сам по себе не разрешает запуск.

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "training-example",
  "parent": "source/M255",
  "training": {"iterations": 5, "learning_rate": 0.000025, "batch_size": 128,
    "updates_per_iteration": 1280, "gradient_clip": 8, "replay_generations": 5},
  "arena": {"every_iterations": 5, "games": 192, "mcts_simulations": 64}
}
```

Для непрерывного обучения замените iterations на null и не добавляйте ab_tests.
Чтобы **после** конечного обучения сравнить два LR, добавьте:

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "online-ab-example",
  "parent": "source/M255",
  "training": {"iterations": 0, "learning_rate": 0.000025, "batch_size": 128,
    "updates_per_iteration": 1280, "gradient_clip": 8, "replay_generations": 5},
  "ab_tests": [{"id": "learning-rate", "iterations": 5,
    "A": {"learning_rate": 0.000025}, "B": {"learning_rate": 0.000015},
    "arena": {"games": 192, "mcts_simulations": 64}}]
}
```

Здесь iterations=0 означает A/B сразу от source/M255. Для training → A/B поставьте
положительное число. A/B `iterations` обязательно положительное целое; overrides
A/B принимают поля training **кроме iterations**, наследуя общие параметры job.
Каждая обычная ветка генерирует свои игры; этот режим нельзя использовать при
запрете нового self-play. Именованные `arms` доступны только offline.

## Offline A/B/C: готовый replay без единой self-play игры

Offline — опция **только эксперимента**, не обычного обучения. Основная часть
обязана иметь `training.iterations: 0`. `offline_replay` содержит ровно один
зарегистрированный source checkpoint на iteration. Цепочка должна последовательно
продолжать parent; разные поколения могут принадлежать разным lineage.

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "offline-batch-example",
  "parent": "source-a/M255",
  "training": {"iterations": 0, "learning_rate": 0.000025,
    "gradient_clip": 8, "replay_generations": 5},
  "ab_tests": [{"id": "batch", "iterations": 5,
    "offline_replay": ["source-a/M256", "source-b/M257", "source-b/M258",
      "source-b/M259", "source-b/M260"],
    "arms": {
      "B64": {"batch_size": 64, "updates_per_iteration": 2560},
      "B128": {"batch_size": 128, "updates_per_iteration": 1280},
      "B256": {"batch_size": 256, "updates_per_iteration": 640}
    },
    "arena": {"games": 192, "mcts_simulations": 64, "tree_reuse": true}
  }]
}
```

Рабочий конкретный пример: `configs/operator/torus9-offline-batch-m255-g256-260-20261007-v1.json`.
Для двух веток можно заменить arms на A/B; нельзя одновременно задавать обе формы.
Все arms получают одну и ту же историческую replay schedule и iteration budget.
Source checkpoint — **описание входных данных итерации**, а не новые веса ветки.
Веса каждой ветки стартуют от одного parent с optimizer и затем продолжают именно
свой предыдущий candidate. Replay берётся из source fresh/rolling manifests:
точные исторические buckets, а не только свежие данные выбранного поколения.
Проверяются совместимость, SHA256 каждого shard, manifest и зарегистрированной
цепочки. Источники используются по ссылкам, не переписываются. Изменение
исторического manifest между итерациями останавливает эксперимент.

Например, историческое rolling окно 5 для поколения 256 — данные 252–256;
следующее — 253–257. Значение replay_generations в experiment JSON не заменяет
исторический manifest самодельным окном. Все дополнительные исторические shards
должны оставаться доступными.

Самплер training engine сохранён: uniform positions с возвращением,
`random.Random(training_seed + ordinary_update)` на каждом update. Общий seed
**не означает одинаковую последовательность sample IDs** при разных batch:
отличаются границы и число updates, меняется optimizer clock. В отчёте указывайте
это ограничение; опции точного общего списка sample IDs пока нет. Не внедряйте
её молча. LR, модель, loss weights, clip и optimizer одинаковы, если они явно
одинаково заданы/унаследованы; автоматического изменения LR из-за batch нет.

Каждая итерация сохраняет checkpoint с optimizer, обычные losses/telemetry,
optimizer-loop wall time (секунды и минуты), updates/sec, samples/sec,
CUDA peak allocated/reserved bytes. Измеренное training time включает progress
publishing и синхронизацию, исключает загрузку replay, validation и сохранение
checkpoint. Mean total loss — сумма policy + WDL/value + ownership + score;
последние два — отдельные aux losses. Для полного job elapsed используйте timestamps
и указывайте отличие от training time. Не подменяйте allocated bytes показанием
всей видеокарты из nvidia-smi.

После всех итераций запускается round-robin **финалов**: каждая неупорядоченная
пара один раз, общий budget/seed, paired starts/color swap, noise off. При N arms
число пар N×(N−1)/2. Игры арены являются оценкой, а не self-play и не добавляются
в replay. Итог `experiments/<run_id>-ab-<test_id>/report.json`: поколения, losses,
minutes, среднее/общее время, ускорение относительно наименьшего batch, все арены.
Ветки имеют отдельные lineage и не выбираются автоматически для production.

## Отдельные арены и выбор победителя

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "arena-only-example",
  "arenas": [{"id": "compare", "candidate": "branch-a/M260",
    "reference": "branch-b/M260", "games": 192, "mcts_simulations": 64,
    "master_seed": 2026092902, "tree_reuse": true}]
}
```

В arena-only parent не нужен. Если training/ab_tests отсутствуют, iterations
автоматически 0; если training указан — поставьте 0 явно. Несколько arenas задают
порядок независимых сравнений. Допустимы только уже зарегистрированные checkpoint,
разрешаемые до запуска; нельзя ссылаться этим списком на ещё не созданный candidate.
Для будущих финалов используйте встроенные training/A/B арены.

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "winner-training-example",
  "winner_selection": {"candidate": "branch-a/M260", "reference": "branch-b/M260",
    "games": 192, "mcts_simulations": 64, "tree_reuse": true},
  "training": {"iterations": 5, "learning_rate": 0.000025}
}
```

winner_selection заменяет parent, запрещает A/B и требует iterations>0/null.
Ничья или невалидная арена останавливают workflow без скрытого выбора.
`arenas` принимает id, candidate, reference, games, mcts_simulations, master_seed,
tree_reuse; winner_selection — те же поля без id. Default master_seed=2026092902.

## Возобновление, мониторинг и остановка

Для возобновления используйте ту же команду `job`, тот же run_id и неизменный JSON.
Durable state/lease защищают от дублирования; existing execution-commit сохраняется.
Это не команда принудительного restart: живой контроллер не заменяется.
Для новых параметров нужен новый run_id. Не меняйте зарегистрированные parameters,
resolved, checkpoint или replay вручную и не обходите immutable code pin.

Параметры находятся в `torus9/orchestration/jobs/<run_id>/`; workflow state и
controller.log — в `torus9/orchestration/workflows/<run_id>/`. Branch checkpoints,
training metrics, heartbeat и manifests — в соответствующей `active/<lineage>/`;
experiment state/report/arenas — в `experiments/<run_id>-ab-<test_id>/`.
Читайте эти файлы пассивно. Сравнивайте liveness, phase, done/total и durable state,
а не только отсутствие stdout. Для внешних GPU/process измерений используйте
[training efficiency monitor](../diagnostics/training-efficiency-monitor.md);
outputs пишите вне live run. Не профилируйте конкурентной GPU нагрузкой измеряемый job.

В публичном parser нет команд `stop`, `restart`, `migrate`, `set-code-pin` или
`resume --force`. Не придумывайте их. Остановку/миграцию выполняйте только по
явному запросу пользователя и существующей документированной процедуре конкретного
runner. Изменение кода действующего run требует отдельного разрешения и worktree.

Для обычного continuous runner SIGINT/SIGTERM контроллеру преобразуются в durable
soft stop: активное поколение или арена завершается, затем lineage становится
SOFT_STOPPED. При возобновлении тем же job запрос потребляется. Обработчики
восстанавливаются после выхода. Эта процедура относится к continuous runner;
не переносите её автоматически на offline experiment или отдельный arena worker.
Подробности: [continuous-training.md](continuous-training.md). Сигнал посылается
только после явного запроса пользователя на остановку и проверки нужного PID.

## Внутренние режимы V2 и служебные команды

RunMode перечисляет continuous, performance_tuning, arena, evaluation, experiment,
calibration, workflow, scenario. Workflow actions: continuous_training, experiment,
arena, calibration, adaptation_phase, select, stop. Эти runner/action interfaces
существуют в движке, но **не являются альтернативной публичной границей запуска**.

| Внутренний режим / интерфейс | Доступность оператору |
| --- | --- |
| continuous | Через training operator job |
| experiment | Через ab_tests; online A/B либо offline multi-arm |
| arena / evaluation | Через arenas, training/A/B arena, winner_selection в пределах их настроек |
| performance_tuning | Нет публичной секции operator job; не заменять A/B этим runner |
| calibration / komi-calibration | Нет публичной секции operator job |
| adaptation_phase | Нет публичной секции operator job для новой адаптации |
| workflow / scenario, select / stop | Компилируются/исполняются внутренне; произвольный DAG оператором не принимается |
| experiment stage2/control-C | Внутренняя возможность; не поддерживается простым ab_tests JSON |

Legacy CLI `run`, `continuous`, `performance-tuning`, `experiment`,
`komi-calibration`, прямой `workflow` блокируются с указанием использовать job.
`workflow --controller` — внутренний подписанный путь с startup handshake, не
обходной способ запуска. При необходимости неподдерживаемого параметра/режима
назовите точное ограничение и запросите разрешение на соответствующее изменение
кода. Не разрешайте это путём низкоуровневого плана или monkey patch.

Служебные публичные команды:

- `guide`: только чтение этой инструкции, без игр/обучения/уведомлений.
- `notifications-drain <saved-root> --timeout 7`: штатная повторная доставка
  сохранённого outbox; сохраняет политику и транспорт. Это не запуск обучения.
  `DELIVERY_UNCERTAIN` не повторяется даже этой командой: сетевой сбой/потеря
  ответа не доказывает, что сообщение не было принято. Только явный HTTP 429
  допускает автоматический retry; ошибки конфигурации requeue разрешает штатная
  команда. Проверяйте сохранённые events/delivery и сообщайте пользователю
  статус и last_error_code. Не редактируйте receipt/outbox и не отправляйте
  пропущенное сообщение вручную для обхода политики. Изменение политики
  повторов требует отдельного разрешения пользователя.
- `telegram-test`: отправляет одно настоящее тестовое сообщение; только если
  пользователь прямо запросил тест транспорта. Не использовать при --check.

## Как обновлять руководство

Изменяя поддерживаемые параметры, ограничения, defaults, маршрутизацию или режимы,
сначала приведите текст и примеры в соответствие конечному коду. После обзора
обновите маркер fingerprint (это подтверждение обзора, не автоматическая генерация
документации) и выполните contract/операторские тесты:

```bash
.venv/bin/python -c 'from gocube_golden.orchestrator_v2.operator_guide import interface_fingerprint; print(interface_fingerprint())'
.venv/bin/python -m pytest tests/test_orchestrator_operator_guide.py tests/test_orchestrator_operator_job.py tests/test_orchestrator_operator_arena_job.py tests/test_orchestrator_operator_winner_selection.py
```

Замените значение в маркере ниже напечатанным SHA256; закоммитьте код, инструкцию
и тесты одним reviewable изменением. Новые изменения интерфейса без обновления
маркера ломают contract test и блокируют новый запуск. Старые job продолжают
исполняться на своём прежнем commit; новую инструкцию читайте вместе с этим pin.

<!-- reviewed-interface-sha256: 39d2a14f9bd43503f4b9c738a4666a560f334d285ebc3375641c4e491e3a9f6f -->

## B64 bounded performance audit

The explicit diagnostic exception is `production_entrypoint b64-perf-audit CONFIG`.
It accepts `gocube-b64-perf-audit-v1` with `runs_root`, `resolved_experiment`
(the saved offline B64/B128/B256 experiment spec), an exclusive `output` outside
`runs_root`, `warmup` (32–64), and `updates` (128–256). See
`configs/diagnostics/b64-perf-audit-20261007.json`. Run through the existing V2
entrypoint; it owns the execution authority. It verifies and reads M255 and
M256–M260 source references in place, measures the first historical B64 generation
in memory, and creates no production checkpoint, lineage, self-play or Arena.
Diagnostics run with the production Adam, sampler, FP32 losses and checks.
The shared ordinary heartbeat uses durable writes and its 10-second pulse.
Place output on the same filesystem as production heartbeat files; `/tmp` may
be tmpfs and would invalidate the I/O comparison. The existing external efficiency
monitor observes isolated diagnostic heartbeat files, with results outside its
diagnostic run root. No lifecycle notifications are synthesized for this audit;
production notification settings and delivery remain unchanged.

For Nsight Systems on hosts where PyTorch CUPTI emits runtime calls without
device activity, set optional `nsys_trace_only: true` in a second audit config
with an exclusive output directory. It performs the same parity check and
32-update warm-up, then 16 measured updates with NVTX spans and CUDA profiler
capture-range markers. Use `nsys profile --trace=cuda,nvtx --sample=none
--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop` around
the V2 command. The entrypoint mints a signed training child permit; the measured
`require_engine_execution` uses the same child authorization branch as production.

Generate the paired Markdown/JSON analysis with:

```sh
.venv/bin/python -m gocube_golden.b64_perf_report /absolute/diagnostic/output \
  --nsys-root /absolute/nsys-audit/output \
  --nsys-sqlite /absolute/nsys-export.sqlite
```

Export SQLite using `nsys export --type=sqlite --output=FILE TRACE.nsys-rep`.
Omit the Nsight arguments for a preliminary report; unavailable CUDA kernel
activity is reported explicitly. The checked-in measured report is
`docs/diagnostics/b64-performance-audit-20261007/REPORT.md` with `report.json`.
