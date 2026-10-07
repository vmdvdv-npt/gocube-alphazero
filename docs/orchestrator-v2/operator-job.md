Полное руководство по режимам и запуску: [ORCHESTRATOR_V2_LAUNCH_GUIDE.md](ORCHESTRATOR_V2_LAUNCH_GUIDE.md). Перед запуском прочитайте его для используемого code pin.

# Запуск только из файла параметров

Операторский вход V2 — `production_entrypoint job job.json`. Он проверяет простой
JSON и передаёт исполнение существующим continuous/experiment runners V2.
Отдельного оркестратора, ручного управления итерациями или отправки Telegram нет.
Низкоуровневые команды сохранены для совместимости внутренних инструментов.

Готовый файл: `configs/operator/torus9-five-iterations.json`.

```sh
python -m gocube_golden.orchestrator_v2.production_entrypoint job job.json --check
python -m gocube_golden.orchestrator_v2.production_entrypoint job job.json
```

`--check` проверяет параметры и зарегистрированный родительский чекпойнт,
показывает полный план и наличие настройки Telegram. Не создаёт запуск,
не отправляет сообщения и не выполняет обучение. Для нестандартного расположения
артефактов обе команды принимают `--runs-root /absolute/path/to/runs`.

Единственный пользовательский вход для запуска обучения и Arena — `job` с этим
файлом параметров. Standalone `run`, `continuous`, `experiment`, `calibration`,
`performance-tuning` и обычный `workflow` отключены. Внутренний workflow controller
запускается только самим `job` с подписанным одноразовым V2 child-permit; ручной
`workflow --controller` также отклоняется. Поэтому lifecycle-уведомления принадлежат
одному операторскому запуску и проходят через стандартный durable outbox.

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "my-five-iterations",
  "parent": "source-lineage/checkpoint-id",
  "training": {
    "iterations": 5,
    "learning_rate": 0.00005,
    "games_per_iteration": 384,
    "mcts_simulations": 200,
    "updates_per_iteration": 160,
    "batch_size": 64,
    "gradient_clip": 1.0,
    "replay_generations": 6
  },
  "arena": {"every_iterations": 5, "games": 192, "mcts_simulations": 128},
  "ab_tests": []
}
```

Числа в примере — значения по умолчанию. `iterations` задаёт остановку после
числа итераций от родителя. Арена запускается на каждом кратном
`every_iterations` шаге и сравнивает с чекпойнтом на этот интервал раньше.
Если число итераций не кратно интервалу, дополнительной арены в конце нет.
Для пяти итераций и арены в конце укажите оба значения равными пяти.
Арена диагностическая: не выбирает автоматически новый режим обучения.

Для непрерывного продолжения до штатного operator stop задайте
`"iterations": null`. Job остаётся detached, а runner продолжает training и
Arena по указанному cadence; `0` по-прежнему означает отсутствие training.
Значение `null` нельзя использовать вместе с A/B тестами в том же job: такой
training не завершится, чтобы передать им управление.

`parent` — точное имя в графе артефактов, не путь к произвольному файлу.
Поддерживаются обычные Torus9 5CH чекпойнты и завершённая адаптация update-2400
с коми 1.5. Шестиканальные модели отклоняются. Коми, архитектура, состояние Adam,
валидация и исходный replay наследуются проверяемым способом; Adam не сбрасывается.
`training.batch_size` — положительное целое число позиций на optimizer step
(по умолчанию 64). Backend требует CUDA/16 workers (4 игры на worker).
Replay — последние N поколений без ограничения числа позиций.
`training.gradient_clip` — положительное конечное число; значение передаётся в
global gradient norm clipping learner-а. Если поле не указано, используется 1.0.
Неизвестные поля и неподдерживаемые значения — ошибка до старта, включая A/B.
Полей для скриптов, команд, отключения уведомлений или смены кода нет.

## Single-arm offline training

Для одной независимой offline lineage укажите top-level `offline_replay` и
конечный `training.iterations: N > 0`. Список должен содержать ровно N
зарегистрированных source checkpoint-ов, начиная с поколения после `parent`.
Pipeline продолжает parent optimizer state, использует historical replay по
ссылкам и не запускает Arena автоматически:

```json
{
  "schema": "gocube-operator-job-v1",
  "run_id": "offline-single-arm",
  "parent": "source/M255",
  "training": {
    "iterations": 5,
    "learning_rate": 0.000025,
    "batch_size": 64,
    "updates_per_iteration": 2560,
    "gradient_clip": 8,
    "replay_generations": 5,
    "replay_sampling": {"mode": "policy_surprise", "weight": 0.5}
  },
  "offline_replay": ["source/M256", "source/M257", "source/M258", "source/M259", "source/M260"]
}
```

Single-arm `offline_replay` нельзя сочетать с `ab_tests` или `winner_selection`.
После завершения используйте отдельный arena-only job с `arenas[]`; multi-arm
offline A/B/C workflow и его round-robin остаются без изменений.

## Переиспользование MCTS-дерева

По умолчанию `tree_reuse` выключен: каждый ход получает новое дерево, как раньше.
Для явного включения добавьте `"tree_reuse": true` в `self_play` и/или `arena`:

```json
"self_play": {"tree_reuse": true},
"arena": {"every_iterations": 5, "games": 192, "mcts_simulations": 128, "tree_reuse": true}
```

Поле принимает только JSON boolean. Его можно задавать также в отдельных
`arenas[]`, `winner_selection` и `ab_tests[].arena`; они наследуют значение
из `arena`, если собственного значения нет. Новый job не наследует включённый
reuse self-play от родителя: требуется явное включение в `self_play`.

После каждого фактически сыгранного действия сохраняется соответствующее
поддерево с visits/Q, priors и expanded nodes. Проверяется полный state key,
включая superko history; отсутствие child или несовпадение состояния приводит
к новому дереву. В self-play используется фактически sampled action. В Arena
каждая партия хранит отдельные деревья кандидата и reference; оба продвигаются
после каждого хода, включая ход противника. Между партиями деревья не передаются.
Dirichlet noise обновляется на каждом self-play root из исходного neural policy,
без повторного смешивания уже зашумлённых priors.

Simulation cap остаётся числом **новых** симуляций за ход. При ON root visits
и policy target учитывают также накопленные симуляции поддерева, поэтому сумма
root visits может превышать cap текущего хода (включая PCR). Режим входит в
search fingerprint, effective config, self-play contract identity и telemetry;
исторические OFF search fingerprints сохраняются.

Для внутренних V2 effective configs используются `self_play.tree_reuse` и
`arena.tree_reuse`; Arena profile ID при ON содержит `|tree_reuse=true`.
Это относится также к Cube V2, использующему общий Golden PUCT.
Для отдельного synchronous search используйте `SearchSettings(tree_reuse=True)`
и вызывайте `advance_root(action, resulting_state)` после каждого реального хода;
при новой партии вызывайте `reset_tree()` или создавайте новый searcher.
Cooperative sessions одной партии получают один `SearchTree`, отдельный для
каждой модели. Изменение режима оформляется новым job; работающий запуск не меняется.

## Playout Cap Randomization (PCR)

### Seed self-play

В объекте `self_play` можно задать `"master_seed": 2026100501` — целое
неотрицательное число (включая 0; boolean, дроби и `null` отклоняются).
Поле работает и с fixed, и с PCR, например:

```json
"self_play": {
  "master_seed": 2026100501,
  "search_mode": "pcr",
  "pcr": {
    "cheap_simulations": 100,
    "full_simulations": 500,
    "full_probability": 0.25
  }
}
```

Операторский seed записывается как `execution.selfplay_master_seed` в effective
config и передаётся существующему self-play adapter. Он определяет seed партий,
root noise и PCR full/cheap sampling. Если поле отсутствует, наследуется seed
self-play родителя. Seed learner-а и состояние Adam сохраняются; seed арены
этим полем не меняется. Поле применяется к основному обучению, A/B arms и обоим
вариантам `winner_selection`. Seed входит в fingerprint и shard identity;
изменение seed требует нового `run_id`, а не правки уже работающего запуска.

Готовый пример без запуска: `configs/operator/torus9-pcr-100-500.json`.
В простом операторском JSON добавьте отдельный объект:

```json
"self_play": {
  "search_mode": "pcr",
  "pcr": {
    "cheap_simulations": 100,
    "full_simulations": 500,
    "full_probability": 0.25
  }
}
```

Перед каждым root search Torus9 выбирает full/cheap по seed партии, ply и
отдельному versioned PCR namespace, независимо от порядка workers. Cheap:
100 sims и root noise OFF; full: 500 sims и обычный root noise ON. Номинальное
среднее — 200 sims/ход. Temperature schedule остаётся одинаковым. Caps должны
быть положительными целыми, full больше cheap, probability строго между 0 и 1.
Значения 100/500/0.25 — пример, а не ограничение.

При отсутствующем `self_play` или `"search_mode": "fixed"` используется
`training.mcts_simulations`. В PCR это поле не определяет caps; caps задаются
только объектом `pcr`. PCR применяется и к A/B arms данного job. Арена остаётся
с собственным фиксированным бюджетом.

Raw `.games.jsonl.gz` хранит все ходы с `search_mode`, `search_simulations` и
`training_eligible`. Learner `.pt` содержит только full-позиции: targets policy,
WDL, ownership и score для cheap не строятся. Полностью cheap партии остаются
только в raw storage. Terminal targets вычисляются по всей trajectory.
Исторический fixed replay читается как прежде и не переписывается.

PCR параметры входят в effective config, fingerprint, manifest и shard resume
identity. Изменение параметров требует нового `run_id`. Новый lineage ссылается
на родительский checkpoint по существующей identity/path/SHA без его копирования.
Raw и новый replay сохраняются в существующем lineage-owned storage. Сведения
о PCR показываются при старте и в block report; `.telemetry.json` каждого shard
и generation summary содержат full/cheap counts/fractions, caps, nominal mean,
`raw_positions` и `training_positions` (число новых learner позиций).
Доля full в маленьком shard может отличаться от заданной вероятности.

## A/B

### Автоматическое продолжение от победителя двух готовых чекпойнтов

Вместо `parent` задайте `winner_selection`. Например:

```json
"winner_selection": {
  "candidate": "source-lineage/M249",
  "reference": "source-lineage/M246",
  "games": 192,
  "mcts_simulations": 200
}
```

Сначала штатный workflow V2 проводит эту арену, сохраняет выбор победителя,
затем запускает обычный continuous runner в новой линии с именем `run_id`.
Остальные `training`, `self_play` и `arena` задают параметры продолжения.
Бюджет `winner_selection.mcts_simulations` относится только к первой арене:
например, `arena.mcts_simulations: 64` оставляет последующие арены по 64 sims.
При победе M246 новые поколения начинаются с M247; при победе M249 — с M250.
Старая линия и её чекпойнты сохраняются. Adam и исходный replay наследуются
от выбранного родителя; обе возможные конфигурации проверяются до старта арены.

Выбирается сторона с большим числом побед. Ничья, невалидный результат,
технические партии, неполный набор игр или несовпадение identity останавливают
workflow в `STOPPED`, без запуска обучения. Дальнейшее решение принимает
оператор новым job. Выбор и SHA чекпойнта сохраняются в состоянии шага `winner`;
возобновление прерванного обучения не повторяет завершённую арену и выбор.

`training.iterations` должен быть положительным числом или `null` для работы
до мягкой остановки. `parent` и `ab_tests` вместе с `winner_selection` запрещены.
Обычные A/B-тесты параметров ниже сохраняют прежнее поведение.
Готовый пример: `configs/operator/torus9-arena-winner-continuation.json`.

Добавьте, например:

```json
"ab_tests": [
  {
    "id": "learning-rate",
    "iterations": 5,
    "A": {"learning_rate": 0.00005},
    "B": {"learning_rate": 0.00003},
    "arena": {"games": 192, "mcts_simulations": 128}
  }
]
```

После основного блока V2 последовательно выполняет тесты. В каждом тесте обе
ветки начинают с одного итогового чекпойнта основного блока, обучаются указанное
число итераций, затем играют друг против друга. Общие параметры наследуются,
в A/B перечисляются только изменения. Арена A/B выполняется в конце теста;
`every_iterations` внутри A/B не допускается. Победитель отражается в отчёте,
автоматического продолжения его ветки нет. Несколько тестов независимы и используют
одного родителя. Для одних A/B установите `training.iterations` в 0.

## Что делает V2 автоматически

Сохраняет исходные параметры, развёрнутый план и commit в
`runs/torus9/orchestration/jobs/<run_id>/`. Контроллер работает из неизменяемой
копии кода этого commit. Обычная смена HEAD разработчиком не меняет его код.
Состояния шагов, артефакты и outbox обслуживаются стандартным V2.
Повтор той же команды использует сохранённое состояние; завершённые шаги
не выполняются повторно. Одновременное исполнение защищает штатная lease V2.
Изменение параметров требует нового `run_id`. Это не команда перезапуска уже
работающего контроллера. После запуска detached controller сначала валидирует
подписанный PID-bound child-permit в pinned runtime и посылает оператору короткий
versioned `READY` через anonymous pipe. `job` сообщает `STARTED` только после
этого handshake; bounded timeout и остановка process group не оставляют
неподтверждённый controller при ошибке запуска. После `READY` controller
продолжает работу независимо от завершения launcher-процесса.

Telegram берётся из стандартного `~/.config/gocube-alphazero/telegram.env`
или переменных `GOCUBE_TELEGRAM_BOT_TOKEN` / `GOCUBE_TELEGRAM_CHAT_ID`.
Пустые/пробельные переменные не выключают настройку из файла.
Без настроенного транспорта новый job не стартует. Сетевые сбои обслуживает
существующий durable outbox; факт наличия настройки не гарантирует доставку.
Событие начала сохраняет параметры обучения, replay, арены и остановки,
поэтому сообщение после чтения из outbox содержит те же настройки.

## Запрет самостоятельных правок

Корневой `AGENTS.md` требует отдельного разрешения пользователя на изменение
реализации, уведомлений, launcher, сервисов, среды запуска и самих правил.
Разрешение запускать/настраивать обучение относится только к JSON и штатным
командам. Нельзя обходить правило скриптом вне репозитория.

`AGENTS.md` — инструкция агенту, не разграничение прав ОС. Правило запрещает
агенту менять реализацию без отдельного разрешения, но не блокирует запись в
файлы технически. Защита GitHub и CODEOWNERS по просьбе пользователя не добавляются.

## Offline A/B и A/B/C на существующем replay

Только в `ab_tests` доступно поле `offline_replay`: список зарегистрированных
checkpoint selectors, чьи fresh/rolling replay-манифесты являются входами каждой
iteration. Checkpoints должны образовывать последовательную цепочку от `parent`.
`training.iterations` должен быть 0; обычное обучение эту опцию не принимает.
Новые self-play игры не генерируются. SHA манифестов и shards проверяются до
старта и при чтении. Исторические окна replay используются целиком, включая
предшествующие поколения, а не только fresh data.

Для offline теста можно указать `arms` с двумя или более именованными ветками
вместо `A`/`B`. Все ветки стартуют от одного parent с Adam state, после обучения
выполняется полный round-robin с одинаковыми arena settings, seed, paired starts
и color swap. Арены диагностические; победитель не продолжает production.

```json
"training": {"iterations": 0, "learning_rate": 0.000025,
             "gradient_clip": 8.0, "replay_generations": 5},
"ab_tests": [{
  "id": "batch", "iterations": 5,
  "offline_replay": ["source/M256", "source/M257", "source/M258", "source/M259", "source/M260"],
  "arms": {
    "B64": {"batch_size": 64, "updates_per_iteration": 2560},
    "B128": {"batch_size": 128, "updates_per_iteration": 1280},
    "B256": {"batch_size": 256, "updates_per_iteration": 640}
  },
  "arena": {"games": 192, "mcts_simulations": 64, "tree_reuse": true}
}]
```

Используйте реальные selectors: поколения могут относиться к разным lineage.
В `runs/torus9/experiments/<experiment_id>/state.json` сохраняются результаты
всех пар и ссылки на независимые output lineages. Production parent и replay
не копируются и не изменяются. Метрики каждой iteration содержат losses,
validation, wall time/минуты, optimizer updates/sec, samples/sec и CUDA allocator
peak VRAM. Время измеряет optimizer loop с синхронизацией CUDA, включая
публикацию progress, без replay loading, validation и checkpoint I/O.

Sampler сохранён: `random.Random(training_seed + ordinary_update)` на каждый
update. При разных batch/числе updates одинаковый seed не означает одинаковый
поток sample IDs. Распределение остаётся uniform-over-positions с replacement.
