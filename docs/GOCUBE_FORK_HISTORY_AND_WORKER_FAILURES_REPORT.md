# Отчёт: история ответвлений GoCube и обработка ошибок worker-процессов

## 1. Итог

Реализованы изменения из ТЗ для двух связанных областей:

1. Исправлено соответствие между историей ходов и историей состояний в pinned/forked GoCube-играх.
2. Добавлена явная передача ошибок из self-play worker-процессов в родительский процесс.
3. Добавлена bounded-остановка worker-процессов при ошибках, включая hard exit без Python traceback.
4. Сохранена существующая атомарная схема записи checkpoint/replay и recovery-поведение.
5. Добавлены регрессионные тесты и выполнены целевые и полные прогоны.

Текущая рабочая база синхронизирована с последним merge-коммитом `origin/main`:

- Репозиторий: [vmdvdv-npt/gocube-alphazero](https://github.com/vmdvdv-npt/gocube-alphazero)
- HEAD: `e7a4137` — merge PR #48
- Изменения пока не закоммичены и не отправлялись в GitHub.

## 2. Исправление истории fork’ов

### 2.1. Проблема

В pinned-игре история ходов может начинаться с абсолютной длины, большей нуля. Это происходит после восстановления из уже существующего candidate-prefix: состояние `state_history[0]` соответствует не началу полной партии, а состоянию после некоторого числа ходов.

Старая логика могла трактовать локальный индекс состояния как абсолютную длину истории. Из-за этого при выборе seki-кандидата возникали две ошибки:

- выбиралось состояние до начала доступного pinned-сегмента;
- `candidate_history` и восстановленное состояние переставали описывать одну и ту же позицию.

### 2.2. Новый инвариант

Для pinned-истории введено явное смещение:

```text
state_history[0] == состояние после move_history[offset]
len(state_history) == len(move_history) - offset + 1
```

Где:

- `move_history` — полная абсолютная история ходов;
- `state_history` — только доступный локальный сегмент состояний;
- `offset` — абсолютная длина истории в начале `state_history`.

При отсутствии префикса `offset == 0`. После восстановления из candidate-prefix `offset == len(candidate_history)`.

### 2.3. Изменения в `PinnedGame`

В [pinned_game.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/pinned_game.py) добавлены:

- инициализация `_pinned_state_history_offset`;
- копирование offset в `clone()`;
- `_assert_pinned_history_alignment()` с диагностикой длины обеих историй;
- `_pinned_state_for_history_len(absolute_history_len)`, который переводит абсолютную длину истории в локальный индекс состояния и проверяет границы;
- проверки после реального хода и после восстановления fork’а.

Выбор seki-кандидата теперь выполняется только внутри доступного pinned-сегмента. Локальный индекс сначала переводится в абсолютную длину истории, после чего состояние извлекается через единый проверяемый helper. В candidate сохраняется точный префикс `move_history[:absolute_history_len]`.

После восстановления проверяется согласованность всех существенных полей:

- полного состояния доски;
- полной истории ходов;
- текущего игрока;
- `last_action`;
- singleton-истории состояний и её offset.

### 2.4. Изменения в `DiversifiedGame`

В [diversified_game.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/diversified_game.py) при обычном fork-восстановлении также устанавливается offset равным длине candidate-prefix.

Это отделено от `_diverse_training_history_offset`: теперь offset, описывающий pinned state-history, и offset, используемый логикой разнообразия обучения, не смешиваются.

### 2.5. Что предотвращено

Теперь невозможно молча принять позицию, если:

- запрошена история меньше текущего offset;
- запрошена история дальше конца полной истории ходов;
- локальный индекс вышел за `state_history`;
- длины историй нарушают инвариант.

Во всех таких случаях возникает `RuntimeError` с абсолютной длиной истории, offset, длинами массивов и вычисленным локальным индексом.

## 3. Явная обработка ошибок worker-процессов

### 3.1. Протокол ошибки

Введён [worker_errors.py](/home/codex/projects/gocube-alphazero/alphazero/worker_errors.py) с исключением `SelfPlayWorkerError` и фабрикой payload для аварийного завершения процесса.

Worker публикует в error queue picklable-словарь фиксированной формы:

```text
worker_id
pid
iteration
game_slot
game_id
stage
exception_type
exception_message
traceback
```

Поддерживаются как обычные Python-исключения, так и случаи, когда worker завершился без возможности отправить traceback.

### 3.2. Контекст ошибки

Во время работы worker поддерживается контекст:

- iteration обучения;
- номер worker’а;
- слот игры;
- game id;
- текущая стадия.

Стадия устанавливается для инициализации, поиска, генерации результата, завершения игры и постановки результата в очередь. Поэтому родитель получает не только текст исключения, но и место сбоя в жизненном цикле worker’а.

### 3.3. Поведение `SelfPlayAgent`

В [SelfPlayAgent.pyx](/home/codex/projects/gocube-alphazero/alphazero/SelfPlayAgent.pyx):

- ошибка worker’а больше не теряется через `print()`;
- payload отправляется в error queue;
- после ошибки всегда устанавливается `stop_event`;
- исходное исключение повторно выбрасывается в дочернем процессе;
- счётчик завершённых игр увеличивается только после успешного закрытия и join output queue;
- ожидание batch release стало прерываемым и регулярно проверяет `stop_event`.

Это исключает ситуацию, когда один worker уже упал, а остальные навсегда ждут batch-сигнал или родитель продолжает считать итерацию успешной.

## 4. Supervision и bounded shutdown в родительском процессе

В [Coach.py](/home/codex/projects/gocube-alphazero/alphazero/Coach.py), [train.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/train.py) и [katago_train.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/katago_train.py) добавлена единая схема контроля.

### 4.1. Проверка состояния

Родитель перед ожиданием следующего batch и перед инференсом:

1. проверяет error queue;
2. проверяет `exitcode` всех worker’ов;
3. немедленно поднимает `SelfPlayWorkerError` при payload или abnormal exit;
4. прекращает дальнейший self-play/inference цикл.

Для hard exit создаётся синтетический payload с PID, worker id, iteration и кодом завершения.

### 4.2. Сохранение причины

Если ошибка возникает в родительском inference или другой родительской стадии, она не заменяется общей ошибкой остановки. Контекст worker’а добавляется к исходному исключению через `Exception.add_note()` на новых версиях Python, а для старых версий сохраняется через дополнение `exc.args`.

### 4.3. Порядок аварийной остановки

`_abort_selfplay_agents()` выполняет bounded-последовательность:

1. выставляет stop;
2. снимает pause;
3. выставляет batch/release/finish события;
4. делает bounded join;
5. завершает оставшиеся процессы через `terminate()`;
6. после повторного bounded join использует `kill()`, если процесс всё ещё жив;
7. закрывает IPC queues и очищает ссылки на процессы, события и shared tensors.

Ни один обычный worker join в error path не может ждать бесконечно.

### 4.4. Успешное завершение

При штатном завершении сохраняются прежние семантики: процессы получают stop-сигнал, корректно завершают очереди и освобождают IPC-ресурсы. Проверки abnormal exit применяются только там, где действительно требуется отличить успешное завершение от падения.

## 5. Сохранённые recovery-контракты

Файл [atomic_io.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/atomic_io.py) не изменялся.

Сохранены следующие правила:

- сначала пишется staging-файл;
- staging публикуется атомарным rename;
- незавершённый staging не считается готовым результатом;
- старый валидный checkpoint остаётся доступен при ошибке записи нового;
- recovery не продолжает обучение по частично записанному replay.

В `Coach.learn()` добавлена финальная уборка worker’ов и writer’а при исключении, при этом исходное исключение сохраняется и повторно выбрасывается.

## 6. Добавленные тесты

### 6.1. История fork’ов

Файл [test_gocube_fork_history_alignment.py](/home/codex/projects/gocube-alphazero/tests/test_gocube_fork_history_alignment.py) проверяет:

- базовое соответствие длины state-history и move-history;
- копирование offset в clone;
- production-подобный сценарий с 20 ходами;
- plain fork после непустого префикса;
- последовательность PASS;
- сценарии `N=1, 2, 4, 6`;
- переход `MAIN -> cleanup1 -> cleanup2 -> scored`;
- ранний и обычный restore;
- seki sampling только внутри локального pinned-сегмента;
- диагностические ошибки при некорректных индексах.

### 6.2. Ошибки worker’ов

Файл [test_gocube_selfplay_worker_failures.py](/home/codex/projects/gocube-alphazero/tests/test_gocube_selfplay_worker_failures.py) проверяет:

- ошибку поиска в Python worker;
- наличие payload и traceback;
- корректную стадию ошибки;
- освобождение sibling worker’ов, ожидающих batch release;
- ошибку на стадии завершения игры;
- ошибку parent inference с сохранением контекста;
- hard exit через `os._exit(17)`;
- фиксированную форму payload;
- failure staging-записи без публикации битого результата;
- сохранность предыдущего checkpoint;
- отсутствие false positive при успешном завершении worker’ов.

Fault injection находится только в тестах; production-код не содержит искусственных аварий.

## 7. Результаты проверки

### 7.1. Целевой набор

Запущен набор тестов из ТЗ:

```bash
.venv/bin/python -m pytest \
  tests/test_gocube_fork_history_alignment.py \
  tests/test_katago_selfplay_diversification.py \
  tests/test_katago_pinned_selfplay_semantics.py \
  tests/test_gocube_ko_contract.py \
  tests/test_gocube_katago_v3_cleanup.py \
  tests/test_gocube_selfplay_worker_failures.py \
  tests/test_gocube_hardened_production.py \
  tests/test_gocube_sweep_resume.py \
  -q
```

Результат:

```text
63 passed, 2 warnings in 2.90s
```

Предупреждения относятся к существующему PyTorch-поведению `torch.load(..., weights_only=False)` и не являются ошибками тестов.

### 7.2. Полный набор

```bash
.venv/bin/python -m pytest -q
```

Результат:

```text
390 passed, 7 warnings in 10.73s
```

Дополнительно выполнена проверка:

```bash
git diff --check
```

Результат — ошибок форматирования diff нет.

## 8. Изменённые файлы

Production:

- [alphazero/Coach.py](/home/codex/projects/gocube-alphazero/alphazero/Coach.py)
- [alphazero/SelfPlayAgent.pyx](/home/codex/projects/gocube-alphazero/alphazero/SelfPlayAgent.pyx)
- [alphazero/worker_errors.py](/home/codex/projects/gocube-alphazero/alphazero/worker_errors.py)
- [alphazero/envs/gocube/pinned_game.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/pinned_game.py)
- [alphazero/envs/gocube/diversified_game.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/diversified_game.py)
- [alphazero/envs/gocube/train.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/train.py)
- [alphazero/envs/gocube/katago_train.py](/home/codex/projects/gocube-alphazero/alphazero/envs/gocube/katago_train.py)

Tests:

- [tests/test_gocube_fork_history_alignment.py](/home/codex/projects/gocube-alphazero/tests/test_gocube_fork_history_alignment.py)
- [tests/test_gocube_selfplay_worker_failures.py](/home/codex/projects/gocube-alphazero/tests/test_gocube_selfplay_worker_failures.py)

## 9. Git-состояние и следующий шаг

Рабочее дерево содержит перечисленные изменения, новые тесты и этот отчёт. Коммит и push не выполнялись.

Для завершения работы достаточно проверить diff, затем при необходимости создать коммит и отправить его в выбранную ветку. По текущему запросу удалённые ветки и GitHub не изменялись.
