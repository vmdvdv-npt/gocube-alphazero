# Методика эксперимента

Команды ниже выполняются из директории `torus_weightspace_starter` в подготовленном
worktree. Используется уже установленная среда; ничего устанавливать не требуется.
В чистом checkout нужны Python ≥3.9, PyTorch 2.4.x и NumPy 1.x.

```bash
export WSP_PY=/home/codex/projects/gocube-alphazero/.venv/bin/python
export WSP_OUT=/tmp/torus9-weightspace-my-experiment
"$WSP_PY" experiment_cli.py schema
"$WSP_PY" -m unittest discover -s tests -v
```

Укажите новый путь `WSP_OUT` для каждого независимого эксперимента. Не размещайте
выходы в production run. Пути в командах являются примерами, существующие версии
не перезаписываются. Файлы из старой папки `artifacts/` не модифицируйте.

## 1. Подготовить программу без обратной связи

Изучите `model.py`, схему тензоров и `design_weights.py`. Реализуйте
`program(weights)`: словарь всех learned tensors содержит независимые NumPy arrays
FP32, которые нужно менять in-place. Используйте матричные операции, циклы и
аналитические конструкции. Не печатайте отдельные сотни тысяч значений.

Буферы соседства отсутствуют в writable dictionary. Не меняйте размеры, dtype,
ключи, архитектуру. LayerNorm scales и biases также изначально нулевые. Все
параметры должны быть конечными. Запрещены обучение/fitting, backpropagation,
optimizer и копирование готовых trained weights.

Это доверенный Python-интерфейс, не security sandbox. Запрет обхода диагностики
должен соблюдаться самой Astra и оператором: нельзя импортировать другую модель и
делать forward самостоятельно. Интерфейс не вызывает сеть и ставит защитные hooks
на supplied model/layers; статический аудит не может доказать отсутствие обучения
в произвольном Python. Используйте самодостаточный writer без внешних данных,
сети, файловых побочных эффектов и динамических imports.

## 2. Zero-pass

```bash
"$WSP_PY" experiment_cli.py init --session "$WSP_OUT/session0" --budget 0
"$WSP_PY" experiment_cli.py zero-pass --writer design_weights.py --output "$WSP_OUT/v000"
"$WSP_PY" experiment_cli.py verify --checkpoint "$WSP_OUT/v000/checkpoint.pt"
```

Ни одна из этих команд не запускает модель на входах. Проверяются имена/формы/
dtype всех state tensors, 396 009 learned parameters, конечность, неизменность
топологии, повторяемость двух вызовов writer и strict roundtrip через
`torch.load(weights_only=True)`. Сохраняются отдельный нулевой `genesis.pt`,
`checkpoint.pt`, JSON metadata, canonical state SHA-256 и SHA-256 файла,
`writer.py`, stdout, копия Python-инфраструктуры и `zero-pass.json`.
CLI добавляет `reproduce.json` с командой и хешами исходников. Канонический hash
содержит имена, формы, dtype и little-endian tensor bytes. Byte hash контейнера
сравним в той же версии PyTorch; межверсионное совпадение serialization не обещается.

Сбой оставляет незавершённую директорию для разбора; повторять нужно в новой
версии. Только `zero-pass.json` со статусом PASS открывает доступ к диагностике.
Для точного повторения используйте сохранённый `writer.py` и ту же инфраструктуру,
runtime и NumPy. `weights.load_arrays/store_arrays` также доступны из Python.
`restore_checkpoint` возвращает строгую FP32 eval-модель и metadata.

## 3. Явные диагностические раунды

Только после завершения zero-pass создайте **одну общую** сессию улучшений с
выбранным оператором бюджетом 10, 25 или 50. Создание новой сессии для обхода
бюджета запрещено. Профиль 0 не предоставляет диагностических запросов.

```bash
"$WSP_PY" experiment_cli.py init --session "$WSP_OUT/session10" --budget 10
"$WSP_PY" experiment_cli.py positions --session "$WSP_OUT/session10" \
  --checkpoint "$WSP_OUT/v000/checkpoint.pt" --komi 1.5 --positions positions.json --top-k 5
```

Один запрос — один раунд, независимо от числа позиций/партий. Поддерживаются 1–256
позиций и 1–64 пары (до 128 игр). Сессия сериализует запросы через file lock;
выданная reservation расходует раунд даже при сбое. Отчёты и запросы лежат в
`session10/rounds/001/`, `002/`, …; предыдущие файлы не заменяются. stdout содержит
компактный JSON с путём отчёта; ошибки CLI — JSON в stderr с ненулевым exit code.

Вход позиций: список `{"id":"имя","moves":[0,40,"PASS"]}`. Это полная легальная
история от пустой доски, а не схема нового движка. Состояние, superko history,
маски и score принадлежат referee. После terminal диагностический forward
запрещён. Выход: observation 5×81, stones, side, legal actions/mask, 82 raw logits,
нормированные вероятности после маски, top-k и WDL. `--activations` добавляет
input projection, выход каждого блока и output norm; отчёт станет больше.

```bash
"$WSP_PY" experiment_cli.py games --session "$WSP_OUT/session10" \
  --checkpoint "$WSP_OUT/v000/checkpoint.pt" --komi 1.5 --opponent random \
  --openings openings.json --seed 20261103 --watchdog 1000
```

Контроли: uniform legal `random` и всегда `pass`. Это простые контрольные
соперники, не новая интеллектуальная стратегия. Candidate policy-only ходы
семплируются с temperature 1; MCTS не используется. Каждое начало играется дважды,
со сменой цветов candidate. Сохраняются opening history, каждый ход/захват,
конечная доска, area/komi/margin, W/L/D. Watchdog означает техническое завершение,
не ничью и не score: вся серия получает INVALID. Правила завершают игру только
двумя PASS; watchdog не добавляет второго способа научного подсчёта.

## 4. Сравнение с AlphaZero

`reference-lock.json` закрепляет локальный M269 AZ23 lr-B (8×80), а не модель с
максимальным номером. Основания и ограничения — в preparation report. Сам lock
не содержит весов; для другого компьютера нужно перенести эталон и evidence,
проверить SHA и создать новый lock через `lock_reference.py --help`.

```bash
"$WSP_PY" experiment_cli.py arena --session "$WSP_OUT/session10" \
  --checkpoint "$WSP_OUT/v000/checkpoint.pt" --reference-lock reference-lock.json \
  --openings openings.json --seed 20261103
```

Адаптер проверяет SHA файла и sidecar эталона и evidence. Используется штатный
`Torus9ArenaProfile` для загрузки модели; production validation не ослабляется.
Оба игрока вызывают один `SequentialPUCT`, одну GoldenSearchAdapter и один referee:
коми 1.5, simulations 64, cpuct 1.25, FPU 0, tree reuse true, noise/resign OFF,
temperature 0, watchdog 1000. WDL — `[WIN,DRAW,LOSS]` для игрока на ходе,
utility = WIN − LOSS. Legal masking принадлежит search.

Разные размеры обслуживаются отдельными моделями. Одинаковая opening history
в каждой цветообменной паре восстанавливает идентичное стартовое состояние с
superko history. Используйте предоставленный явный список начал; это маленький
экспериментальный startset, не полный production startset. Для турнира оператор
должен отдельно подготовить и сохранить больший список легальных историй.
Адаптер ограничен 128 играми за запрос; длинные турниры не запускаются автоматически.

В отчёте сохраняются обе architecture configs, общий search/rules fingerprint,
W/L/D, technical terminations и раздельное время forward для каждой модели.
Исполнение CPU FP32, batch 1, threads 1 по умолчанию. Скорость — локальная
diagnostic latency без warmup; это не GPU/production throughput benchmark.
Для сравнений применяйте одинаковый runtime и thread count. Production Arena
патч не нужен, загрузка экспериментального checkpoint туда намеренно не добавлена.

## 5. Следующие версии

После разрешённой диагностики исправьте writer и выполните новый zero-pass с
`--output "$WSP_OUT/v001"`, затем запросите следующий раунд **той же** сессии.
Каждая программа запускается с нулевых весов; все версии и отчёты сохраняются.
Никаких API/Astra launcher в инфраструктуре нет. API/token cost при будущем
внешнем запуске должен учитывать оператор; текущий этап ничего не расходует.
