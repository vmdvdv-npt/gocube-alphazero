# Отчёт подготовки — 2026-11-03

## Реально реализовано

Работа изолирована в ветке `codex/torus-weightspace-preparation`, worktree
`/home/codex/projects/gocube-weightspace-prep`, от свежего `origin/main`
`d33975c284fa5de6e7fea6c6ffb6bc931d569a06`. Исходный checkout с незакоммиченными
изменениями не изменялся. Изменения находятся только в экспериментальной директории.
Исходный starter прочитан и расширен. `ASTRA_TORUS9_WEIGHTSPACE_TASK.md` в доступных
файлах проекта не найден. Источники Experiment 6 изучены и связаны в README.

Существующая архитектура сохранена: GraphNet 8×120, 5CH, FP32, **396 009** learned
parameters, 92 state tensors (включая 8 buffers соседства). Существующие вспомогательные
heads ownership и score оставлены; policy имеет форму `[B,82]`, WDL — `[B,3]`,
ownership — `[B,81,3]`, score — `[B]`. Структура не заменялась production 8×80.
Исправлена фактическая инициализация: конструктор теперь обнуляет все learned tensors,
включая LayerNorm, а не полагается на отдельный вызов `zero_all_weights`.

Реализованы NumPy writer, просмотр схемы, строгий model-only checkpoint v1,
валидация shapes/dtype/count/finiteness/topology, SHA-256 файла и канонического
state, повторная генерация весов и восстановление без forward. Файлы и директории
новых версий создаются эксклюзивно; предыдущие результаты не перезаписываются.
Сохраняются программа, инфраструктура, хеши, команда и JSON-отчёты.

Zero-pass отделён от positions/games/Arena. Бюджеты 0/10/25/50 реально ограничивают
число запросов; reservations сериализуются, неудачный начатый раунд тоже учитывается.
Позиции содержат observation, legal mask/actions, logits, probabilities, top-k,
WDL и опциональные активации. Games сохраняют парные начала, цвета, историю ходов,
захваты и W/L/D; watchdog маркируется технической ошибкой, а не ничьей.

Отдельный экспериментальный Arena adapter использует существующие referee,
5CH observation builder, PUCT, WDL utility и штатный loader эталона. Оба игрока
получают одинаковые MCTS settings, коми и начальную историю. Сохраняются архитектуры
и отдельные CPU forward timings. Production Engine, модели, replay, Orchestrator,
уведомления и существующие checkpoint не редактировались. Большие зависимости,
API, обучение и Astra не запускались. Интеллектуальная стратегия не создавалась:
`design_weights.py` оставлен пустым шаблоном.

## Правила и интерфейсы

Единственный источник — `gocube_golden` из executing worktree. Соседства всех 81
пунктов сверены с `TORUS_9X9`. Протестированы захват через wrap-around, дамэ,
occupied/suicide, immediate ko и более раннее повторение через positional superko,
PASS без добавления дубликата в history, terminal после двух PASS, запрет ходов
после terminal и graph-area с komi. Независимого движка правил нет.

5CH: own, opponent, side-to-move color (`+1` black / `−1` white), previous pass,
legal-point mask. Индексы пунктов 0..80 row-major; PASS logit/index 81 соответствует
действию referee `"PASS"`. WDL относится к игроку на ходе: WIN, DRAW, LOSS.

## Найденный Arena reference

Зафиксирован **AZ23 lr-B/M269**, а не исторический M137 или максимальное поколение.
Полный абсолютный путь, архитектура, search contract и хеши находятся в
[reference-lock.json](reference-lock.json).

- Model hash: `sha256:a51b5c1f971d51c8337559f888ad96372892d3a2143d4a0456134b44bc6ef9ba`.
- Artifact SHA-256: `f16af6c473cd5c3d89687f9edc99ba94ec0bf4d7e3da00166368fccae41609f7`.
- 8×80, 5CH, 177 609 параметров. Проверена реальная загрузка штатным loader без forward.
- Последние найденные прямые Arena AZ26: reference выиграл **230–154** у A/M270 и
  **199–185** у B/M270. Обе серии: 384 valid games, 192 valid pairs, validity VALID,
  0 technical games. Более поздние номера поколений не являются доказательством
  смены чемпиона. Выбор означает сохранённый локальный Arena reference, а не
  доказательство глобально сильнейшей сети среди всех возможных checkpoint.
- Утверждение starter «252:132 над H266» не используется: первичное подтверждение
  этого конкретного результата не найдено. Небольшое преимущество 199:185 само по
  себе статистически не доказывает превосходство; B/M270 не подтвердил замену reference.
- Коми 1.5; 64 simulations, cpuct 1.25, FPU 0, tree reuse true, watchdog 1000.
- Rules fingerprint отсутствует в sidecar, поэтому явно выведен существующим
  `rules_fingerprint_for(TORUS_9X9, 1.5)`, а не выдуман или скопирован от другого коми.
  Получено `sha256:1f982c35a1c21b1e4f2f90b4b41bc1f04e7ffd83078a58bce12aefbb7a562a74`.

`lock_reference.py` требует совместимые завершённые VALID Arena evidence, совпадение
artifact/model identity, победный результат и явное обоснование выбора. Он не
пытается автоматически выбрать champion по имени файла. Во время эксперимента
проверяются SHA checkpoint, sidecar и evidence; изменения приводят к отказу.
Чтение production evidence использовалось для выбора reference, не для тестов.

## Что проверено

**20 автоматических тестов прошли**, включая существующие четыре теста starter.
Полный запуск и CLI smoke: [verification/reproduce.sh](verification/reproduce.sh).
Реальный stdout/stderr: [verification/local-checks.log](verification/local-checks.log).

Проверены нулевой genesis, алгоритмическая запись NumPy, shape/dtype/finiteness,
изменение buffer, воспроизводимость и byte SHA в одном runtime, corrupted checkpoint,
отказ перезаписи, отсутствие forward при zero-pass, правила, masked probabilities,
активации, WDL orientation, paired colors/starts, technical watchdog, бюджеты,
провал запроса с расходом раунда и evidence rejection. Через реальный PUCT сыграна
короткая синтетическая pass-фикстура 120-vs-80 с одинаковым бюджетом 2 simulations.
В тестах нет production games, checkpoint или транспортеров уведомлений.

CLI smoke отдельно выполнил zero-pass пустого writer, restore, 4 позиции с
активациями и техническую остановку двух игр после одного хода. Последняя серия
намеренно имеет INVALID и 0 W/L/D — это успешная проверка обработки watchdog,
не результат силы сети. Никакого турнира candidate-vs-champion не проводилось.

Тестовый runtime: Python 3.9, torch 2.4.1+cu124, NumPy 1.24.4; операции CPU,
1 torch thread. Наличие CUDA build не означает GPU execution.

## Оставшиеся ограничения

- Writer — доверенный Python, не защищённый sandbox. Запрет обучения и скрытых
  forward требует соблюдения протокола и проверки программы оператором.
- Диагностический Arena adapter последовательный CPU, batch 1, до 128 игр на запрос.
  Production GPU batching/throughput и большой tournament startset сюда не входят.
  Минимальный патч production Arena не требуется: эксперимент работает отдельно.
- Предоставлены короткие легальные диагностические истории, без оценок правильного
  хода и без обучающего датасета. Полнота тактического benchmark не заявляется.
- Lock ссылается на реальные локальные артефакты вне Git; для переноса нужны эти
  файлы/evidence с теми же SHA или повторное evidence-based закрепление reference.
- Веса starter v0 сохранены нетронутыми и не выдаются за результат. Новый loader
  намеренно требует v1; нельзя передавать старый demo как завершённый zero-pass.
- Сила будущей сети неизвестна. Повторяемость writer проверена двумя вызовами в
  текущем процессе; межплатформенная побитовая идентичность не гарантируется.
- Стандартный GitHub CI репозитория собирает production tests из `tests/` и не
  подхватывает вложенные experiment tests автоматически. Их явная команда выше.

## Что теперь должна сделать Astra

Получить путь к [ASTRA_HANDOFF.md](ASTRA_HANDOFF.md), прочитать модель/протокол,
реализовать только `program(weights)` и сохранить новую zero-pass версию.
После разрешения оператором диагностического бюджета — использовать отдельные
positions/games запросы, улучшать программу и сохранять новые версии. Arena против
закреплённого reference запускать явно и трактовать результаты отдельно от
структурных тестов. Инженерная подготовка не требует запуска Astra сейчас.
