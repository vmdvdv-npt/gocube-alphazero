# Отчёт о выполнении ТЗ: шаги 4–6

Дата отчёта: 2026-09-08

Проект: `gocube-alphazero`

Статус публикации: этот отчёт обновлён вместе с исправлениями и публикуется в
ветке `main` после текущего commit/push.

## 0. Статус относительно `main`

В текущем batch исправлены расхождения между отчётом и GitHub `main`:

- reference job запускает корректную команду без лишнего аргумента `PY`;
- bridge и symmetry tests помечены `katago_reference`;
- 25 fixtures требуют явного `postconditions` блока;
- bridge cases теперь создают реальные seam/wrap позиции, а не одну star-позицию;
- после push подтверждением публикации является commit в GitHub и его CI.

Фактическое прохождение CI фиксируется отдельно результатом GitHub Actions;
локальный тестовый прогон не подменяет этот результат.

## 1. Итог

Шаги 4–6 реализованы локально и проверены. В проект добавлены:

- независимый rule-only reference harness на базе pinned KataGo Rules V3;
- корректная семантика `NO_RESULT` как отдельного класса value-target;
- replay contract v2 с семью согласованными тензорами;
- версионированные sample-clock и training contracts;
- детерминированное порождение seed для worker/game;
- immutable manifest для запуска и строгая проверка resume;
- конечные проверки на `NaN`/`Inf` и корректное применение масок;
- обязательный CI differential suite для KataGo reference.

Системное окружение не изменялось: `sudo` и `apt install` не использовались.
KataGo oracle собран локально имеющимся `g++`; отдельного missing dependency
blocker нет.

## 2. Что именно изменено

### 2.1. Правила KataGo Rules V3

Зафиксирован reference commit:

```text
f6bc4b19a1686caa2d088b56251e8c11c8be6d51
```

Добавлены:

- локальный pinned source cache и detached checkout;
- C++ rule-only oracle поверх `Board` и `BoardHistory` KataGo;
- JSONL-протокол oracle;
- адаптер координат и action indexing без переноса игровой логики в адаптер;
- статические fixtures, random differential tests и topology bridge tests;
- machine-readable allowlist только для технических различий представления.

Сравниваемые семантики включают:

- occupancy, adjacency, groups и liberties;
- placement, captures и suicide;
- simple ko и ko recap restrictions;
- legality и pass;
- переходы `MAIN → CLEANUP_1 → CLEANUP_2 → SCORED`;
- `NO_RESULT`, winner и terminal state;
- territory/seki scoring, prisoners, komi и Cleanup 2 accounting.

Прямой KataGo oracle используется для planar topology. Cube/Torus seam и wrap
проверяются отдельными graph-isomorphism и symmetry/metamorphic tests; эти
особенности не маскируются как допустимые расхождения KataGo.

Основные материалы:

- [KataGo rule reference](KATAGO_RULE_REFERENCE.md)
- [KataGo oracle README](../tools/katago_reference/README.md)
- [oracle build script](../tools/katago_reference/build_oracle.sh)
- [reference test runner](../tests/katago_reference_runner.py)

### 2.2. `NO_RESULT` и training targets

Введён трёхклассовый value target:

```text
[win, loss, no_result]
```

Семантика:

| Состояние | Value target | Score target | Score mask | Ownership mask |
| --- | --- | --- | --- | --- |
| scored win | one-hot win/loss | finite | 1 | по точкам |
| scored draw | `[0.5, 0.5, 0]` | finite | 1 | по точкам |
| `NO_RESULT` | `[0, 0, 1]` | `NaN` | 0 | 0 |

`NO_RESULT` сохраняет policy и value training rows. Score и ownership не
обучаются на таких строках. Маскирование выполняется до арифметики, поэтому
неактивные `NaN` не попадают в loss и gradient.

Добавлены проверки форм, dtype, finite values и согласованности mask/target.

### 2.3. Replay format v2

Один replay commit содержит ровно семь тензоров в фиксированном порядке:

1. observations;
2. policy targets;
3. value targets `[win, loss, no_result]`;
4. normalized black-minus-white score targets;
5. score applicability masks;
6. Rules V3 ownership targets;
7. ownership point masks.

Completion marker записывается последним. Replay v1 явно отвергается, потому
что из него нельзя восстановить retained `NO_RESULT` policy/value rows.

### 2.4. Sample clock и контракты

Зафиксированы версии:

| Контракт | Версия |
| --- | --- |
| sample clock | `sample-clock-v2` |
| replay | `2` |
| value semantics | `win-loss-noresult-v1` |
| score semantics | `normalized-score-with-applicability-mask-v1` |
| ownership semantics | `formal-v3-with-point-mask-v1` |
| seed derivation | `gocube-seed-derivation-v1` |
| training contract | `2` |

Sample clock advances by the number of consumed examples, а не по wall-clock
или номинальному номеру iteration. Несовместимый resume state завершается
ошибкой.

### 2.5. Воспроизводимость и resume

Добавлен deterministic seed derivation от координат запуска:

```text
master_seed + iteration + worker + game_slot + game_sequence
```

Каждый hardened run сохраняет:

- `run-manifest.json`;
- `effective-config.json`;
- environment artifact;
- `source.patch`, если источник dirty.

Manifest фиксирует source commit, KataGo commit, rules fingerprint, topology,
коми, формы tensors, target/replay/sample-clock versions, parameter origins и
эффективную конфигурацию.

Resume запрещается при изменении immutable полей, включая:

- seed;
- komi;
- правила и KataGo commit;
- target/replay/sample-clock/training contracts;
- effective configuration;
- source commit или dirty-state без явного разрешения.

### 2.6. CI и документация

Добавлен отдельный обязательный CI job `katago_reference`, который:

1. checkout-ит проект;
2. собирает локальный oracle;
3. запускает `pytest -m katago_reference`.

CI не устанавливает системные пакеты и не требует `sudo`.

Авторитетные документы:

- [Training loop contract](TRAINING_LOOP_CONTRACT.md)
- [Japanese Rules V3 notes](KATAGO_JAPANESE_V3.md)
- [Production hardening](PRODUCTION_TRAINING_HARDENING.md)

## 3. Результаты проверок

### Полный regression suite

```text
507 passed, 7 warnings
```

Предупреждения относятся только к будущему изменению default-поведения
`torch.load(weights_only=False)` и не являются ошибками тестов.

### Reference и topology checks

Проверены:

- 25 pinned static KataGo fixtures; static test module содержит 26 tests,
  включая inventory check;
- 32 детерминированные random oracle-driven игры для размеров
  `3×3`, `5×3`, `5×5`, `7×4`;
- 22 реальные Cube/Torus seam/wrap bridge cases для group/liberty/capture/
  multi-capture/suicide/ko/cleanup-ko/pass-for-ko/pass-alive/scoring;
- 19 topology symmetry/metamorphic cases;
- semantic postconditions для всех 25 fixtures;
- D4/translations для Torus;
- graph automorphisms для Cube;
- scoring/Benson/seki/cleanup differential cases;
- finite guards и masked-gradient behavior;
- reproducibility manifest и resume rejection;
- parameter-origin registry;
- документационный contract suite.

Локальная команда для обязательного reference набора:

```bash
.venv/bin/python -m pytest -m katago_reference
```

Последний локальный запуск marker suite: `99 passed, 408 deselected`.

Последний полный regression suite: `507 passed, 7 warnings`.

Сборка oracle:

```bash
tools/katago_reference/build_oracle.sh
```

Полный набор:

```bash
.venv/bin/python -m pytest -q
```

## 4. Матрица оценки

| Область | Критерий | Статус | Подтверждение |
| --- | --- | --- | --- |
| KataGo | source commit pinned | выполнено | `f6bc4b19...` |
| KataGo | oracle не переimplement-ит rules | выполнено | `Board`/`BoardHistory` adapter |
| KataGo | system env не изменяется | выполнено | build script без package manager |
| Rules | legality/capture/ko/phase/score | выполнено | static + random differential |
| `NO_RESULT` | отдельный value class | выполнено | target contract + tests |
| Replay | 7 tensors, atomic marker | выполнено | replay v2 tests |
| Masking | inactive NaN не влияет на loss | выполнено | gradient tests |
| Reproducibility | deterministic worker/game seeds | выполнено | seed contract tests |
| Resume | immutable manifest validation | выполнено | changed-seed/config rejection |
| CI | reference suite обязательна | выполнено | `.github/workflows/ci.yml` |
| Regression | полный тестовый набор | выполнено | `505 passed` |

## 5. Ограничения и замечания

1. KataGo reference является rule-only harness. Нейросетевая модель и search
   KataGo намеренно не входят в область проверки.
2. Прямое сравнение с KataGo выполняется на planar graph. Производственные
   Cube/Torus topology проверяются через отдельный topology bridge и
   metamorphic tests.
3. В проекте сохранены отдельные исторические integration-документы и
   baseline fixtures. Они не являются текущим production training contract;
   актуальный контракт описан в `TRAINING_LOOP_CONTRACT.md`.
4. Для локального запуска требуется уже доступный project virtualenv с
   Python-зависимостями проекта и установленный компилятор `g++`; системная
   установка зависимостей в harness не выполняется.

## 6. Вывод

Требования шагов 4–6 закрыты: правила вынесены на независимую pinned
KataGo-проверку, `NO_RESULT` и replay semantics формализованы, обучение
защищено от некорректных masked values, воспроизводимость и resume сделаны
проверяемыми, а полный regression suite проходит.
