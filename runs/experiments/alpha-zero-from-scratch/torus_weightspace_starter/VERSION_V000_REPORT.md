# Первая версия весов: v000 — 2026-11-03

**Zero-pass: PASS. Игровая сила: NOT_EVALUATED.**
Реализован `program(weights)` в существующем worktree
`/home/codex/projects/gocube-weightspace-prep`, ветка `codex/torus-weightspace-preparation`.
Сохранена одна версия с нулевым диагностическим бюджетом. Forward, positions-pass,
games-pass и Arena для неё не выполнялись. Обучения, fitting, optimizer,
заимствования trained checkpoint, запуска API/Astra и изменений production не было.

## Артефакты

Checkpoint:
`/home/codex/projects/gocube-weightspace-prep/runs/experiments/alpha-zero-from-scratch/torus_weightspace_starter/versions/v000/checkpoint.pt`

| Данные | Значение |
|---|---|
| Архитектура | Torus9 GraphNet, 5CH, 8 блоков ×120, FP32 |
| Параметры | 396 009 |
| Ненулевые параметры | 1 921 (включая biases и LayerNorm) |
| Learned tensors / все state tensors | 84 / 92 |
| Логические парные регистры | 55 + постоянная anchor-пара |
| Forward / диагностические раунды | 0 / 0 |
| Runtime | Python 3.9.25, PyTorch 2.4.1+cu124, NumPy 1.24.4; CPU, 1 thread |

SHA-256 checkpoint-файла:
`5095d39b05435c5880f1e20362b3cc73c17ab02d11d4cd65b143e0266b9d64e1`

Canonical state SHA-256:
`51daee0611f2bdb5d97b038dc15b0752a7aec40623eb065b9eccbb34b0f7619f`

Writer SHA-256:
`b994526c665ce3696df90d0af6caa4620377dc77f29b02c84a5a81d021fbcfbd`

В [versions/v000](versions/v000) также сохранены независимый нулевой `genesis.pt`,
metadata обоих checkpoint, `writer.py`, stdout, Python-инфраструктура выполнения,
[zero-pass.json](versions/v000/zero-pass.json) и
[reproduce.json](versions/v000/reproduce.json). Все файлы версии сохраняются
без перезаписи; локальные файлы переведены в read-only. Это версионирование и
контроль целостности SHA, не аппаратный WORM. Старый `artifacts/` не изменён.
Git не сохраняет режим read-only: после нового checkout проверяйте manifest и
соблюдайте запрет перезаписи версий.

## Техническая проверка

Штатные команды:

```bash
cd /home/codex/projects/gocube-weightspace-prep/runs/experiments/alpha-zero-from-scratch/torus_weightspace_starter
WSP_PY=/home/codex/projects/gocube-alphazero/.venv/bin/python
"$WSP_PY" experiment_cli.py schema
"$WSP_PY" experiment_cli.py init --session versions/session0 --budget 0
"$WSP_PY" experiment_cli.py zero-pass --writer design_weights.py --output versions/v000
"$WSP_PY" experiment_cli.py verify --checkpoint versions/v000/checkpoint.pt
"$WSP_PY" verification/v000/audit_zero_pass.py
```

Для повторной генерации нужны **новые** пути session/output: существующие версии
не перезаписываются. Замороженный writer и инфраструктура позволяют повторить
создание; `reproduce.json` фиксирует использованную команду и исходные хеши.

Zero-pass проверил формы/имена/dtype, конечность всех тензоров, 396 009 параметров,
неизменность topology buffers, два независимых вызова writer и strict roundtrip
через `torch.load(weights_only=True)`. Отдельный CLI verify прошёл.

[Структурный аудит](verification/v000/structural-audit.json) дополнительно подтвердил:

- Все 396 009 параметров genesis нулевые; ненулевые параметры candidate пересчитаны.
- Парные записи и чтения матриц антисимметричны; anchors не перезаписываются.
- Каждый блок читает только ранее созданные регистры и пишет только новые.
- LayerNorm scales и общий финальный offset соответствуют конструкции.
- Кратность участия пустого пункта в локальной звезде ≤2 для всей топологии.
- Все 648 комбинаций переносов тора и D4 сохраняют соседства.
- Writer совпадает с обеими сохранёнными копиями и SHA в metadata.
- Импорты writer ограничены json/math/numpy; выполнен дополнительный статический
  аудит символов. Это проверка данного исходника, не универсальный security sandbox.
- Бюджет 0, каталог диагностических раундов пуст.

Аудит выполнялся с запрещённым `nn.Module._call_impl`; входы/наблюдения не создавались.
Симметрия проверялась по топологии и структуре весов, без сравнения forward outputs.
Оценка влияния LayerNorm относится к идеальной арифметике. Поведение FP32 на позициях,
тактика, вероятности WDL и реальная сила остаются непроверенными.

Полный набор тестов подготовки не запускался повторно: он содержит forward и игры.
Исторические проверки в `PREPARATION_REPORT.md` относятся к подготовке, не к v000.
Стандартный GitHub CI запускается после финального push; он не подхватывает вложенный
эксперимент и не является проверкой игровой силы этого checkpoint.

## Что запрограммировано

Подробные формулы, размещение блоков и будущие проверки:
[CIRCUITS_V000.md](CIRCUITS_V000.md).

Сеть распознаёт цвета и пустоты, считает соседние дамэ, строит достаточные признаки
≥2/≥3 различных дамэ и распространяет их внутри групп. На их основе задаются
приоритеты захвата/спасения, штрафы самоатари, заполнения глаз и неподдержанного
контакта. Диффузия даёт локальное влияние; pooled признаки задают PASS и
приближённый WDL с фиксированным коми 1,5.

Главные ограничения для первых разрешённых раундов: неполный охват длинных групп,
ложные атари при распределённых дамэ, отсутствие точного чтения после хода,
приближённая проверка ложных глаз, некалиброванный WDL и возможный ранний PASS.
Ни один диагностический раунд в рамках этой задачи не расходуется.
