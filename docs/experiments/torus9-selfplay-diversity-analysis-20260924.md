# Torus9 self-play diversity analysis

Дата анализа: 2026-09-24. Это read-only диагностика сохранённых self-play artifacts; новые self-play, обучение, Arena, A/B и benchmark не запускались, runs/ не изменялись.

## Краткий вывод

- В актуальном окне M130–M135 найдено 2304 строк, 2304 полноценных игр; exact unique — 2302 (99.9%), полных дублей — 2.
- В новом окне 211810 unique raw states из 216759 (97.7%); repeat rate 2.3%.
- Aggregate selected-action entropy почти не меняется после ply 8: 4.386 → 4.378 nats/ply; эта coarse-метрика смешивает разные states.
- MCTS root entropy за тот же переход падает: 2.900 → 2.381 nats, effective candidates 28.28 → 19.03, top-1 26.0% → 39.9%; selected ход совпадает с visit argmax после cutoff в 99.9% случаев.
- Между M100–M105 и M130–M135 unique-game ratio: 99.9% → 99.9%; unique-position ratio: 97.7% → 97.7%.

Ответы на пять вопросов ТЗ:

1. 384 игры достаточно разнообразны как полные траектории? В каждом поколении exact duplicate не найден; для объединённого окна unique ratio приведён ниже. Это не заменяет diversity состояний.
2. Есть ли резкий спад после move 8? Да, в MCTS search distribution есть структурный спад entropy и effective branching непосредственно после cutoff; aggregate entropy выбранных координат почти плоская и маскирует state-conditioned эффект. Причинность температуры не доказана.
3. Diversity хуже с ростом поколения? Монотонного ухудшения в M130–M135 нет: unique-position ratio M130 = 98.1%, M135 = 98.1%.
4. Даст ли больше игр просто больше похожих партий? Для полных траекторий и exact prefixes такого вывода нет: в sampled windows prefix 8/12/16 почти все singleton. Но внутри одинакового state после cutoff MCTS выбирается почти детерминированно; увеличение числа игр может сильнее повторять conditional decisions, если states начнут встречаться чаще.
5. Есть ли основания тестировать изменение температурной схемы? Да, как отдельный контролируемый диагностический эксперимент: есть резкий перелом MCTS branching после ply 8 при почти плоской coarse action entropy. Production параметры не менялись.

## 1. Данные и provenance

Основное production окно: current-M130-M135, поколения M130–M135; lineage /home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-m125-continuous-v2-gen6-20260922-v1.
Старое окно: historical-M100-M105, поколения M100–M105; lineage /home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-m92-continuous-v2-20260920-v1.

| Окно | Найдено строк | Parsed | Полных игр | Invalid | Позиций | Длина median / mean / max |
|---|---:|---:|---:|---:|---:|---:|
| historical-M100-M105 | 2304 | 2304 | 2304 | 0 | 217592 | 89.0 / 94.4 / 208 |
| current-M130-M135 | 2304 | 2304 | 2304 | 0 | 216759 | 91.0 / 94.1 / 195 |

Ожидалось 384 × 6 = 2304 игры в каждом окне; все 12 файлов найдены. Все 12 SHA-256 файлов совпали с selfplay_artifact_sha256 в generation completion records; повреждённых строк/records нет.

### Effective configuration

Новый effective config: /home/codex/projects/gocube-alphazero/runs/torus9/active/torus9-m125-continuous-v2-gen6-20260922-v1/metadata/effective-config-v2/sha256:0aa3a1c4987618683f479f4464a346476961bc16371f23ad071e3b88b014684c.json (sha256 sha256:bdfd8145905a51df0003156b0b499145045deb72332c234539ae598b1f76b4e1). Подтверждено: temperature = 1.0 on plies 1–8, then 0, temperature_after = 0.0, temperature_plies = [1, 8], MCTS simulations = 200, Dirichlet alpha = 0.11, epsilon = 0.25, games = 384. Старый effective config имеет ту же temperature/MCTS/Dirichlet схему.

Сохранены root_visits (216759 records) и policy targets pi (216759 records).

## 2. Exact full-game uniqueness

| Поколение | Games | Exact unique | Full duplicates | Unique ratio |
|---:|---:|---:|---:|---:|
| M100 | 384 | 384 | 0 | 100.0% |
| M101 | 384 | 384 | 0 | 100.0% |
| M102 | 384 | 384 | 0 | 100.0% |
| M103 | 384 | 384 | 0 | 100.0% |
| M104 | 384 | 384 | 0 | 100.0% |
| M105 | 384 | 384 | 0 | 100.0% |
| **historical-M100-M105 together** | **2304** | **2301** | **3** | **99.9%** |
| M130 | 384 | 384 | 0 | 100.0% |
| M131 | 384 | 384 | 0 | 100.0% |
| M132 | 384 | 384 | 0 | 100.0% |
| M133 | 384 | 384 | 0 | 100.0% |
| M134 | 384 | 384 | 0 | 100.0% |
| M135 | 384 | 384 | 0 | 100.0% |
| **current-M130-M135 together** | **2304** | **2302** | **2** | **99.9%** |

Signature = exact ordered final_action_trace, включая PASS; metadata и model hash не входят.

## 3. Prefix diversity

| Prefix plies | New eligible | New unique | New ratio | Largest cluster | Old ratio |
|---:|---:|---:|---:|---:|---:|
| 4 | 2291 | 2283 | 99.7% | 2 | 99.8% |
| 8 | 2268 | 2268 | 100.0% | 1 | 100.0% |
| 10 | 2257 | 2257 | 100.0% | 1 | 100.0% |
| 12 | 2257 | 2257 | 100.0% | 1 | 100.0% |
| 16 | 2257 | 2257 | 100.0% | 1 | 100.0% |
| 20 | 2257 | 2257 | 100.0% | 1 | 100.0% |
| 24 | 2257 | 2257 | 100.0% | 1 | 100.0% |
| 28 | 2256 | 2256 | 100.0% | 1 | 100.0% |
| 32 | 2256 | 2256 | 100.0% | 1 | 100.0% |
| 40 | 2256 | 2256 | 100.0% | 1 | 100.0% |
| 48 | 2256 | 2256 | 100.0% | 1 | 100.0% |
| 56 | 2249 | 2249 | 100.0% | 1 | 100.0% |
| 64 | 2210 | 2210 | 100.0% | 1 | 100.0% |
| 72 | 2147 | 2147 | 100.0% | 1 | 100.0% |
| 80 | 1955 | 1955 | 100.0% | 1 | 100.0% |
| 96 | 856 | 856 | 100.0% | 1 | 100.0% |
| 112 | 380 | 380 | 100.0% | 1 | 100.0% |
| 128 | 185 | 185 | 100.0% | 1 | 100.0% |
| 160 | 31 | 31 | 100.0% | 1 | 100.0% |
| 192 | 1 | 1 | 100.0% | 1 | 100.0% |
| 256 | 0 | 0 | 0.0% | 0 | 0.0% |

Все sampled exact prefixes длиной 8/12/16 уникальны (largest cluster = 1), поэтому этот анализ не обнаруживает концентрации полных линий. Prefix ratio не является entropy одной decision state; observed cutoff effect проявляется в MCTS visits, а не в exact prefix duplication.

## 4. Positions and decision diversity

| Окно | Total positions | Unique raw | Unique ratio | Repeat rate | Repeated states >=2 | Repeated occurrences |
|---|---:|---:|---:|---:|---:|---:|
| historical-M100-M105 | 217592 | 212642 | 97.7% | 2.3% | 458 | 2.5% |
| current-M130-M135 | 216759 | 211810 | 97.7% | 2.3% | 470 | 2.5% |

| Window | 1 occurrence | 2 | 3–5 | >5 |
|---|---:|---:|---:|---:|
| historical-M100-M105 | 99.8% | 0.2% | 0.0% | 0.0% |
| current-M130-M135 | 99.8% | 0.2% | 0.0% | 0.0% |

В новом окне weighted conditional choice entropy для повторных exact states = 3.291 nats; weighted top-1 action share = 13.0%. Повторных exact states мало (2.5% encounters), поэтому это полезная, но маломощная conditional диагностика.

Symmetry-canonicalized uniqueness не приводится: нет production-trusted canonicalizer для полного state, включая superko_history.

## 5. Selected actions and temperature-cutoff intervals

| Interval | Unique actions | Mean action entropy | Action top-1 | MCTS entropy | MCTS top-1 | Effective candidates | Selected=argmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| moves 1 8 | 82 | 4.386 | 1.4% | 2.900 | 26.0% | 28.282 | 26.8% |
| moves 9 12 | 82 | 4.378 | 1.5% | 2.381 | 39.9% | 19.029 | 99.9% |
| moves 13 16 | 81 | 4.377 | 1.5% | 2.349 | 40.3% | 17.795 | 100.0% |
| moves 17 24 | 82 | 4.377 | 1.4% | 2.249 | 42.2% | 15.341 | 100.0% |
| moves 25 plus | 82 | 2.960 | 11.2% | 1.826 | 44.2% | 9.164 | 88.8% |

Граница cutoff (текущий window, отдельные plies):

| Ply | Selected-action entropy | MCTS entropy | MCTS top-1 | Selected=visit argmax |
|---:|---:|---:|---:|---:|
| 7 | 4.387 | 1.452 | 54.1% | 54.4% |
| 8 | 4.388 | 3.572 | 18.5% | 18.8% |
| 9 | 4.386 | 1.306 | 59.0% | 99.6% |
| 10 | 4.374 | 3.471 | 20.7% | 100.0% |

Интервальные selected-action metrics агрегируют разные позиции и являются coarse diagnostics; их почти неизменная entropy не опровергает cutoff effect. MCTS metrics используют сохранённые root_visits и показывают реальное уменьшение search branching после ply 8.

## 6. Prefix families

Family = exact prefix. top 1/5/10% означает top соответствующую долю distinct prefix families, с долей покрытых игр.

| Prefix | Window | Top 1% | Top 5% | Top 10% |
|---:|---|---:|---:|---:|
| 8 | historical-M100-M105 | 1.0% | 5.0% | 10.0% |
| 8 | current-M130-M135 | 1.0% | 5.0% | 10.0% |
| 12 | historical-M100-M105 | 1.0% | 5.0% | 10.0% |
| 12 | current-M130-M135 | 1.0% | 5.0% | 10.0% |
| 16 | historical-M100-M105 | 1.0% | 5.0% | 10.0% |
| 16 | current-M130-M135 | 1.0% | 5.0% | 10.0% |

## 7. Change across generations

| Generation | Unique game | Unique position | Prefix-8 | Prefix-12 | Position repeat |
|---:|---:|---:|---:|---:|---:|
| M100 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M101 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M102 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M103 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M104 | 100.0% | 98.0% | 100.0% | 100.0% | 2.0% |
| M105 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M130 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M131 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M132 | 100.0% | 98.0% | 100.0% | 100.0% | 2.0% |
| M133 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M134 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |
| M135 | 100.0% | 98.1% | 100.0% | 100.0% | 1.9% |

Внутри M130–M135 нет монотонного ухудшения; метрики колеблются. Сравнение со старым окном observational, а не причинный тест.

## 8. Limitations

- Full-game uniqueness не оценивает разнообразие позиций и решений.
- Aggregate action entropy смешивает разные board states; exact-state conditional entropy рассчитана только для повторных states.
- Root visits показывают search branching, но не доказывают причинность temperature: также влияют network strength, Dirichlet noise, MCTS budget и tie-breaking.
- Новые симметрии не добавлялись, отсутствующие visits не восстанавливались, production parameters не менялись.

Компактный JSON содержит lineage, effective config, sample sizes, aggregate/per-range metrics, validation inputs and conclusions. Raw per-position records и полные per-ply series намеренно не хранятся в репозитории.
