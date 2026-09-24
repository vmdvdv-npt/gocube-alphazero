# Torus9 MCTS odd/even diagnostic

Дата анализа: 2026-09-24. Read-only анализ сохранённых self-play records M100–M105 и M130–M135. Новые self-play, inference, training, Arena, benchmark и изменение runs/ не выполнялись.

## Краткий вывод

Главный факт: sawtooth возникает уже в root visits, а не при формировании pi или post-search temperature selection.

- historical-M100-M105: odd H=1.664, even H=2.327; BLACK H=1.664, WHITE H=2.327; positions=217592, invalid=0.
- current-M130-M135: odd H=1.608, even H=2.390; BLACK H=1.608, WHITE H=2.390; positions=216759, invalid=0.
- В новом окне odd/even и Black/White совпадают по mapping: odd=BLACK, even=WHITE; sawtooth не является отдельным эффектом, независимым от цвета.
- Новый high-branch even minus low-branch odd delta entropy = 0.782 nats; effective candidates = 10.05.
- Root budget одинаков: sum(root_visits)=200 для всех валидных позиций; illegal non-zero visits и vector-shape violations не обнаружены.
- pi является нормализацией root_visits с max absolute error 0.00e+00. Поэтому sawtooth уже присутствует в visits.
- Наиболее вероятное нормальное объяснение — устойчивое Black/White asymmetry learned policy/value при komi=0.5, усиленная side-to-move representation; code audit не нашёл parity/color-specific search branch.

## 1. Scope, config and sanity

| Window | MCTS simulations | cpuct | FPU | Komi | Root noise | Dirichlet α/ε | Temperature | Replay generations |
|---|---:|---:|---:|---:|---|---|---|---:|
| historical-M100-M105 | 200 | 1.25 | 0.00 | 0.5 | True | 0.11/0.25 | 1.0 on plies 1–8, then 0 | 2 |
| current-M130-M135 | 200 | 1.25 | 0.00 | 0.5 | True | 0.11/0.25 | 1.0 on plies 1–8, then 0 | 6 |

| Window | Games | Positions | Invalid records | PASS actions | Side starts |
|---|---:|---:|---:|---:|---|
| historical-M100-M105 | 2304 | 217592 | 0 | 19753 | {'BLACK': 2304} |
| current-M130-M135 | 2304 | 216759 | 0 | 18339 | {'BLACK': 2304} |

Оба окна прочитаны напрямую. Не пересчитывались SHA многогигабайтных self-play artifacts. Санити-проверки включали state chain, legal selected action, side switch после каждого action включая PASS, vector lengths и terminal trace. Ненулевые counters, кроме ожидаемого temperature-related selected_not_visit_argmax, отсутствуют: historical={}, current={}.

Корректная интерпретация конфигурации: 200 simulations дают общий root budget на position; temperature=1.0 на plies 1–8 используется только для выбора action после search, затем temperature=0.

## 2. Odd/even root MCTS metrics

| Window | Group | N | H mean | H median | H q05–q95 | Effective candidates | Top-1 | Top-2 | Top-3 | Non-zero visits | Top max visits mean / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical-M100-M105 | odd | 109339 | 1.664 | 1.553 | 0.377–3.172 | 7.81 | 49.1% | 64.9% | 73.4% | 16.57 | 98.20 / 200 |
| historical-M100-M105 | even | 108253 | 2.327 | 2.467 | 0.493–4.022 | 16.57 | 36.9% | 50.3% | 58.6% | 34.20 | 73.75 / 200 |
| current-M130-M135 | odd | 108945 | 1.608 | 1.500 | 0.386–3.086 | 7.07 | 49.2% | 65.6% | 74.5% | 14.22 | 98.40 / 200 |
| current-M130-M135 | even | 107814 | 2.390 | 2.583 | 0.502–3.892 | 17.12 | 35.0% | 48.2% | 56.5% | 34.91 | 70.02 / 200 |

## 3. Black/White direct comparison

| Window | Side to move | N | H mean | H median | H q05–q95 | Effective candidates | Top-1 | Top-2 | Top-3 | Non-zero visits | Top max visits mean / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical-M100-M105 | BLACK | 109339 | 1.664 | 1.553 | 0.377–3.172 | 7.81 | 49.1% | 64.9% | 73.4% | 16.57 | 98.20 / 200 |
| historical-M100-M105 | WHITE | 108253 | 2.327 | 2.467 | 0.493–4.022 | 16.57 | 36.9% | 50.3% | 58.6% | 34.20 | 73.75 / 200 |
| current-M130-M135 | BLACK | 108945 | 1.608 | 1.500 | 0.386–3.086 | 7.07 | 49.2% | 65.6% | 74.5% | 14.22 | 98.40 / 200 |
| current-M130-M135 | WHITE | 107814 | 2.390 | 2.583 | 0.502–3.892 | 17.12 | 35.0% | 48.2% | 56.5% | 34.91 | 70.02 / 200 |

В records стартовая сторона всегда BLACK, а side-to-move переключается после каждого action. PASS также переключает сторону. Поэтому для этих данных odd=BLACK и even=WHITE подтверждено state-chain проверкой, а не предположено по номеру ply.

## 4. Persistence across the game

| Window | Interval | Group | N | H mean | Effective candidates | Top-1 | Non-zero visits |
|---|---|---|---:|---:|---:|---:|---:|
| historical-M100-M105 | moves_1_8 | odd | 9176 | 2.467 | 18.18 | 32.6% | 27.92 |
| historical-M100-M105 | moves_1_8 | even | 9143 | 3.885 | 51.76 | 13.3% | 77.98 |
| historical-M100-M105 | moves_1_8 | BLACK | 9176 | 2.467 | 18.18 | 32.6% | 27.92 |
| historical-M100-M105 | moves_1_8 | WHITE | 9143 | 3.885 | 51.76 | 13.3% | 77.98 |
| historical-M100-M105 | moves_9_24 | odd | 18096 | 1.532 | 6.03 | 56.4% | 16.71 |
| historical-M100-M105 | moves_9_24 | even | 18088 | 3.148 | 27.26 | 27.6% | 63.91 |
| historical-M100-M105 | moves_9_24 | BLACK | 18096 | 1.532 | 6.03 | 56.4% | 16.71 |
| historical-M100-M105 | moves_9_24 | WHITE | 18088 | 3.148 | 27.26 | 27.6% | 63.91 |
| historical-M100-M105 | moves_25_64 | odd | 45203 | 1.599 | 6.77 | 53.0% | 19.30 |
| historical-M100-M105 | moves_25_64 | even | 45193 | 2.163 | 11.75 | 41.4% | 31.66 |
| historical-M100-M105 | moves_25_64 | BLACK | 45203 | 1.599 | 6.77 | 53.0% | 19.30 |
| historical-M100-M105 | moves_25_64 | WHITE | 45193 | 2.163 | 11.75 | 41.4% | 31.66 |
| historical-M100-M105 | moves_65_plus | odd | 36864 | 1.607 | 7.39 | 44.8% | 10.33 |
| historical-M100-M105 | moves_65_plus | even | 35829 | 1.721 | 8.27 | 41.8% | 11.24 |
| historical-M100-M105 | moves_65_plus | BLACK | 36864 | 1.607 | 7.39 | 44.8% | 10.33 |
| historical-M100-M105 | moves_65_plus | WHITE | 35829 | 1.721 | 8.27 | 41.8% | 11.24 |
| current-M130-M135 | moves_1_8 | odd | 9175 | 2.017 | 10.07 | 38.0% | 15.77 |
| current-M130-M135 | moves_1_8 | even | 9142 | 3.786 | 46.56 | 13.9% | 77.97 |
| current-M130-M135 | moves_1_8 | BLACK | 9175 | 2.017 | 10.07 | 38.0% | 15.77 |
| current-M130-M135 | moves_1_8 | WHITE | 9142 | 3.786 | 46.56 | 13.9% | 77.97 |
| current-M130-M135 | moves_9_24 | odd | 18065 | 1.343 | 4.55 | 58.8% | 11.96 |
| current-M130-M135 | moves_9_24 | even | 18056 | 3.272 | 29.21 | 23.5% | 65.40 |
| current-M130-M135 | moves_9_24 | BLACK | 18065 | 1.343 | 4.55 | 58.8% | 11.96 |
| current-M130-M135 | moves_9_24 | WHITE | 18056 | 3.272 | 29.21 | 23.5% | 65.40 |
| current-M130-M135 | moves_25_64 | odd | 45032 | 1.599 | 7.02 | 52.3% | 17.88 |
| current-M130-M135 | moves_25_64 | even | 45000 | 2.277 | 13.38 | 38.9% | 32.84 |
| current-M130-M135 | moves_25_64 | BLACK | 45032 | 1.599 | 7.02 | 52.3% | 17.88 |
| current-M130-M135 | moves_25_64 | WHITE | 45000 | 2.277 | 13.38 | 38.9% | 32.84 |
| current-M130-M135 | moves_65_plus | odd | 36673 | 1.647 | 7.62 | 43.4% | 10.45 |
| current-M130-M135 | moves_65_plus | even | 35616 | 1.726 | 8.15 | 41.4% | 11.01 |
| current-M130-M135 | moves_65_plus | BLACK | 36673 | 1.647 | 7.62 | 43.4% | 10.45 |
| current-M130-M135 | moves_65_plus | WHITE | 35616 | 1.726 | 8.15 | 41.4% | 11.01 |

Representative per-ply rows (the original sawtooth example is visible before and after the temperature cutoff):

| Window | Ply | Side to move | N | H mean | Effective candidates | Top-1 |
|---|---:|---|---:|---:|---:|---:|
| historical-M100-M105 | 7 | BLACK | 2279 | 1.791 | 7.59 | 48.4% |
| historical-M100-M105 | 8 | WHITE | 2269 | 3.617 | 41.12 | 19.1% |
| historical-M100-M105 | 9 | BLACK | 2269 | 1.617 | 6.27 | 53.2% |
| historical-M100-M105 | 10 | WHITE | 2261 | 3.399 | 33.69 | 24.4% |
| historical-M100-M105 | 11 | BLACK | 2261 | 1.600 | 6.26 | 53.8% |
| historical-M100-M105 | 12 | WHITE | 2261 | 3.373 | 33.06 | 24.1% |
| current-M130-M135 | 7 | BLACK | 2279 | 1.452 | 4.96 | 54.1% |
| current-M130-M135 | 8 | WHITE | 2268 | 3.572 | 38.02 | 18.5% |
| current-M130-M135 | 9 | BLACK | 2266 | 1.306 | 4.23 | 59.0% |
| current-M130-M135 | 10 | WHITE | 2257 | 3.471 | 34.50 | 20.7% |
| current-M130-M135 | 11 | BLACK | 2257 | 1.332 | 4.37 | 58.3% |
| current-M130-M135 | 12 | WHITE | 2257 | 3.420 | 33.07 | 21.5% |

## 5. Root visit accounting and actual budget

| Window | Group | Sum visits distribution | Legal count mean | Illegal non-zero mean | Illegal non-zero max | Vector violations |
|---|---|---|---:|---:|---:|---:|
| historical-M100-M105 | odd | {'200': 109339} | 40.22 | 0.00 | 0 | 0 |
| historical-M100-M105 | even | {'200': 108253} | 39.67 | 0.00 | 0 | 0 |
| historical-M100-M105 | BLACK | {'200': 109339} | 40.22 | 0.00 | 0 | 0 |
| historical-M100-M105 | WHITE | {'200': 108253} | 39.67 | 0.00 | 0 | 0 |
| current-M130-M135 | odd | {'200': 108945} | 40.46 | 0.00 | 0 | 0 |
| current-M130-M135 | even | {'200': 107814} | 39.94 | 0.00 | 0 | 0 |
| current-M130-M135 | BLACK | {'200': 108945} | 40.46 | 0.00 | 0 | 0 |
| current-M130-M135 | WHITE | {'200': 107814} | 39.94 | 0.00 | 0 | 0 |

Production search code performs one root-edge backup per configured simulation. Thus sum(root_visits)=200 is the relevant invariant, not a per-action 200 count. All analyzed groups satisfy it; no systematic Black/White budget difference exists.

## 6. pi versus root visits

| Window | Group | pi entropy | pi effective candidates | pi top-1 | Max abs(pi − visits/sum) | Illegal pi mass | Selected=visit-argmax |
|---|---|---:|---:|---:|---:|---:|---:|
| historical-M100-M105 | odd | 1.664 | 7.81 | 49.1% | 0.00e+00 | 0.00e+00 | 94.5% |
| historical-M100-M105 | even | 2.327 | 16.57 | 36.9% | 0.00e+00 | 0.00e+00 | 92.7% |
| historical-M100-M105 | BLACK | 1.664 | 7.81 | 49.1% | 0.00e+00 | 0.00e+00 | 94.5% |
| historical-M100-M105 | WHITE | 2.327 | 16.57 | 36.9% | 0.00e+00 | 0.00e+00 | 92.7% |
| current-M130-M135 | odd | 1.608 | 7.07 | 49.2% | 0.00e+00 | 0.00e+00 | 94.9% |
| current-M130-M135 | even | 2.390 | 17.12 | 35.0% | 0.00e+00 | 0.00e+00 | 92.8% |
| current-M130-M135 | BLACK | 1.608 | 7.07 | 49.2% | 0.00e+00 | 0.00e+00 | 94.9% |
| current-M130-M135 | WHITE | 2.390 | 17.12 | 35.0% | 0.00e+00 | 0.00e+00 | 92.8% |

pi не создаёт sawtooth: в текущих artifacts она с точностью сериализации равна нормализованным root visits. В plies 1–8 selected_is_visit_argmax ниже из-за temperature=1.0, а на plies 9+ он равен 100% при temperature=0; это post-search temperature effect, но root-visits entropy того же position он не меняет. Для current окна argmax selection в plies 1–8: odd=39.1%, even=14.8%; в plies 9+ — 100% для обеих групп.

## 7. Legal-action count control

historical-M100-M105: odd vs even common legal-count buckets (>=30 positions each) = 71, conditional entropy delta = -0.594 nats, max bucket delta = 2.172; BLACK vs WHITE conditional delta = -0.594 nats.
current-M130-M135: odd vs even common legal-count buckets (>=30 positions each) = 70, conditional entropy delta = -0.696 nats, max bucket delta = 2.431; BLACK vs WHITE conditional delta = -0.696 nats.

Legal-action count отличается с ходом игры, но не объясняет sawtooth: при одинаковом legal-count bucket Black/White separation сохраняется. В vectors illegal actions имеют zero visits; PASS входит в legal action count.

## 8. Code audit of confirmed path

- `gocube_golden/torus9_monolith.py`: observation uses relative own/opponent stone planes plus one side-to-move color channel (+1 BLACK, -1 WHITE); there are no absolute stone-color planes.
- `gocube_golden/search.py`: search uses side-to-move WDL, utility WIN minus LOSS, and one sign flip per traversed edge. No odd/even or Black/White branch exists in PUCT traversal, backup, or root accounting.
- `gocube_golden/search.py`: root visits are legal edge visits mapped into the 82-action vector; pi is constructed directly as count/sum.
- `gocube_golden/selfplay_policy.py`: Dirichlet is applied to the root evaluator policy before search expansion. Raw NN policy and noisy prior are not stored in self-play records, so the pre-search prior cannot be compared from these artifacts.
- `gocube_golden/selfplay_policy.py`: temperature is applied only after root visits are produced. It explains selected-action determinism after ply 8, not the same-position root-visits sawtooth.
- Komi=0.5 is a real game asymmetry; a Black/White difference alone is not evidence of a bug.

## 9. Historical comparison and answers

| Question | Answer |
|---|---|
| 1. Насколько велик effect? | Current high-branch even minus low-branch odd root entropy delta = 0.782 nats; effective candidates delta = 10.05. It is large and stable across intervals. |
| 2. Эквивалентен ли Black/White? | Да для analyzed records: odd=BLACK, even=WHITE и group metrics совпадают по state-chain mapping. Это color asymmetry, not an independent parity mechanism. |
| 3. Одинаков ли budget? | Да: root visit sum is 200 for all valid positions; no illegal visits. |
| 4. Объясняется ли legal moves? | Нет: separation remains within common exact legal-count buckets. |
| 5. Возникает ли в root visits? | Да. pi лишь нормализует visits; post-search temperature не источник sawtooth. |
| 6. Был ли effect в M100–M105? | Да; historical window contains the same odd/even and Black/White separation. |
| 7. Усилился ли с обучением? | В aggregate-сравнении да: entropy delta вырос с 0.663 до 0.782 nats, effective-candidate delta — с 8.76 до 10.05; двух окон недостаточно для вывода о монотонном тренде. |
| 8. Есть ли evidence implementation bug? | По records и confirmed code path — нет: budget, legality, side switching, pi normalization и value sign semantics consistent. |
| 9. Наиболее вероятное объяснение? | Learned Black/White policy/value asymmetry при komi=0.5, проявляющаяся через side-to-move encoding; raw NN prior не сохранён, поэтому pre-search contribution неразделим. |
| 10. Связь с замедлением обучения? | Прямая связь не доказана. Это может менять effective self-play target distribution по цветам, но diversity/full-prefix analysis не показывает collapse; отдельного learning-causality теста нет. |

### Delta comparison

| Window | High-branch group | Low-branch group | Delta entropy | Delta effective candidates | Delta top-1 visit share |
|---|---|---|---:|---:|---:|
| historical-M100-M105 | even | odd | 0.663 | 8.76 | -12.2% |
| current-M130-M135 | even | odd | 0.782 | 10.05 | -14.2% |

## 10. Limitations

- Records do not serialize raw pre-MCTS NN policy, noisy Dirichlet prior, WDL logits, or root Q, so the exact boundary between network prior and PUCT cannot be identified from existing artifacts alone.
- Exact legal-action reconstruction uses production rules and reads only saved states; it does not run MCTS or neural inference.
- Black/White asymmetry can be legitimate under komi=0.5 and a learned non-color-equivariant network. The diagnostic does not prove that the model asymmetry is desirable.
- No production code or parameter was changed. If a future experiment tests color-equivariant observation/modeling or temperature alternatives, it must be a separate task.
