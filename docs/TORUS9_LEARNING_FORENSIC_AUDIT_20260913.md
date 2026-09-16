TORUS 9×9 LEARNING FORENSIC AUDIT

OVERALL:
MULTIPLE CONTRIBUTING CAUSES

M4 DEGRADATION EXPLAINED:
YES

M8≈M1 EXPLAINED:
PARTIALLY

CRITICAL IMPLEMENTATION BUGS:
None found in the frozen run's correctness path. Checkpoint, optimizer, replay, sign, index, PASS, topology, and Arena bookkeeping audits passed.

TRAINING-DYNAMICS PROBLEMS:
Fresh-only one-pass training creates a tightly coupled nonstationary loop. The M3 corpus is unusually short and black-skewed (1962 positions, 46 BLACK / 18 WHITE); after training on it, M4 shows a strong fixed-state WIN bias and self-play becomes 55 BLACK / 9 WHITE. Updates vary from 31 to 117 per generation, and the M4 11-sample final batch receives a full Adam step.

ARENA-ONLY PROBLEMS:
The 22 truncated Arena games are legal and fail-closed. They exhibit interleaved PASS and long capture/repetition behavior, not illegal actions, superko failure, or Arena mis-scoring.

REJECTED HYPOTHESES:
Wrong WDL sign/perspective, PUCT double-negation, policy/PASS index mismatch, stale checkpoints, optimizer reset, replay corruption, topology mismatch, and LR-too-high as the primary cause.

CONFIRMED HYPOTHESES:
(1) the 5×5→9×9 scale transition removed full-board point-policy coverage; (2) fresh-only one-pass data feedback causes generation-to-generation oscillation and value drift; (3) variable update counts and partial-batch Adam steps are stability risks; (4) 64-sim teacher targets are variable on the diagnostic sample.

## Executive conclusion

The exact causal chain is: M2 produces an atypically long corpus (7348 positions), M3 self-play then collapses to short games (1962 positions, 46–18 BLACK/WHITE). M4 is trained only on that short corpus, with 44 full Adam updates and a final 11-sample update. The M4 checkpoint is not a stale or corrupted artifact: its model hash is exactly reproducible from M3 plus iter-04 replay. Its policy on the frozen set remains close to M1, but its WDL prediction moves sharply toward side-to-move WIN; its next self-play corpus is correspondingly short/black-dominant (2763 positions, 55–9), and Arena reports M4 vs M0 = 6/58/0 and M4 vs M1 = 8/24/0.

M8 is not a clean recovery. M5–M8 continue the same fresh-generation feedback loop; M8 has a different value bias and a long/white-dominant corpus (7342 positions, 13–51), so it is behaviorally near M1 in Arena while still materially different in fixed-state WDL. Thus M8≈M1 is explained as oscillatory/self-play distribution feedback plus weak/variable teacher targets, not as seven effective cumulative improvements.

## Ranked findings

| RANK | FINDING | SEVERITY | EVIDENCE | CAUSAL CONFIDENCE |
|---:|---|---|---|---|
| 1 | 4-block Torus9 point-policy receptive field is radius 4 while board diameter is 8 | HIGH | Torus5 diameter=4; Torus9 diameter=8; exact distant point-logit equality for 4 blocks; 4-block source gradient=0 | HIGH for representational bottleneck; MEDIUM for M4 causal share |
| 2 | M3 corpus collapsed to 1962 positions / 46-18 winner skew; M4 learned a strong fixed-state WIN value bias and produced 55-9 black games | HIGH | M3: 1962 positions, 46/18; M4 next corpus: 2763 positions, 55/9; fixed-state M4 mean WDL≈[0.832,0.000,0.167] | HIGH |
| 3 | Fresh-only training couples each next self-play distribution to one noisy generation and forgets/oscillates across generations | HIGH | fresh-only matrix: old-corpus value CE drifts; M4 improves D4 but does not preserve D1/D7 behavior | MEDIUM-HIGH |
| 4 | Variable one-pass updates span 31..117 per iteration; M4 partial batch 11 receives a full Adam step and has 1.20x mean per-batch parameter delta | MEDIUM | phase updates 31..117; M4 last batch=11; exact reproduction ratio=1.20× mean batch parameter delta | MEDIUM as stability contributor |
| 5 | 64-sim policy targets are variable against 256-sim diagnostic searches | MEDIUM | 64→256 diagnostic: M1 mean KL 1.133, M3 0.564 with 0% top-action agreement, M4 0.430 with 50% agreement | MEDIUM |
| 6 | History is not represented beyond current board/legal mask; future superko constraints can differ for identical observations | MEDIUM | same parent observation/legal mask, different child legal mask after history-only superko state | LOW for this run |
| 7 | WDL-only corpus contains two labels and no DRAW targets | MEDIUM | all eight corpora have only WIN/LOSS targets; DRAW count=0 | LOW-MEDIUM |
| 8 | 22 Arena truncations are legal, non-consecutive-pass long games with captures/repeating pass boards | MEDIUM | 9 M8/M1 + 13 M8/M7 technical games; every replay legal; pass/capture/unique-board table below | HIGH for Arena symptom; not root of training |

## Updated hypothesis ranking

The ranking below separates architecture/scale transition, training dynamics, teacher quality, and runtime/search. `OLD-DATA LOSS CHANGE` and `NEW-DATA LOSS CHANGE` below use total CE; positive means the child model is worse than the comparison model.

| RANK | CLASS | HYPOTHESIS | PLAUSIBILITY BEFORE TEST | EVIDENCE | TEST | RESULT | CAUSAL CONFIDENCE |
|---:|---|---|---|---|---|---|---|
| 1 | ARCHITECTURE / SCALE TRANSITION | 4-block receptive-field limitation on Torus9 | HIGH | Torus5 diameter=4, Torus9 diameter=8; distant point-logit equality and zero 4-block source gradient | Graph-distance proof, multi-point perturbations on M0, autograd, and contradictory global-dependency fixture | CONFIRMED REPRESENTATIONAL BOTTLENECK; causal contribution to M4 supported but not quantified | HIGH representation / MEDIUM M4 share |
| 2 | ARCHITECTURE / SCALE TRANSITION | Insufficient global context in point-policy | HIGH | Global mean feeds value/PASS but not point logits | Current 4-block versus diagnostic 8-block dependency probe on the same distant-marker pair | 4-block contradictory point ordering is impossible; 8-block path exists in diagnostic initialization | MEDIUM |
| 3 | ARCHITECTURE / SCALE TRANSITION | Observation/history information loss | MEDIUM | Current observation omits full superko history | Same parent observation/legal mask with different history; apply same move and compare child masks | INFORMATION LOSS CONFIRMED; no run-specific causal link demonstrated | LOW for this run |
| 4 | TRAINING DYNAMICS | Fresh-only replay / catastrophic forgetting | HIGH | Cross-generation matrix shows old-corpus WDL drift while each phase trains only its fresh corpus | Full M1..M8 × D1..D8 policy/WDL/total CE matrix | PARTIALLY CONFIRMED AS VALUE/DYNAMICS DRIFT, not pure policy collapse | MEDIUM-HIGH |
| 5 | TRAINING DYNAMICS | Variable optimizer updates per generation | MEDIUM | One pass yields 31..117 Adam updates for 1962..7470 samples | Lineage, batch statistics, exact phase reproduction | CONFIRMED STABILITY RISK; not independently proven as sole cause | MEDIUM |
| 6 | TRAINING DYNAMICS | LR too high | MEDIUM | LR is 0.001, inherited from successful Stage4 | Parameter deltas, gradients, layer maxima, exact transition reproduction | REJECTED AS PRIMARY; M4 is not a gradient/parameter outlier | HIGH for rejection |
| 7 | TRAINING DYNAMICS | Gradient spikes | MEDIUM | Possible because corpus sizes and target distributions vary | Per-batch gradient distribution and cross-iteration comparison | M3 has the largest spike; M4 is not an outlier | HIGH for M4-specific rejection |
| 8 | TRAINING DYNAMICS | Partial batches receive destabilizing updates | MEDIUM | Final batches range from 11 to 52 samples and still receive full Adam steps | Per-batch parameter deltas with exact M3→M4 reproduction | CONFIRMED AS STABILITY CONTRIBUTOR; M4 final step is 1.20× mean delta | MEDIUM |
| 9 | TEACHER QUALITY | 64-sim policy target quality | MEDIUM | Fixed sample shows variable 64→256 visit distributions | Two representative positions per generating model at 64 and 256 simulations | WEAK/VARIABLE DIAGNOSTIC TEACHER; not a search correctness failure | MEDIUM |
| 10 | TEACHER QUALITY | WDL-only sparse supervision | MEDIUM | 40910 targets contain only WIN/LOSS and no DRAW; phase calibration differs | Offline target diversity and early/mid/late WDL CE analysis | PLAUSIBLE SAMPLE-EFFICIENCY BOTTLENECK, not proven root cause | LOW-MEDIUM |
| 11 | RUNTIME / SEARCH | PASS/endgame pathology | MEDIUM | 22 Arena traces hit 500 actions with passes or long capture cycles | Independent legal replay, pass adjacency, captures, unique boards, final occupancy | MODEL/SEARCH SYMPTOM; no rules/superko deadlock found | HIGH for symptom classification |
| 12 | RUNTIME / SEARCH | Arena-only defect | LOW | Arena is a detector and technical games are fail-closed | Replay all technical records and compare stored W/L/D bookkeeping | REJECTED; Arena exposes the symptom but does not create the training failure | HIGH for rejection |

## Transition and optimizer audit

| TRANSITION | UPDATES | SAMPLES | PARAM DELTA | REL DELTA | ADAM STEP | MAX LAYER DELTA |
|---|---:|---:|---:|---:|---:|---:|
| M0->M1 | 83 | 5265 | 3.207700 | 0.001923 | 83 | 1.235824 (value_head.0.weight) |
| M1->M2 | 198 | 7348 | 2.177852 | 0.001306 | 198 | 0.862749 (value_head.0.weight) |
| M2->M3 | 229 | 1962 | 1.431581 | 0.000858 | 229 | 0.451474 (blocks.2.message.neighbor_linear.weight) |
| M3->M4 | 273 | 2763 | 2.038180 | 0.001222 | 273 | 0.773215 (blocks.3.output.weight) |
| M4->M5 | 352 | 5033 | 2.317089 | 0.001389 | 352 | 0.749761 (blocks.1.message.neighbor_linear.weight) |
| M5->M6 | 469 | 7470 | 2.236140 | 0.001340 | 469 | 0.769354 (blocks.1.message.neighbor_linear.weight) |
| M6->M7 | 528 | 3727 | 2.486080 | 0.001490 | 528 | 0.850969 (blocks.1.message.neighbor_linear.weight) |
| M7->M8 | 643 | 7342 | 1.808980 | 0.001084 | 643 | 0.626807 (blocks.3.output.weight) |

Lineage status: `PASS`. Exact training reproduction status: `PASS`.

## Batch-level audit

All recorded per-update metrics were matched by replaying the training loop. The relevant M3/M4/M5 summary is:

| ITER | POSITIONS | UPDATES | LAST BATCH | GRAD MIN/MAX/MEAN/P95 | TOTAL LOSS MIN/MAX/MEAN/P95 |
|---:|---:|---:|---:|---|---|
| M1 | 5265 | 83 | 17 | 0.2005/2.4325/0.8182/1.9627 | 3.8719/5.4766/4.2325/4.8911 |
| M2 | 7348 | 115 | 52 | 0.1539/2.4017/0.7906/1.8414 | 3.6219/4.2820/3.8965/4.1060 |
| M3 | 1962 | 31 | 42 | 0.2992/3.1156/0.7769/1.4119 | 4.5815/4.8839/4.7758/4.8712 |
| M4 | 2763 | 44 | 11 | 0.3408/1.8631/0.8950/1.4959 | 3.6397/4.3584/4.0144/4.2646 |
| M5 | 5033 | 79 | 41 | 0.2655/2.5511/0.9311/2.1069 | 3.9511/4.5876/4.1969/4.4330 |
| M6 | 7470 | 117 | 46 | 0.3412/2.8236/0.9395/1.8398 | 3.4277/4.0714/3.7210/3.9826 |
| M7 | 3727 | 59 | 15 | 0.1999/2.2203/0.8751/1.9366 | 4.0118/4.7227/4.2869/4.6523 |
| M8 | 7342 | 115 | 46 | 0.3397/2.1642/0.9303/1.7729 | 3.3172/3.9953/3.6175/3.8465 |

M3 has the largest recorded gradient spike (3.1156, on its 42-sample last batch), but M4 is not a gradient outlier (max 1.8631). M4's 11-sample final batch has gradient 1.4959 and parameter delta 0.098749, which is 1.20× its mean per-batch delta. This supports a stability risk, not an M4-specific explosive jump.

## Independent replay target audit

`40910` rows were independently recomputed from raw traces with `0` mismatches. The independent check covered WDL side perspective, state-before-action, legal mask, superko legality, observation channels, PASS=81, and π=root_visits/sum.

## Corpus distribution and causal M3→M4 evidence

| ITER | POSITIONS | AVG/MED/MIN/MAX PLY | WINNERS | PASS FREQ | OCCUPIED MEAN | MARGIN MEAN |
|---:|---:|---|---|---:|---:|---:|
| M1 | 5265 | 82.27/92.5/7/194 | {'BLACK': 26, 'WHITE': 38} | 0.0712 | 62.86 | 0.609 |
| M2 | 7348 | 114.81/103.0/5/324 | {'BLACK': 34, 'WHITE': 30} | 0.1436 | 74.50 | 1.266 |
| M3 | 1962 | 30.66/29.0/6/89 | {'BLACK': 46, 'WHITE': 18} | 0.1606 | 25.58 | 0.672 |
| M4 | 2763 | 43.17/21.0/3/176 | {'BLACK': 55, 'WHITE': 9} | 0.1542 | 30.09 | 1.500 |
| M5 | 5033 | 78.64/73.5/3/276 | {'BLACK': 22, 'WHITE': 42} | 0.1494 | 57.86 | 2.594 |
| M6 | 7470 | 116.72/106.5/87/201 | {'BLACK': 35, 'WHITE': 29} | 0.1387 | 74.30 | -0.188 |
| M7 | 3727 | 58.23/56.5/6/87 | {'WHITE': 57, 'BLACK': 7} | 0.0875 | 47.48 | -1.625 |
| M8 | 7342 | 114.72/112.0/50/211 | {'WHITE': 51, 'BLACK': 13} | 0.1347 | 70.25 | -3.000 |

The distribution is not ordinary stationary noise: M2→M3 changes 7348→1962 positions and 34/30→46/18 winners; M3→M4 changes to 2763 positions and 55/9 winners. Because every next generation is trained only on the immediately preceding generation's fresh data, these changes feed directly into the next model and back into self-play.

## Iteration budget and strength correlation

This table keeps games, samples, and optimizer updates separate. The loss changes use total CE: old-data change compares child versus parent on the previous corpus; new-data change compares child versus parent on the current corpus. Positive means the child is worse.

| ITER | POSITIONS | UPDATES | AVG PLY | PARAMETER DELTA | MEAN GRAD NORM | OLD-DATA LOSS CHANGE | NEW-DATA LOSS CHANGE | ARENA STRENGTH |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | 5265 | 83 | 82.27 | 3.207700 | 0.8182 | — | -1.3806 | — |
| M2 | 7348 | 115 | 114.81 | 2.177852 | 0.7906 | -0.0306 | -0.0771 | — |
| M3 | 1962 | 31 | 30.66 | 1.431581 | 0.7769 | +0.1446 | -0.1082 | — |
| M4 | 2763 | 44 | 43.17 | 2.038180 | 0.8950 | +0.0597 | -0.1424 | M4-vs-M0 [6, 58, 0]; M4-vs-M1 [8, 24, 0] |
| M5 | 5033 | 79 | 78.64 | 2.317089 | 0.9311 | +0.3431 | -0.3213 | — |
| M6 | 7470 | 117 | 116.72 | 2.236140 | 0.9395 | +0.1149 | -0.1493 | — |
| M7 | 3727 | 59 | 58.23 | 2.486080 | 0.8751 | +0.5509 | -0.4860 | — |
| M8 | 7342 | 115 | 114.72 | 1.808980 | 0.9303 | +0.0568 | -0.1263 | M8-vs-M0 [75, 53, 0]; M8-vs-M1 [62, 57, 0] |

The frozen evidence does not support replacing the declared budget with a guessed fixed update count: strength does not monotonically track positions or updates (for example M4 is weak after 44 updates, while M8 is near M1 after 115). The actionable finding is that the current budget is implicitly `ceil(samples/64)` and therefore varies with self-play distribution.

## Fixed-state model behavior

On 64 identical frozen states, M4's legal policy is very close to M1 (mean symmetric KL≈0.0009), while its WDL L1 drift is 0.8713. M4's mean WDL is approximately `[0.832, 0.000, 0.167]`; M1 is `[0.397, 0.003, 0.601]`. M8's policy remains close to M1 (KL≈0.0022) but its WDL is still materially different, so policy similarity does not imply value/strength similarity.

## Replay forgetting matrix

Each cell is `policy CE / WDL CE / total CE` for MODEL on DATA. The full machine-readable matrix is in the JSON artifact.

| MODEL\DATA | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |
|---|---|---|---|---|---|---|---|---|
| M1 | 3.409/0.683/4.093 | 3.239/0.693/3.933 | 4.155/0.769/4.924 | 3.574/0.746/4.319 | 3.619/0.665/4.284 | 3.239/0.696/3.935 | 3.920/0.612/4.532 | 3.291/0.661/3.952 |
| M2 | 3.407/0.655/4.062 | 3.228/0.627/3.855 | 4.158/0.686/4.845 | 3.557/0.632/4.190 | 3.613/0.687/4.300 | 3.222/0.646/3.868 | 3.928/0.722/4.650 | 3.293/0.685/3.978 |
| M3 | 3.484/0.712/4.197 | 3.332/0.668/4.000 | 4.118/0.619/4.737 | 3.506/0.599/4.105 | 3.603/0.767/4.370 | 3.224/0.683/3.907 | 3.899/0.885/4.784 | 3.302/0.770/4.072 |
| M4 | 3.433/0.799/4.232 | 3.251/0.688/3.940 | 4.140/0.656/4.796 | 3.435/0.528/3.962 | 3.567/0.915/4.482 | 3.151/0.692/3.844 | 3.898/1.187/5.085 | 3.246/0.887/4.132 |
| M5 | 3.464/0.676/4.139 | 3.287/0.680/3.967 | 4.141/0.948/5.089 | 3.467/0.839/4.305 | 3.549/0.612/4.161 | 3.127/0.706/3.833 | 3.872/0.475/4.347 | 3.223/0.582/3.805 |
| M6 | 3.467/0.648/4.114 | 3.281/0.618/3.898 | 4.154/0.669/4.822 | 3.457/0.610/4.067 | 3.557/0.718/4.276 | 3.082/0.602/3.684 | 3.853/0.812/4.666 | 3.154/0.765/3.919 |
| M7 | 3.561/0.921/4.482 | 3.411/0.931/4.341 | 4.149/1.415/5.564 | 3.530/1.236/4.767 | 3.607/0.761/4.368 | 3.191/1.044/4.235 | 3.817/0.363/4.180 | 3.192/0.530/3.722 |
| M8 | 3.530/0.775/4.305 | 3.349/0.777/4.127 | 4.178/1.058/5.236 | 3.513/0.927/4.440 | 3.601/0.669/4.270 | 3.131/0.800/3.931 | 3.838/0.399/4.236 | 3.121/0.475/3.595 |

Verdict: catastrophic forgetting is not a pure policy collapse; policy CE is comparatively stable. There is, however, clear value drift on old corpora (notably M4 on D7), so fresh-only training is a confirmed dynamics risk rather than a proven standalone correctness bug.

## TORUS5 → TORUS9 SCALE TRANSITION AUDIT

5×5 FULL-BOARD RECEPTIVE FIELD: YES
9×9 FULL-BOARD RECEPTIVE FIELD: NO
POINT POLICY HAS GLOBAL CONTEXT: PARTIAL
64-SIM MCTS COMPENSATES: PARTIAL
CATASTROPHIC FORGETTING: INCONCLUSIVE
HISTORY INFORMATION BOTTLENECK: CONFIRMED
POLICY TARGET QUALITY: WEAK
WDL-ONLY BOTTLENECK: INCONCLUSIVE
MOST IMPORTANT DIFFERENCE FROM SUCCESSFUL TORUS5:
The same four blocks covered the 5×5 diameter but leave the 9×9 point-policy head local; this is the key board-size-specific regression.
MOST IMPORTANT DIFFERENCE FROM KATAGO-LIKE TRAINING:
The frozen line trains one fresh generation for one pass, so samples and optimizer updates vary with self-play distribution instead of coming from a controlled multi-generation window.
RECOMMENDED MINIMAL FIX:
Run a small 4-block versus 8-block/global-context supervised fixture, then a fixed-samples/updates replay-window diagnostic; keep both separate from the immutable run.
WHAT MUST NOT BE CHANGED YET:
Do not change LR, clipping, move limit, Arena size, rules, WDL sign, PASS index, or checkpoint artifacts based on this audit alone.

### Receptive-field proof

With four message-passing blocks, a point node can receive information from graph distance at most four. The 5×5 torus diameter is four, so all points can reach one another. The 9×9 torus diameter is eight, so distant changes are exactly invisible to the 9×9 point head. On actual M0, a point-3 perturbation leaves point-40 and point-41 logits exactly unchanged; the autograd source gradient is zero for 4 blocks and nonzero for an 8-block diagnostic initialization. This is a real scale-transition regression in representational coverage.

A controlled contradictory policy task (same local radius-4 neighborhoods, target point 40 for one state and 41 for the distant-marker state) is impossible for the four-block point head for any parameters: both candidate logits are equal across the pair. This meets the addendum's causal threshold for a representational bottleneck. It does not by itself prove that this bottleneck is the largest contributor to M4's Arena score.

### History information

Two synthetic states have identical current board, side-to-move, previous-pass, komi, and legal mask, but different superko histories. After the same first move, the child legal masks diverge. The network cannot distinguish the parent states; MCTS still carries the full history. This is an information-loss finding, not a demonstrated M4 root cause.

### Search target quality

The machine report repeats 2 positions per model at 64 and 256 simulations. Stored 64-sim targets reproduce exactly, but KL and top-action agreement vary substantially; M3 has 0% top-action agreement on its two sampled positions, M1 mean KL≈1.133, and M4 mean KL≈0.430. This supports a weak/variable teacher signal hypothesis, not a search correctness bug.

### Controlled MCTS compensation

For distant-marker pairs with unchanged target-local neighborhoods, the audit records raw point policy, raw WDL, 64-simulation visits, and the selected action. Complete per-state values are in the JSON artifact; the compact diagnostic is:
d5 source 3→target 40: raw-logit Δ=0; state0 policy={'40': 0.012138443998992443, '41': 0.012138443998992443}, WDL=[0.34351566433906555, 0.2870558798313141, 0.36942845582962036], visits(target/alt/PASS)={'40': 0, '41': 0}/2, selected=0; state1 policy={'40': 0.012151197530329227, '41': 0.012151197530329227}, WDL=[0.34333527088165283, 0.28728049993515015, 0.369384229183197], visits(target/alt/PASS)={'40': 2, '41': 2}/2, selected=7 | d8 source 0→target 40: raw-logit Δ=0; state0 policy={'40': 0.012138443998992443, '41': 0.012138443998992443}, WDL=[0.34351566433906555, 0.2870558798313141, 0.36942845582962036], visits(target/alt/PASS)={'40': 0, '41': 0}/2, selected=0; state1 policy={'40': 0.012151195667684078, '41': 0.012151195667684078}, WDL=[0.34333527088165283, 0.28728049993515015, 0.369384229183197], visits(target/alt/PASS)={'40': 2, '41': 2}/2, selected=4 | d8 source 40→target 0: raw-logit Δ=0; state0 policy={'0': 0.012138443998992443, '1': 0.012138443998992443}, WDL=[0.34351566433906555, 0.2870558798313141, 0.36942845582962036], visits(target/alt/PASS)={'0': 2, '1': 2}/2, selected=0; state1 policy={'0': 0.012151197530329227, '1': 0.012151197530329227}, WDL=[0.34333527088165283, 0.28728049993515015, 0.369384229183197], visits(target/alt/PASS)={'0': 2, '1': 2}/2, selected=0
Verdict: 64-sim MCTS is not promoted to a guaranteed global-policy repair. These fixed M0 probes are a controlled compensation check, but they do not define ground-truth optimal actions without a separate solver.

## Truncated Arena games

All `22` technical records replay with zero legality errors. They are not software deadlocks: PASS actions are never consecutive (otherwise the rules would terminate), and point moves continue legally under superko. The M8-vs-M1 group has high interleaved PASS frequency; M8-vs-M7 has mostly long capture/placement cycles. Classification: model/search PASS and long-game pathology (B), not rules/superko/Arena bug (C/D).

| COMPARISON | GAME | PASSES | LAST-100 PASS | CAPTURES | LAST-100 CAPTURES | UNIQUE BOARDS | FINAL OCCUPIED | CLASS |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M8-vs-M1 | `M8-vs-M1--prefix-02-accepted-06--g1` | 116 | 32 | 332 | 80 | 385 | 54 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-04-accepted-04--g2` | 150 | 40 | 315 | 90 | 351 | 39 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-04-accepted-05--g2` | 161 | 38 | 305 | 90 | 340 | 38 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-06-accepted-01--g2` | 189 | 47 | 258 | 40 | 312 | 59 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-08-accepted-02--g1` | 137 | 27 | 313 | 76 | 364 | 58 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-08-accepted-03--g2` | 164 | 42 | 276 | 50 | 337 | 68 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-12-accepted-05--g2` | 162 | 44 | 282 | 50 | 339 | 68 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-14-accepted-02--g1` | 62 | 15 | 415 | 110 | 439 | 37 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M1 | `M8-vs-M1--prefix-16-accepted-00--g2` | 175 | 33 | 285 | 62 | 326 | 56 | MODEL_PASS_POLICY_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-01--g2` | 10 | 0 | 440 | 99 | 491 | 52 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-02--g1` | 83 | 35 | 363 | 63 | 418 | 56 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-02--g2` | 8 | 2 | 446 | 98 | 493 | 48 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-03--g1` | 66 | 22 | 378 | 84 | 435 | 58 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-04--g1` | 42 | 33 | 416 | 91 | 459 | 44 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-04--g2` | 15 | 2 | 432 | 97 | 486 | 55 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-05--g2` | 13 | 1 | 436 | 98 | 488 | 53 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-06--g2` | 29 | 9 | 412 | 90 | 472 | 61 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-02-accepted-07--g1` | 42 | 31 | 402 | 73 | 459 | 58 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-04-accepted-03--g1` | 47 | 22 | 402 | 78 | 454 | 55 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-04-accepted-04--g2` | 19 | 1 | 427 | 98 | 482 | 58 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-04-accepted-07--g1` | 17 | 6 | 421 | 88 | 484 | 66 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |
| M8-vs-M7 | `M8-vs-M7--prefix-04-accepted-07--g2` | 19 | 2 | 429 | 95 | 482 | 56 | MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY |

## KataGo differential benchmark

| FEATURE | KATAGO APPROACH | OUR APPROACH | WHY DIFFERENT | WAS DIFFERENCE PRESENT ON 5×5? | COULD BECOME MATERIAL ON 9×9? | EVIDENCE | RISK / CLASS |
|---|---|---|---|---|---|---|---|
| Replay window | Shuffled multi-generation window with independent data consumption | Fresh corpus only, one pass | Simplicity/contract isolation versus retaining recent experience | Stage4 tolerated the smaller regime | Yes: generation feedback and value drift | Full M1..M8 × D1..D8 matrix | High / STABILITY + SAMPLE EFFICIENCY |
| Training budget | Explicit sample/update budget separate from games | Updates equal `ceil(positions/64)` | Corpus length controls optimizer exposure | Less material when corpus scale is narrower | Yes: 31..117 updates across 9×9 phases | Per-phase samples, updates, deltas, losses | Medium / STABILITY |
| Global context | Global features and pathways support board-wide decisions | Global mean only for value/PASS; point policy local | Point logits cannot use pooled feature | Four blocks cover 5×5 diameter | Yes: diameter doubles from 4 to 8 | Exact distant-logit equality and gradient probe | High / ARCHITECTURE |
| History | Richer temporal/history features | Current stones + legal mask; full history only in rules/search | Network input is a partial view of rule state | Less material on 5×5, but still absent | Yes when superko future legality matters | Identical parent observations, divergent child masks | Medium / SAMPLE EFFICIENCY |
| Targets | Rich policy/value/ownership/score auxiliary supervision | Policy + WDL only; no DRAW in corpus | Less dense calibration/supervision | Proven adequate for Stage4, not necessarily scalable | Plausible, not demonstrated as root | 40910 rows, two WDL classes, phase CE | Medium / SAMPLE EFFICIENCY |
| Search target generation | Expensive/weighted target searches separated from game generation | 64 sims for essentially every position | Cheap uniform teacher may be noisy | Worked as Golden baseline | Possibly: long-range weakness raises teacher variance | 64→256 fixed-sample KL/agreement | Medium / SAMPLE EFFICIENCY |
| Arena/gating | Gatekeeper, not the training loop | Frozen detector, technical fail-closed | Diagnostic selection versus learning signal | Correct in baseline | No: not a training fix | W/L/D and technical replay audit | Low / CORRECTNESS REJECTED |
| Long-game handling | Larger limits/pathology controls can be used operationally | 500-action watchdog | Runtime cap is a symptom detector, not learning signal | No baseline technical games reported | Yes as a symptom on 9×9 | 22 legal truncated traces, no pass adjacency | Medium / RUNTIME SYMPTOM |

The important difference from successful Torus5 is not Adam or LR: it is that the same four blocks changed functional meaning when diameter grew from 4 to 8, while the fresh one-pass loop amplified generation-distribution changes. The important KataGo-like difference is explicit multi-generation data/update control, which is a stability/sample-efficiency enhancement, not a correctness requirement.

## Arena evidence

| COMPARISON | W/L/D | VALID PAIRS | TECHNICAL | MEAN PAIR SCORE |
|---|---|---:|---:|---:|
| M4-vs-M0 | [6, 58, 0] | 32 | 0 | 0.09375 |
| M4-vs-M1 | [8, 24, 0] | 16 | 0 | 0.25 |
| M8-vs-M0 | [75, 53, 0] | 64 | 0 | 0.5859375 |
| M8-vs-M1 | [62, 57, 0] | 55 | 9 | 0.5181818181818182 |
| M8-vs-M4 | [80, 48, 0] | 64 | 0 | 0.625 |
| M8-vs-M7 | [10, 9, 0] | 6 | 13 | 0.5833333333333334 |

## Minimal recommendation

Do not merge PR #85 or start another full run yet. The minimum next diagnostic is a small controlled comparison of (A) four-block, (B) eight-block, and (optionally) a point-policy global-context variant on a held-out/global-dependency supervised fixture, followed by a deliberately fixed samples/updates multi-generation replay-window experiment. Keep the exact Golden rules/search/target contracts unchanged until those diagnostics are complete.

RECOMMENDED MINIMAL FIX:
For a future Torus9 experiment, first remove the representational scale mismatch with the smallest proven global-context/deeper-policy change, and separately fix the training budget/replay schedule so samples/updates are declared explicitly. Treat both as new experiment branches; do not rewrite the immutable run.

WHAT MUST NOT BE CHANGED YET:
Do not change LR, gradient clipping, move_limit, Arena size, rules, WDL sign, PASS index, or checkpoint artifacts based on this audit alone. Do not call the result stochasticity-only.

## Reproducibility and verification

Run: `/home/codex/projects/gocube-alphazero/runs/torus9/archive/torus9-golden-learning-proof-20260913-v3`; source commit: `9723bb5ac8eb28d55d21d607e3673da5bd894315`; PR final head context: `68ba2de3a67bc65e40cae73a293ea2454ef8e111`.
Independent replay: `PASS`; exact transition reproduction: `PASS`; checkpoint manifest match: `True`; deterministic inference/tiny overfit: `True` / `True`.

Generated by `tools/torus9_learning_forensic_audit.py`; no historical run file is modified.
