# Frozen replay learner diagnostics

Launch and resume only with the standard operator job, after reading the guide
from the executing committed revision and running `job CONFIG --check`.
This mode measures fitting of existing targets, with no games or search.

The H269/M270 example uses two uninterrupted 2560-update trajectories with
original Adam, batch64, clip8 and the same actual sample draws. LR A=1e-4,
B=2.5e-5. Snapshots are at 0,64,256,640,1280,2560. See the launch guide for
strict fields, output identities, metrics and crash recovery semantics.

The primary full-window evaluation samples historical training positions
uniformly, so longer games receive proportionally more weight. It deliberately
replaces the generation/phase-balanced panel for this question. Fresh D270 and
each old generation have separate fixed panels. Their equal-weight means are
not a reconstruction of the full-window objective. Score MSE uses the original
normalized target; score MAE additionally reports points (scale81.5).

Interpretation rules fixed before seeing the curves:

- Errors decrease then increase: investigate fewer updates/early stopping.
- A worsens while B improves: investigate smaller LR.
- Both worsen immediately: inspect fixed-gradient dot actual Adam step; retain
  optimizer history, do not reset Adam automatically.
- Full-window loss decreases while fresh loss rises: investigate replay conflicts.
- Both improve the objective: revise the non-learning hypothesis; transfer to
  decisions still requires further investigation.

The first curve is deliberately a training-data diagnostic. Select one candidate
change using its evidence. Before a separate short confirmation job, freeze an
exclusion manifest of whole fresh games, its SHA and budget. Use the same filtered
replay for control and candidate starting at H269. Existing H270/L270 cannot
retroactively become unseen-game models. Bootstrap intervals cluster by original
game and describe this diagnostic sample; they do not establish playing strength.
