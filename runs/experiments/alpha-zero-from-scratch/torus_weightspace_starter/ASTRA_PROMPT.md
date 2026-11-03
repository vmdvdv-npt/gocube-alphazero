# Astra: Torus9 direct weight programming

Follow [ASTRA_HANDOFF.md](ASTRA_HANDOFF.md), then
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Implement only the algorithmic
NumPy writer `program(weights)` in `design_weights.py`. Read the actual model and
schema before designing its weights. All learned parameters start at zero.

First deliver an immutable zero-pass version. Do not execute the model or inspect
game positions during zero-pass. No gradients, backpropagation, optimizers,
regression, data fitting, pretrained-weight copying or trained targets. Demo
weights and the unchanged empty writer are fixtures, not your result.

Diagnostic rounds require an explicitly configured nonzero session budget and
separate commands. Use the production referee as the sole source of legality and
scoring. Torus9 has no ordinary Go edges or corners. Record every writer version,
checkpoint, hash and diagnostic report. Never infer strength from structural tests.
