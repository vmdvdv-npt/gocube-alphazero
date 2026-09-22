# GoCube runtime boundaries

Stage 5 freezes one dependency direction for scientific generation:

```text
Orchestrator
    lifecycle / refs / storage / scheduling
        ↓
Generation Driver
    one generation / resolved config / adapter composition
        ↓
Execution Engines
    SelfPlayEngine / TrainingEngine / Arena execution
        ↓
Scientific Adapters
    topology / rules / observation / network / targets
        ↓
Rules + Search
    canonical Golden semantics / canonical SequentialPUCT
```

`SelfPlayEngine` owns worker processes, cooperative scheduling, shared-memory
slots, central batching, transport, cleanup, failure propagation, and
telemetry. `SequentialPUCT`/`SequentialPUCTSession` own the single PUCT tree
algorithm. Scientific adapters own only topology-specific composition and game
records. `build_cube_training_samples()` owns Cube target construction; it is
not part of the execution engine.

Torus9 and Cube V2 therefore have separate scientific adapters but one
cooperative runner, one root-noise helper, one temperature sampler, one
policy/WDL inference owner, one PUCT implementation, and one process execution
engine. Common modules do not import a concrete topology. Execution settings
(`workers`, contexts, batch cap/wait, device, and process start method) are
not scientific contract fields.
