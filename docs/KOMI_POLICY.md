# GoCube komi policy

**Status: authoritative for new GoCube and Torus development.**

This policy supersedes older project-wide statements that GoCube komi must
always equal `0.5`. Historical experiment documents may still record an exact
komi because reproducibility requires preserving the experiment that actually
ran; those local pins do not define the project-wide policy.

## Current baseline

`0.5` is the current default and canonical baseline for development, reference
runs, and ordinary experiments unless an experiment explicitly states
otherwise.

It is **not** yet a scientific claim that `0.5` is the fair komi for a
boundaryless Torus. The fair-komi question is intentionally left open for a
later controlled research stage. New rules/reference code must therefore not
encode `komi == 0.5` as a universal validity condition.

An explicitly selected komi must be finite and must be recorded as part of the
rules/evaluation identity. Checkpoints, manifests, result artifacts, and rules
fingerprints must never silently change komi on load or resume.

## Forbidden legacy value: 7.5

`7.5` is a special case and is forbidden in current/new GoCube runtime paths.
It is historically associated with stale ordinary-Go/legacy defaults in this
repository. If `7.5` appears in runtime configuration, checkpoint metadata,
run manifests, generated experiment settings, or automatic migration, treat
that as likely legacy contamination or a bug.

The required behavior is fail-closed:

1. stop before playing, scoring, training, or evaluating under that value;
2. report that legacy `7.5` was encountered and identify the source when
   possible;
3. contact/escalate to the project owner before continuing;
4. never silently coerce `7.5` to `0.5` and never relabel an old checkpoint as
   `0.5` merely to make it load.

Historical text or archived evidence may mention that an old artifact really
used `7.5`; such text is evidence, not an allowed runtime configuration.

## Implementation rule

The shared policy validator accepts explicit finite komi values except the
forbidden legacy sentinel `7.5`. `0.5` remains the default. New/reference code
should use this validator rather than a literal equality assertion.

A component may still pin an exact value for a **named frozen experiment or
legacy compatibility contract**. Examples include the frozen Cube-4 production
sweep and the B0/B1 experiment. Such a check must be described as an
experiment-specific reproducibility constraint, not as "the only supported
GoCube komi".

The old Japanese-V3 training stack contains historical path-specific `0.5`
pins. They are retained only to reproduce those already-defined contracts and
must not be copied into the new Torus reference/training path.

## Torus fair-komi research

A later project stage will estimate intrinsic first-player advantage and fair
komi on boundaryless Torus topologies. The study should use the independent
Torus referee/Golden path, paired color swaps, controlled search conditions,
and enough games to report uncertainty rather than a single noisy point
estimate. Candidate komi values must be tested as actual game/search
parameters, not only applied post hoc to finished scores, because changing
komi can change search and play.

Until that study is complete, use `0.5` as the baseline without treating it as
scientifically final.
