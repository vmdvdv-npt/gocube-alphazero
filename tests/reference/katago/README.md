# KataGo rule fixtures

The fixtures in `rules_fixtures.json` are inputs, not local golden outputs.
Each case is executed against the pinned KataGo rule oracle and the same move
prefix is applied to the test-only rectangular GoCube topology.  Expected
occupancy, legality, phase, ko, terminal, capture, and score values are read
from the oracle at test time.

The source commit for every fixture is
`f6bc4b19a1686caa2d088b56251e8c11c8be6d51`.  A fixture may cite a KataGo test
source even when its exact board is a small manually-authored input; in that
case `source_test` documents the semantic rule being exercised and KataGo
still computes the expected result. Every fixture also carries explicit
`postconditions`; the runner checks named semantic facts such as phase,
captures, ko-recap blocks, score, no-result, pass-alive coverage, and
board-preservation. Fixture names alone are not accepted as coverage.
