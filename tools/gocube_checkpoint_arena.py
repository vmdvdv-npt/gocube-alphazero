#!/usr/bin/env python3
"""Compatibility entrypoint for the color-balanced GoCube checkpoint Arena."""

from tools import gocube_checkpoint_arena_complete as _impl
from tools.gocube_balanced_arena import (
    ArenaWorkerAssignment,
    BalancedArenaSelfPlayAgent,
    arena_worker_assignments,
    install_balanced_checkpoint_arena,
)


install_balanced_checkpoint_arena(_impl)

ARENA_SIMS = _impl.ARENA_SIMS
DEFAULT_SEED = _impl.DEFAULT_SEED
EXPECTED_KOMI = _impl.EXPECTED_KOMI
HELDOUT_SCHEMA_VERSION = _impl.HELDOUT_SCHEMA_VERSION
_checkpoint_path = _impl._checkpoint_path
_coalesced_batched_summary = _impl._coalesced_batched_summary
_copy_search_output = _impl._copy_search_output
_game_from_prefix = _impl._game_from_prefix
_heldout_summary = _impl._heldout_summary
_load_network = _impl._load_network
_load_payload = _impl._load_payload
_non_batched_summary = _impl._non_batched_summary
_play_from_position = _impl._play_from_position
_require_same_contract = _impl._require_same_contract
_resolve_device = _impl._resolve_device
_result_for_model_a = _impl._result_for_model_a
_safe_name = _impl._safe_name
_summarize_endgame = _impl._summarize_endgame
_summarize_outcomes = _impl._summarize_outcomes
_terminal_diagnostics = _impl._terminal_diagnostics
_validate_heldout_suite = _impl._validate_heldout_suite
_wilson_interval = _impl._wilson_interval
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
