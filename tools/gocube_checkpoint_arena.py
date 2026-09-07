#!/usr/bin/env python3
"""Compatibility entrypoint for the complete GoCube checkpoint Arena."""

from tools.gocube_checkpoint_arena_complete import (
    ARENA_SIMS,
    DEFAULT_SEED,
    EXPECTED_KOMI,
    HELDOUT_SCHEMA_VERSION,
    _checkpoint_path,
    _coalesced_batched_summary,
    _copy_search_output,
    _game_from_prefix,
    _heldout_summary,
    _load_network,
    _load_payload,
    _non_batched_summary,
    _play_from_position,
    _require_same_contract,
    _resolve_device,
    _result_for_model_a,
    _safe_name,
    _summarize_endgame,
    _summarize_outcomes,
    _terminal_diagnostics,
    _validate_heldout_suite,
    _wilson_interval,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
