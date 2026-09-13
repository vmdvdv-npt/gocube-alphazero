"""Resolve and smoke-test the canonical Torus 5x5 Golden Standard.

This command prepares the immutable run manifest and performs a cheap model
and one-step training boundary check. It intentionally does not start a
multi-hour training or Arena job; a future training runner should consume the
emitted resolved configuration rather than reintroduce inline defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from gocube_golden.standard import (
    CURRENT_ALIAS,
    CURRENT_PRESET_ID,
    build_torus5_model,
    resolve_torus5_golden,
    write_run_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Run manifest path (default: runs/torus5-golden/<run-name>/run-manifest.json).",
    )
    parser.add_argument("--preset", default="current")
    parser.add_argument("--allow-legacy-config", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the one-forward-pass model smoke check (no training).",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest or Path("runs") / "torus5-golden" / args.run_name / "run-manifest.json"
    resolved = resolve_torus5_golden(
        args.preset,
        allow_legacy_config=bool(args.allow_legacy_config),
    )
    effective_argv = list(sys.argv if argv is None else argv)
    metadata = write_run_manifest(
        manifest_path,
        run_name=args.run_name,
        preset=args.preset,
        allow_legacy_config=bool(args.allow_legacy_config),
        argv=effective_argv,
    )

    report: dict[str, object] = {
        "status": "RESOLVED",
        "manifest": str(manifest_path),
        "requested_preset": resolved.requested_preset,
        "resolved_preset_id": resolved.preset_id,
        "resolved_config_sha256": resolved.fingerprint,
    }
    if not resolved.is_legacy:
        report["current_alias"] = CURRENT_ALIAS
        report["concrete_version"] = CURRENT_PRESET_ID
    if args.smoke:
        model = build_torus5_model(
            args.preset,
            allow_legacy_config=bool(args.allow_legacy_config),
        )
        import torch

        from gocube_golden import build_observation, initial_state

        with torch.inference_mode():
            policy, value = model(build_observation(initial_state()).unsqueeze(0))
        model.train()
        observations = build_observation(initial_state()).unsqueeze(0).repeat(4, 1, 1)
        target_policy = torch.full((4, 26), 1.0 / 26.0)
        target_value = torch.full((4, 3), 1.0 / 3.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        policy_logits, value_logits = model(observations)
        loss = (
            -(target_policy * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            -(target_value * torch.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Torus 5x5 training smoke produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        report["smoke"] = {
            "passed": True,
            "architecture": model.architecture_config,
            "policy_shape": list(policy.shape),
            "value_shape": list(value.shape),
            "training_step": 1,
            "training_batch_size": 4,
            "training_loss": float(loss.detach()),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
