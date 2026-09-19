"""Thin Orchestrator V2 boundary for the existing production Arena engine.

Checkpoint discovery and ancestry belong to ``ArtifactResolver``.  This
runner accepts resolver-produced checkpoint nodes and only coordinates
identity, canonical evaluation storage, and the already-qualified Arena
engine.  It intentionally has no checkpoint lookup or copying logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Callable, Mapping

from ..arena_identity import (
    evaluation_fingerprint,
    evaluation_id,
    load_reusable_evaluation,
    stamp_evaluation_identity_metadata,
    write_evaluation_identity,
)
from ..provenance import sha256_fingerprint
from ..run_storage import evaluation_dir
from .artifact_resolver import ResolvedCheckpointNode
from .contracts import (
    ArtifactRef,
    EvaluationIdentity,
    StartsetRef,
)

from tools.arena import run_arena as production_arena
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE


ARENA_RESULT_PROVENANCE_SCHEMA = "gocube-arena-evaluation-provenance-v2"


def torus9_startset_ref(*, master_seed: int, games: int) -> StartsetRef:
    """Describe the deterministic paired Torus9 start corpus used by the engine.

    The production engine deterministically regenerates this corpus from the
    seed and pair count.  The artifact reference is an identity descriptor,
    not a checkpoint or a file to be copied into the run.
    """
    if games <= 0 or games % 2:
        raise ValueError("Arena games must be a positive even number")
    descriptor = {
        "generator": "torus9-evaluation-starts-v2",
        "master_seed": int(master_seed),
        "pairs": int(games) // 2,
        "schema": "golden-evaluation-startset-v2",
    }
    fingerprint = sha256_fingerprint(descriptor)
    return StartsetRef(
        id="torus9-evaluation-starts-v2",
        artifact=ArtifactRef("startsets/torus9-evaluation-starts-v2.json", fingerprint),
        fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class ArenaRunRequest:
    """Inputs already selected and verified by the caller/resolver."""

    candidate: ResolvedCheckpointNode
    reference: ResolvedCheckpointNode
    master_seed: int
    startset: StartsetRef
    config: ArenaExecutionConfig
    profile: str = "torus9"
    workload: Mapping[str, object] = field(default_factory=dict)
    scientific_contract: Mapping[str, object] | None = None
    execution_contract: Mapping[str, object] | None = None
    candidate_label: str | None = None
    reference_label: str | None = None
    comparison: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ResolvedCheckpointNode):
            raise TypeError("candidate must be resolved by ArtifactResolver")
        if not isinstance(self.reference, ResolvedCheckpointNode):
            raise TypeError("reference must be resolved by ArtifactResolver")
        if self.candidate.topology != self.reference.topology:
            raise ValueError("Arena checkpoints must share topology")
        if int(self.config.games) <= 0 or int(self.config.games) % 2:
            raise ValueError("Arena games must be a positive even number")


@dataclass(frozen=True)
class ArenaRunResult:
    evaluation_id: str
    evaluation_fingerprint: str
    output_dir: Path
    identity: EvaluationIdentity
    summary: Mapping[str, object]
    validity: str

    @property
    def wld(self) -> tuple[int, int, int]:
        value = self.summary.get("W/L/D")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("Arena summary is missing W/L/D")
        return int(value[0]), int(value[1]), int(value[2])


class ArenaRunner:
    """Run one V2 Arena evaluation through the existing production engine."""

    def __init__(
        self,
        engine: Callable[..., Mapping[str, object]] | None = None,
    ) -> None:
        self.engine = engine or production_arena

    @staticmethod
    def _identity(request: ArenaRunRequest) -> EvaluationIdentity:
        scientific = dict(
            request.scientific_contract
            or TORUS9_ARENA_PROFILE.scientific_contract(request.config)
        )
        execution = dict(
            request.execution_contract
            or {
                "engine": "process-central-inference-v1",
                "workers": int(request.config.workers),
                "contexts": int(request.config.workers)
                * int(request.config.games_per_worker),
                "inference_batch_cap": int(request.config.inference_batch_rows),
                "inference_batch_wait_ms": float(request.config.inference_batch_wait_ms),
                "monitoring_acceptance": bool(request.config.monitoring_acceptance),
                "config": asdict(request.config),
            }
        )
        workload = dict(request.workload)
        workload.setdefault("pairs", int(request.config.games) // 2)
        workload.setdefault("paired_starts", True)
        workload.setdefault("color_swap", True)
        return EvaluationIdentity(
            candidate=request.candidate.ref,
            reference=request.reference.ref,
            games=int(request.config.games),
            master_seed=int(request.master_seed),
            startset=request.startset,
            scientific_contract=scientific,
            execution_contract=execution,
            workload=workload,
        )

    @staticmethod
    def _validity(summary: Mapping[str, object]) -> str:
        telemetry = summary.get("telemetry")
        if not isinstance(telemetry, Mapping):
            return "INVALID"
        if int(telemetry.get("technical_games", -1)) != 0:
            return "TECHNICAL"
        if (
            str(telemetry.get("performance_status", "")).upper() == "CRITICAL"
            or bool(telemetry.get("performance_failures"))
        ):
            return "CRITICAL"
        return "VALID"

    @staticmethod
    def _provenance(
        request: ArenaRunRequest,
        identity: EvaluationIdentity,
        run_id: str,
    ) -> dict[str, object]:
        payload = identity.to_dict()
        return {
            "schema": ARENA_RESULT_PROVENANCE_SCHEMA,
            "evaluation_id": run_id,
            "candidate": request.candidate.ref.to_dict(),
            "reference": request.reference.ref.to_dict(),
            "profile": str(identity.scientific_contract.get("profile", request.profile)),
            "master_seed": int(request.master_seed),
            "startset": payload["startset"],
            "arena_contract": {
                "scientific": payload["scientific_contract"],
                "execution": payload["execution_contract"],
                "workload": payload["workload"],
            },
            "training_mutated": False,
        }

    def run(self, request: ArenaRunRequest) -> ArenaRunResult:
        """Run exactly one evaluation for the supplied explicit refs."""
        config = request.config
        config.validate_base()
        identity = self._identity(request)
        # Use the shared V1/staged canonical identity primitive for the
        # persisted/reuse key.  ``EvaluationIdentity.fingerprint`` is the
        # contract-layer ``sha256:...`` form; staged Arena IDs intentionally
        # use the bare digest and the shared primitive preserves that format.
        fingerprint = evaluation_fingerprint(identity.to_dict())
        run_id = evaluation_id(
            candidate_lineage_id=request.candidate.lineage_id,
            candidate_generation=request.candidate.generation,
            reference_lineage_id=request.reference.lineage_id,
            reference_generation=request.reference.generation,
            fingerprint=fingerprint,
        )
        output = evaluation_dir(request.candidate.topology, run_id).resolve()

        if output.exists():
            existing = load_reusable_evaluation(output, identity.to_dict(), fingerprint)
            if existing is not None:
                return ArenaRunResult(
                    evaluation_id=run_id,
                    evaluation_fingerprint=fingerprint,
                    output_dir=output,
                    identity=identity,
                    summary=existing,
                    validity=self._validity(existing),
                )
            # The identity marker is already checked above; only incomplete
            # Arena output is removable. Checkpoints are in lineage storage and
            # are never children of this evaluation directory.
            shutil.rmtree(output)

        write_evaluation_identity(output, run_id, identity.to_dict(), fingerprint)
        try:
            summary = dict(
                self.engine(
                    candidate_path=request.candidate.path,
                    reference_path=request.reference.path,
                    profile_name=request.profile,
                    output_dir=output,
                    candidate_label=request.candidate_label or request.candidate.checkpoint_id,
                    reference_label=request.reference_label or request.reference.checkpoint_id,
                    run_id=run_id,
                    comparison=request.comparison
                    or f"{request.candidate.checkpoint_id}-vs-{request.reference.checkpoint_id}",
                    master_seed=int(request.master_seed),
                    config=config,
                    expected_candidate_artifact_sha256=request.candidate.ref.sha256,
                    expected_reference_artifact_sha256=request.reference.ref.sha256,
                )
            )
        except Exception:
            # Keep the identity marker for fail-closed diagnosis/retry, just as
            # the staged V1 mechanism does for interrupted Arenas.
            raise

        provenance_path = output / "provenance.json"
        provenance_path.write_text(
            json.dumps(self._provenance(request, identity, run_id), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        stamp_evaluation_identity_metadata(
            output,
            run_id,
            fingerprint,
            identity_schema=identity.schema,
        )
        return ArenaRunResult(
            evaluation_id=run_id,
            evaluation_fingerprint=fingerprint,
            output_dir=output,
            identity=identity,
            summary=summary,
            validity=self._validity(summary),
        )


ArenaRunnerV2 = ArenaRunner


__all__ = [
    "ARENA_RESULT_PROVENANCE_SCHEMA",
    "ArenaRunRequest",
    "ArenaRunResult",
    "ArenaRunner",
    "ArenaRunnerV2",
    "torus9_startset_ref",
]
