"""Thin Orchestrator V2 boundary for the existing production Arena engine.

Checkpoint discovery and ancestry belong to ``ArtifactResolver``.  This
runner accepts resolver-produced checkpoint nodes and only coordinates
identity, canonical evaluation storage, and the already-qualified Arena
engine.  It intentionally has no checkpoint lookup or copying logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Mapping

from ..arena_identity import (
    evaluation_fingerprint,
    evaluation_id,
    load_reusable_evaluation,
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
from .immutable_runtime import (
    ImmutableRuntime,
    ImmutableRuntimeManager,
    execution_commit_from_lineage,
)
from .version import require_v2_process

from tools.arena import (
    ARENA_RESULT_PROVENANCE_SCHEMA,
    normalize_arena_validity,
    run_arena as production_arena,
)
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE


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
    # Cross-lineage evaluations stay in the canonical evaluations namespace.
    # A coordinator may opt a same-lineage evaluation into its owning lineage
    # without changing the Arena identity or execution semantics.
    output_dir: Path | None = None
    execution_code_commit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ResolvedCheckpointNode):
            raise TypeError("candidate must be resolved by ArtifactResolver")
        if not isinstance(self.reference, ResolvedCheckpointNode):
            raise TypeError("reference must be resolved by ArtifactResolver")
        if self.candidate.topology != self.reference.topology:
            raise ValueError("Arena checkpoints must share topology")
        if int(self.config.games) <= 0 or int(self.config.games) % 2:
            raise ValueError("Arena games must be a positive even number")
        if self.output_dir is not None:
            if self.candidate.lineage_id != self.reference.lineage_id:
                raise ValueError(
                    "custom Arena output_dir is allowed only for same-lineage evaluations"
                )
            candidate_root = Path(self.candidate.owner_root).resolve()
            reference_root = Path(self.reference.owner_root).resolve()
            if candidate_root != reference_root:
                raise ValueError(
                    "custom Arena output_dir requires one shared lineage owner root"
                )
            arena_root = (candidate_root / "arena").resolve()
            output_root = Path(self.output_dir).resolve()
            if output_root == arena_root:
                raise ValueError("custom Arena output_dir must be inside the lineage arena directory")
            try:
                output_root.relative_to(arena_root)
            except ValueError as exc:
                raise ValueError(
                    "custom Arena output_dir must be inside the lineage arena directory"
                ) from exc


@dataclass(frozen=True)
class ArenaRunResult:
    evaluation_id: str
    evaluation_fingerprint: str
    output_dir: Path
    identity: EvaluationIdentity
    summary: Mapping[str, object]
    validity: str
    execution_code_commit: str | None = None

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
        self._repo_root = Path(__file__).resolve().parents[2]
        self._runtime_manager = ImmutableRuntimeManager(self._repo_root)
        self._python_executable = sys.executable

    def _production_summary(
        self,
        *,
        request: ArenaRunRequest,
        output: Path,
        engine_kwargs: Mapping[str, object],
    ) -> tuple[dict[str, object], ImmutableRuntime]:
        pinned = request.execution_code_commit or execution_commit_from_lineage(
            request.candidate.owner_root
        )
        runtime = self._runtime_manager.ensure(pinned)
        request_path = output / ".arena-child-request.json"
        result_path = output / ".arena-child-result.json"
        payload = {
            "schema": "gocube-orchestrator-v2-arena-request-v1",
            "execution_code_commit": runtime.commit,
            "candidate_path": str(engine_kwargs["candidate_path"]),
            "reference_path": str(engine_kwargs["reference_path"]),
            "profile_name": str(engine_kwargs["profile_name"]),
            "output_dir": str(output),
            "candidate_label": str(engine_kwargs["candidate_label"]),
            "reference_label": str(engine_kwargs["reference_label"]),
            "run_id": str(engine_kwargs["run_id"]),
            "comparison": str(engine_kwargs["comparison"]),
            "master_seed": int(engine_kwargs["master_seed"]),
            "config": asdict(request.config),
            "expected_candidate_artifact_sha256": str(
                engine_kwargs["expected_candidate_artifact_sha256"]
            ),
            "expected_reference_artifact_sha256": str(
                engine_kwargs["expected_reference_artifact_sha256"]
            ),
            "evaluation_identity": engine_kwargs["evaluation_identity"],
            "evaluation_fingerprint": str(engine_kwargs["evaluation_fingerprint"]),
            "workload": dict(request.workload),
            "allowed_lineage_arena_root": (
                None
                if engine_kwargs.get("allowed_lineage_arena_root") is None
                else str(engine_kwargs["allowed_lineage_arena_root"])
            ),
            # The coordinator has already selected this output directory from
            # the configured Run Storage root.  The runtime checkout has its
            # own source-tree-relative ``runs`` directory, so carry the
            # coordinator's verified evaluation root across the process
            # boundary without copying or rediscovering artifacts.
            "allowed_evaluation_root": str(output.parent),
        }
        request_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        env = runtime.environment(os.environ)
        env["AZ_ORCHESTRATOR_VERSION"] = "V2"
        completed = subprocess.run(
            [
                self._python_executable,
                "-m",
                "gocube_golden.orchestrator_v2.arena_child",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            cwd=runtime.path,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "immutable production Arena child failed with exit code "
                f"{completed.returncode} (execution commit {runtime.commit})"
            )
        try:
            summary = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("immutable production Arena child returned no summary") from exc
        if not isinstance(summary, dict):
            raise RuntimeError("immutable production Arena child returned a non-object summary")
        return summary, runtime

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
    def _boundary_validity(summary: Mapping[str, object]) -> str:
        """Read the status already normalized by the production Arena.

        The default production boundary always supplies this field.  The
        missing-field compatibility fallback is limited to injected legacy
        synthetic engines used by older unit tests; it delegates to the same
        production boundary classifier instead of maintaining local rules.
        """
        value = summary.get("validity")
        if value is None:
            return normalize_arena_validity(summary)
        normalized = str(value).upper()
        if normalized in {"VALID", "TECHNICAL", "CRITICAL", "INVALID"}:
            return normalized
        return "INVALID"

    def run(self, request: ArenaRunRequest) -> ArenaRunResult:
        """Run exactly one evaluation for the supplied explicit refs."""
        if self.engine is production_arena:
            require_v2_process("gocube_golden.orchestrator_v2.ArenaRunnerV2")
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
        if request.output_dir is not None:
            requested_output = request.output_dir.resolve()
            # A coordinator may provide either a concrete run directory or a
            # lineage-owned generation directory.  The latter gets the same
            # identity-derived leaf name used by canonical evaluations.
            output = (
                requested_output
                if requested_output.name == run_id
                else requested_output / run_id
            )
        else:
            output = evaluation_dir(request.candidate.topology, run_id).resolve()

        if output.exists():
            existing = load_reusable_evaluation(
                output,
                identity.to_dict(),
                fingerprint,
                allow_legacy_synthetic=self.engine is not production_arena,
                allow_completed_non_valid=True,
            )
            if existing is not None:
                return ArenaRunResult(
                    evaluation_id=run_id,
                    evaluation_fingerprint=fingerprint,
                    output_dir=output,
                    identity=identity,
                    summary=existing,
                    validity=self._boundary_validity(existing),
                    execution_code_commit=request.execution_code_commit,
                )
            # The identity marker is already checked above; only incomplete
            # Arena output is removable. Checkpoints are in lineage storage and
            # are never children of this evaluation directory.
            shutil.rmtree(output)

        write_evaluation_identity(output, run_id, identity.to_dict(), fingerprint)
        engine_kwargs: dict[str, object] = {
            "candidate_path": request.candidate.path,
            "reference_path": request.reference.path,
            "profile_name": request.profile,
            "output_dir": output,
            "candidate_label": request.candidate_label or request.candidate.checkpoint_id,
            "reference_label": request.reference_label or request.reference.checkpoint_id,
            "run_id": run_id,
            "comparison": request.comparison
            or f"{request.candidate.checkpoint_id}-vs-{request.reference.checkpoint_id}",
            "master_seed": int(request.master_seed),
            "config": config,
            "expected_candidate_artifact_sha256": request.candidate.ref.sha256,
            "expected_reference_artifact_sha256": request.reference.ref.sha256,
            "evaluation_identity": identity.to_dict(),
            "evaluation_fingerprint": fingerprint,
            "workload": dict(request.workload),
        }
        if request.output_dir is not None:
            # The production engine enforces canonical evaluation storage. A
            # same-lineage coordinator is the one explicit exception: it has
            # already validated the path against this checkpoint owner's
            # lineage arena, so pass that narrow allowance through the engine
            # boundary without changing cross-lineage storage policy.
            engine_kwargs["allowed_lineage_arena_root"] = (
                Path(request.candidate.owner_root).resolve() / "arena"
            )
        runtime: ImmutableRuntime | None = None
        try:
            if self.engine is production_arena:
                summary, runtime = self._production_summary(
                    request=request,
                    output=output,
                    engine_kwargs=engine_kwargs,
                )
            else:
                summary = dict(self.engine(**engine_kwargs))
        except Exception:
            # Keep the identity marker for fail-closed diagnosis/retry, just as
            # the staged V1 mechanism does for interrupted Arenas.
            raise

        return ArenaRunResult(
            evaluation_id=run_id,
            evaluation_fingerprint=fingerprint,
            output_dir=output,
            identity=identity,
            summary=summary,
            validity=self._boundary_validity(summary),
            execution_code_commit=(None if runtime is None else runtime.commit),
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
