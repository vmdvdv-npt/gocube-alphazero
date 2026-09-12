from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .experiment_profile import (
    ARENA_CONTRACT_ID,
    BASELINE_KOMI,
    CHECKPOINT_REQUIRED_METADATA,
    EXPERIMENT_FINGERPRINT,
    NETWORK_HEADS_AND_SHAPES,
    OBSERVATION_FINGERPRINT,
    OBSERVATION_SCHEMA_ID,
    OBSERVATION_SCHEMA_VERSION,
    POINT_ID_ORDER_IDENTITY,
    PROFILE_ID,
    RULES_FINGERPRINT,
    RULES_PROFILE_ID,
    SEARCH_CONTRACT_FINGERPRINT,
    SEARCH_IMPLEMENTATION_ID,
    SEARCH_IMPLEMENTATION_FINGERPRINT,
    SEED_DERIVATION_ID,
    TARGET_CONTRACT_ID,
    TARGET_CONTRACT_VERSION,
    TARGET_FINGERPRINT,
    TOPOLOGY_FINGERPRINT,
    VALUE_HEAD_SEMANTICS,
)

PROVENANCE_SCHEMA_VERSION = 1
POLICY_SEMANTICS = "root-visits-over-legal-actions:[25-points+PASS]"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class CodeIdentity:
    git_commit_sha: str
    git_tree_sha: str
    working_tree_clean: bool

    @property
    def canonical(self) -> bool:
        return self.working_tree_clean

    def validate(self, *, require_canonical: bool = False) -> None:
        if not _GIT_SHA_RE.fullmatch(self.git_commit_sha):
            raise ValueError("Golden code identity requires a full 40-hex git commit SHA")
        if not _GIT_SHA_RE.fullmatch(self.git_tree_sha):
            raise ValueError("Golden code identity requires a full 40-hex git tree SHA")
        if require_canonical and not self.working_tree_clean:
            raise ValueError("Canonical Golden evidence rejects a dirty working tree")


@dataclass(frozen=True)
class PlayerIdentity:
    logical_player_id: str
    player_kind: str
    source_identity: str
    model_file_sha256: str | None
    checkpoint_metadata_fingerprint: str | None
    observation_contract_id: str | None
    observation_fingerprint: str | None
    target_contract_id: str | None
    target_fingerprint: str | None
    value_semantics: str | None
    policy_semantics: str | None

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(asdict(self))

    def validate(self) -> None:
        if not self.logical_player_id or not self.source_identity:
            raise ValueError("Player identity requires logical_player_id and source_identity")
        if self.player_kind not in {"scripted", "evaluator", "checkpoint"}:
            raise ValueError(f"Unsupported Golden player_kind {self.player_kind!r}")
        if self.player_kind == "checkpoint":
            required = {
                "model_file_sha256": self.model_file_sha256,
                "checkpoint_metadata_fingerprint": self.checkpoint_metadata_fingerprint,
                "observation_contract_id": self.observation_contract_id,
                "observation_fingerprint": self.observation_fingerprint,
                "target_contract_id": self.target_contract_id,
                "target_fingerprint": self.target_fingerprint,
                "value_semantics": self.value_semantics,
                "policy_semantics": self.policy_semantics,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError("Checkpoint player identity is incomplete: " + ", ".join(sorted(missing)))
            if not _SHA256_RE.fullmatch(str(self.model_file_sha256)):
                raise ValueError("Checkpoint model_file_sha256 must be sha256:<64 lowercase hex>")
            if not _SHA256_RE.fullmatch(str(self.checkpoint_metadata_fingerprint)):
                raise ValueError("Checkpoint metadata fingerprint must be sha256:<64 lowercase hex>")
            if self.observation_contract_id != OBSERVATION_SCHEMA_ID:
                raise ValueError("Checkpoint observation contract is incompatible with Golden profile")
            if self.observation_fingerprint != OBSERVATION_FINGERPRINT:
                raise ValueError("Checkpoint observation fingerprint is incompatible with Golden profile")
            if self.target_contract_id != TARGET_CONTRACT_ID:
                raise ValueError("Checkpoint target contract is incompatible with Golden profile")
            if self.target_fingerprint != TARGET_FINGERPRINT:
                raise ValueError("Checkpoint target fingerprint is incompatible with Golden profile")
            if self.value_semantics != VALUE_HEAD_SEMANTICS:
                raise ValueError("Checkpoint value semantics are incompatible with Golden profile")
            if self.policy_semantics != POLICY_SEMANTICS:
                raise ValueError("Checkpoint policy semantics are incompatible with Golden profile")


def run_identity_payload(
    *,
    schema_version: int,
    run_id: str,
    experiment_profile_id: str,
    experiment_fingerprint: str,
    git_commit_sha: str,
    git_tree_sha: str,
    git_worktree_clean: bool,
    rules_profile_id: str,
    rules_fingerprint: str,
    topology_fingerprint: str,
    komi: float,
    arena_contract_id: str,
    search_implementation_id: str,
    search_implementation_fingerprint: str,
    search_contract_fingerprint: str,
    search_settings: tuple[tuple[str, object], ...],
    seed_derivation_id: str,
    master_seed: int,
    player_A_identity_fingerprint: str,
    player_B_identity_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "experiment_profile_id": experiment_profile_id,
        "experiment_fingerprint": experiment_fingerprint,
        "git_commit_sha": git_commit_sha,
        "git_tree_sha": git_tree_sha,
        "git_worktree_clean": git_worktree_clean,
        "canonical_evidence": git_worktree_clean,
        "rules_profile_id": rules_profile_id,
        "rules_fingerprint": rules_fingerprint,
        "topology_fingerprint": topology_fingerprint,
        "komi": komi,
        "arena_contract_id": arena_contract_id,
        "search_implementation_id": search_implementation_id,
        "search_implementation_fingerprint": search_implementation_fingerprint,
        "search_contract_fingerprint": search_contract_fingerprint,
        "search_settings": search_settings,
        "seed_derivation_id": seed_derivation_id,
        "master_seed": master_seed,
        "player_A_identity_fingerprint": player_A_identity_fingerprint,
        "player_B_identity_fingerprint": player_B_identity_fingerprint,
    }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    experiment_profile_id: str
    experiment_fingerprint: str
    git_commit_sha: str
    git_tree_sha: str
    git_worktree_clean: bool
    canonical_evidence: bool
    rules_profile_id: str
    rules_fingerprint: str
    topology_fingerprint: str
    komi: float
    arena_contract_id: str
    search_implementation_id: str
    search_implementation_fingerprint: str
    search_contract_fingerprint: str
    search_settings: tuple[tuple[str, object], ...]
    seed_derivation_id: str
    master_seed: int
    pair_count: int
    game_count: int
    pair_schedule: tuple[tuple[str, str, str], ...]
    player_A: PlayerIdentity
    player_B: PlayerIdentity

    def identity_payload(self) -> dict[str, object]:
        return run_identity_payload(
            schema_version=self.schema_version,
            run_id=self.run_id,
            experiment_profile_id=self.experiment_profile_id,
            experiment_fingerprint=self.experiment_fingerprint,
            git_commit_sha=self.git_commit_sha,
            git_tree_sha=self.git_tree_sha,
            git_worktree_clean=self.git_worktree_clean,
            rules_profile_id=self.rules_profile_id,
            rules_fingerprint=self.rules_fingerprint,
            topology_fingerprint=self.topology_fingerprint,
            komi=self.komi,
            arena_contract_id=self.arena_contract_id,
            search_implementation_id=self.search_implementation_id,
            search_implementation_fingerprint=self.search_implementation_fingerprint,
            search_contract_fingerprint=self.search_contract_fingerprint,
            search_settings=self.search_settings,
            seed_derivation_id=self.seed_derivation_id,
            master_seed=self.master_seed,
            player_A_identity_fingerprint=self.player_A.fingerprint,
            player_B_identity_fingerprint=self.player_B.fingerprint,
        )

    @property
    def run_identity_fingerprint(self) -> str:
        return sha256_fingerprint(self.identity_payload())

    @property
    def manifest_fingerprint(self) -> str:
        return sha256_fingerprint(asdict(self))

    def validate(self, *, require_canonical: bool = False) -> None:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError("Unsupported Golden provenance schema version")
        if not self.run_id:
            raise ValueError("Golden RunManifest requires non-empty run_id")
        if self.experiment_profile_id != PROFILE_ID or self.experiment_fingerprint != EXPERIMENT_FINGERPRINT:
            raise ValueError("RunManifest experiment identity drift")
        CodeIdentity(self.git_commit_sha, self.git_tree_sha, self.git_worktree_clean).validate(require_canonical=require_canonical)
        if self.canonical_evidence != self.git_worktree_clean:
            raise ValueError("RunManifest canonical_evidence must reflect git working-tree cleanliness")
        if require_canonical and not self.canonical_evidence:
            raise ValueError("RunManifest is explicitly non-canonical")
        if self.rules_profile_id != RULES_PROFILE_ID or self.rules_fingerprint != RULES_FINGERPRINT:
            raise ValueError("RunManifest rules identity drift")
        if self.topology_fingerprint != TOPOLOGY_FINGERPRINT:
            raise ValueError("RunManifest topology fingerprint drift")
        if self.komi != BASELINE_KOMI:
            raise ValueError("RunManifest Golden komi must be exactly 0.5")
        if self.arena_contract_id != ARENA_CONTRACT_ID:
            raise ValueError("RunManifest Arena contract id drift")
        if self.search_implementation_id != SEARCH_IMPLEMENTATION_ID or self.search_implementation_fingerprint != SEARCH_IMPLEMENTATION_FINGERPRINT:
            raise ValueError("RunManifest search implementation identity drift")
        if self.search_contract_fingerprint != SEARCH_CONTRACT_FINGERPRINT:
            raise ValueError("RunManifest search contract fingerprint drift")
        expected_search_settings = tuple(sorted({
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": False,
            "fast_search": False,
            "resign": False,
            "root_policy_temperature": False,
            "move_temperature": 0.0,
            "deterministic_tie_break": True,
        }.items()))
        if self.search_settings != expected_search_settings:
            raise ValueError("RunManifest search settings drift")
        if self.seed_derivation_id != SEED_DERIVATION_ID:
            raise ValueError("RunManifest seed derivation contract drift")
        if self.pair_count < 0 or self.game_count < 0 or self.game_count != 2 * self.pair_count:
            raise ValueError("Golden paired evidence requires exactly two games per pair")
        if len(self.pair_schedule) != self.pair_count:
            raise ValueError("RunManifest pair_schedule length does not match pair_count")
        pair_ids: list[str] = []
        game_ids: list[str] = []
        for entry in self.pair_schedule:
            if len(entry) != 3 or not all(isinstance(item, str) and item for item in entry):
                raise ValueError("RunManifest pair_schedule entries must be (pair_id, A-black game_id, B-black game_id)")
            pair_id, a_black, b_black = entry
            pair_ids.append(pair_id)
            game_ids.extend((a_black, b_black))
        if len(pair_ids) != len(set(pair_ids)) or len(game_ids) != len(set(game_ids)):
            raise ValueError("RunManifest pair_schedule contains duplicate pair/game identity")
        if len(game_ids) != self.game_count:
            raise ValueError("RunManifest pair_schedule does not cover exactly game_count games")
        self.player_A.validate()
        self.player_B.validate()
        if require_canonical and (self.player_A.player_kind != "checkpoint" or self.player_B.player_kind != "checkpoint"):
            raise ValueError("Canonical Golden evidence requires exact checkpoint identities for both players")


def capture_code_identity(repo_root: str | Path | None = None) -> CodeIdentity:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        try:
            result = subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("Golden provenance cannot determine git code identity") from exc
        return result.stdout.strip()

    identity = CodeIdentity(
        git_commit_sha=git("rev-parse", "HEAD"),
        git_tree_sha=git("rev-parse", "HEAD^{tree}"),
        working_tree_clean=not bool(git("status", "--porcelain", "--untracked-files=normal")),
    )
    identity.validate()
    return identity


def derive_seed(master_seed: int, *parts: object) -> int:
    text = ":".join([str(int(master_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def derive_game_seeds(master_seed: int, pair_id: str, game_id: str) -> tuple[int, int, int]:
    if not pair_id or not game_id:
        raise ValueError("Golden seed derivation requires non-empty pair_id and game_id")
    seed_game = derive_seed(master_seed, pair_id, game_id, "game")
    return seed_game, derive_seed(seed_game, "A"), derive_seed(seed_game, "B")


def _stable_player_config(player: object) -> object:
    if not is_dataclass(player):
        return {}
    payload: dict[str, object] = {}
    for field in fields(player):
        if field.name in {"player_id", "is_search_player", "identity", "evaluator", "search"}:
            continue
        payload[field.name] = _jsonable(getattr(player, field.name))
    return payload


def infer_player_identity(player: object) -> PlayerIdentity:
    explicit = getattr(player, "identity", None)
    if isinstance(explicit, PlayerIdentity):
        explicit.validate()
        if explicit.logical_player_id != str(getattr(player, "player_id", "")):
            raise ValueError("Structured player identity logical_player_id disagrees with player_id")
        return explicit
    logical = str(getattr(player, "player_id", ""))
    if not logical:
        raise ValueError("Golden player requires non-empty player_id")
    cls = type(player)
    source = f"{cls.__module__}.{cls.__qualname__}"
    config = _stable_player_config(player)
    if config:
        source += ":" + sha256_fingerprint(config)
    is_search = bool(getattr(player, "is_search_player", False))
    evaluator = getattr(player, "evaluator", None)
    if is_search and evaluator is not None:
        evaluator_cls = type(evaluator)
        source += f"|evaluator={evaluator_cls.__module__}.{evaluator_cls.__qualname__}"
    identity = PlayerIdentity(
        logical_player_id=logical,
        player_kind="evaluator" if is_search else "scripted",
        source_identity=source,
        model_file_sha256=None,
        checkpoint_metadata_fingerprint=None,
        observation_contract_id=None,
        observation_fingerprint=None,
        target_contract_id=None,
        target_fingerprint=None,
        value_semantics=None,
        policy_semantics=None,
    )
    identity.validate()
    return identity


def evaluator_player_identity(logical_player_id: str, evaluator: object) -> PlayerIdentity:
    cls = type(evaluator)
    identity = PlayerIdentity(
        logical_player_id=str(logical_player_id),
        player_kind="evaluator",
        source_identity=f"{cls.__module__}.{cls.__qualname__}",
        model_file_sha256=None,
        checkpoint_metadata_fingerprint=None,
        observation_contract_id=None,
        observation_fingerprint=None,
        target_contract_id=None,
        target_fingerprint=None,
        value_semantics=None,
        policy_semantics=None,
    )
    identity.validate()
    return identity


def checkpoint_player_identity(
    *,
    logical_player_id: str,
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
    source_identity: str | None = None,
) -> PlayerIdentity:
    from .arena_contract import reject_checkpoint_arena_overrides

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"Golden checkpoint artifact does not exist: {path}")
    reject_checkpoint_arena_overrides(metadata)
    missing = [key for key in CHECKPOINT_REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError("Golden checkpoint metadata is incomplete: " + ", ".join(missing))

    expected = {
        "rules_profile_id": RULES_PROFILE_ID,
        "rules_fingerprint": RULES_FINGERPRINT,
        "topology_fingerprint": TOPOLOGY_FINGERPRINT,
        "board_size": [5, 5],
        "point_id_order_identity": POINT_ID_ORDER_IDENTITY,
        "komi": BASELINE_KOMI,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_fingerprint": OBSERVATION_FINGERPRINT,
        "target_contract_id": TARGET_CONTRACT_ID,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "target_fingerprint": TARGET_FINGERPRINT,
        "value_head_semantics": VALUE_HEAD_SEMANTICS,
        "network_heads_and_shapes": NETWORK_HEADS_AND_SHAPES,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Golden checkpoint metadata mismatch for {key}: expected {value!r}, got {metadata.get(key)!r}")
    if not isinstance(metadata.get("parent_or_source_run_identity"), str) or not metadata["parent_or_source_run_identity"]:
        raise ValueError("Golden checkpoint requires non-empty parent_or_source_run_identity")

    model_hash = file_sha256(path)
    normalized = str(metadata["model_hash"])
    if re.fullmatch(r"[0-9a-f]{64}", normalized):
        normalized = f"sha256:{normalized}"
    if normalized != model_hash:
        raise ValueError("Checkpoint metadata model_hash does not match artifact SHA256")

    identity = PlayerIdentity(
        logical_player_id=str(logical_player_id),
        player_kind="checkpoint",
        source_identity=source_identity or str(path.resolve()),
        model_file_sha256=model_hash,
        checkpoint_metadata_fingerprint=sha256_fingerprint(dict(metadata)),
        observation_contract_id=OBSERVATION_SCHEMA_ID,
        observation_fingerprint=OBSERVATION_FINGERPRINT,
        target_contract_id=TARGET_CONTRACT_ID,
        target_fingerprint=TARGET_FINGERPRINT,
        value_semantics=VALUE_HEAD_SEMANTICS,
        policy_semantics=POLICY_SEMANTICS,
    )
    identity.validate()
    return identity


def validate_run_manifest(manifest: RunManifest, *, require_canonical: bool = False) -> None:
    manifest.validate(require_canonical=require_canonical)
