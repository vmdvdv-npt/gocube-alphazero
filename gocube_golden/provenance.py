"""Small provenance primitives shared by the current Golden adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess


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
    return "sha256:" + digest.hexdigest()


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


def capture_code_identity(repo_root: str | Path | None = None) -> CodeIdentity:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
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


__all__ = [
    "CodeIdentity",
    "canonical_json",
    "capture_code_identity",
    "derive_seed",
    "file_sha256",
    "sha256_fingerprint",
]
