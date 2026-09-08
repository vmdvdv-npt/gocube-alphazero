"""Canonical V1 verification matrix metadata.

The matrix is intentionally data-shaped so CI and the human report use the
same family names and acceptance statuses.  Counts for the native oracle are
passed in by the test runner because the checked-in KataGo fixtures are the
source of truth, not a copied Python expected-output table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .fixtures import VerificationFixture, fixture_counts

PASS = "PASS"
EXPLAINED_DIFFERENCE = "EXPLAINED_DIFFERENCE"
ACCEPTED_STATUSES = frozenset((PASS, EXPLAINED_DIFFERENCE))

REQUIRED_FAMILIES = (
    "rectangular basic placement",
    "rectangular capture",
    "suicide",
    "true ko",
    "false ko",
    "PASS",
    "MAIN→C1",
    "C1→C2",
    "C2→score",
    "cleanup ko",
    "cycle",
    "pass-alive",
    "vertex groups",
    "vertex capture",
    "seam groups",
    "seam capture",
    "global connectivity",
    "eyes",
    "false eye",
    "seki",
    "dame",
    "scoring/setup",
    "S1 intruder",
    "ownership",
    "Torus wrap group",
    "Torus capture",
    "Torus ko",
    "rotations",
)


@dataclass(frozen=True)
class MatrixRow:
    family: str
    fixture_count: int
    independent_oracle: str
    katago_applicable: bool
    production_compared: bool
    rotations: str
    status: str = PASS
    difference_id: str = "none"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.family not in REQUIRED_FAMILIES:
            raise ValueError(f"Unknown V1 matrix family: {self.family}")
        if self.fixture_count < 0:
            raise ValueError("fixture_count must be non-negative")
        if self.status not in ACCEPTED_STATUSES:
            raise ValueError(f"Unaccepted V1 matrix status: {self.status}")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "fixture_count": self.fixture_count,
            "independent_oracle": self.independent_oracle,
            "katago_applicable": self.katago_applicable,
            "production_compared": self.production_compared,
            "rotations": self.rotations,
            "status": self.status,
            "difference_id": self.difference_id,
            "notes": self.notes,
        }


def _count(fixture_counts_by_family: Mapping[str, int], *names: str) -> int:
    return sum(int(fixture_counts_by_family.get(name, 0)) for name in names)


def canonical_matrix(
    cube_fixtures: Iterable[VerificationFixture],
    torus_fixtures: Iterable[VerificationFixture],
    *,
    rectangular_fixture_count: int = 28,
    generated_sequence_count: int = 8,
) -> tuple[MatrixRow, ...]:
    """Build the one accepted matrix used by the V1 report and tests."""

    cube_fixtures = tuple(cube_fixtures)
    torus_fixtures = tuple(torus_fixtures)
    cube_counts = fixture_counts(cube_fixtures)
    torus_counts = fixture_counts(torus_fixtures)
    rows = (
        MatrixRow("rectangular basic placement", rectangular_fixture_count, "pinned KataGo", True, True, "n/a", notes="Includes generated sequence pre-state masks."),
        MatrixRow("rectangular capture", rectangular_fixture_count, "pinned KataGo", True, True, "n/a"),
        MatrixRow("suicide", rectangular_fixture_count, "pinned KataGo + graph control", True, True, "n/a"),
        MatrixRow("true ko", 1, "pinned KataGo + independent graph restoration", True, True, "24 for Cube safe fixtures"),
        MatrixRow("false ko", 1, "independent graph restoration", False, True, "24 for Cube safe fixtures"),
        MatrixRow("PASS", rectangular_fixture_count, "pinned KataGo + independent PASS transition", True, True, "n/a"),
        MatrixRow("MAIN→C1", 1, "pinned KataGo", True, True, "n/a"),
        MatrixRow("C1→C2", 1, "pinned KataGo", True, True, "history-aware"),
        MatrixRow("C2→score", 1, "pinned KataGo", True, True, "history-aware"),
        MatrixRow("cleanup ko", 4, "pinned KataGo", True, True, "history-aware"),
        MatrixRow("cycle", 1, "pinned KataGo", True, True, "history-aware"),
        MatrixRow("pass-alive", 2, "pinned KataGo + independent graph", True, True, "Cube/Torus smoke"),
        MatrixRow("vertex groups", _count(cube_counts, "vertex_groups"), "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("vertex capture", 3, "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("seam groups", _count(cube_counts, "seam_tactics"), "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("seam capture", 1, "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("global connectivity", _count(cube_counts, "global_connectivity"), "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("eyes", 2, "independent graph + reviewed proof", False, True, "24 per rotation-safe fixture"),
        MatrixRow("false eye", 1, "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("seki", 1, "manual-reviewed graph proof", False, True, "24 per rotation-safe fixture"),
        MatrixRow("dame", 1, "independent graph", False, True, "24 per rotation-safe fixture"),
        MatrixRow("scoring/setup", 4, "pinned KataGo", True, True, "not multiplied by rotations"),
        MatrixRow("S1 intruder", 1, "independent graph + pinned rectangular analog", False, True, "history-aware"),
        MatrixRow("ownership", 4, "pinned KataGo + independent region proof", True, True, "rotation-invariant"),
        MatrixRow("Torus wrap group", _count(torus_counts, "wrap_group"), "independent graph", False, True, "Torus smoke"),
        MatrixRow("Torus capture", _count(torus_counts, "wrap_capture"), "independent graph", False, True, "Torus smoke"),
        MatrixRow("Torus ko", _count(torus_counts, "wrap_ko"), "independent graph", False, True, "Torus smoke"),
        MatrixRow("rotations", sum(1 for fixture in cube_fixtures if fixture.rotation_safe), "metamorphic + independent source fixture", False, True, "24 per source fixture", notes=f"{generated_sequence_count} generated rectangular seeds are not rotation multiplication."),
    )
    if tuple(row.family for row in rows) != REQUIRED_FAMILIES:
        raise AssertionError("V1 matrix family order drifted")
    return rows


def assert_matrix_accepted(rows: Iterable[MatrixRow]) -> None:
    rows = tuple(rows)
    missing = set(REQUIRED_FAMILIES) - {row.family for row in rows}
    if missing:
        raise AssertionError(f"V1 matrix is missing families: {sorted(missing)}")
    failures = [row for row in rows if row.status not in ACCEPTED_STATUSES]
    if failures:
        raise AssertionError(f"V1 matrix contains unaccepted rows: {failures!r}")
