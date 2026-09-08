#!/usr/bin/env python3
"""Export the typed verification registry without network or production state."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphazero.envs.gocube.core import cube_topology
from tests.support.fixtures import cube_verification_fixtures, write_fixture_json
from tests.support.product_boundary import export_product_boundary_fixture, write_product_boundary_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tests/reference")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = cube_verification_fixtures()
    write_fixture_json(output_dir / "cube_verification_fixtures.json", fixtures)
    topology = cube_topology(4)
    boundary = [
        export_product_boundary_fixture(fixture, topology)
        for fixture in fixtures
        if fixture.family in ("cleanup", "early_termination")
    ]
    write_product_boundary_json(output_dir / "gocube_product_boundary_fixtures.json", boundary)


if __name__ == "__main__":
    main()
