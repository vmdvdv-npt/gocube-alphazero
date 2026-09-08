#!/usr/bin/env python3
"""Export the typed verification registry without network or production state."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphazero.envs.gocube.core import cube_topology, torus_topology
from tests.support.fixtures import (
    cube_verification_fixtures,
    torus_verification_fixtures,
    write_fixture_json,
)
from tests.support.product_boundary import (
    export_verified_product_boundary_fixtures,
    write_product_boundary_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tests/reference")
    parser.add_argument(
        "--product-output",
        help="optional GoCube fixture path; writes the same deterministic product-boundary artifact",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = cube_verification_fixtures()
    write_fixture_json(output_dir / "cube_verification_fixtures.json", fixtures)
    write_fixture_json(output_dir / "gocube_v1_torus_fixtures.json", torus_verification_fixtures())
    topology = cube_topology(4)
    boundary = export_verified_product_boundary_fixtures(fixtures, topology)
    boundary.extend(
        export_verified_product_boundary_fixtures(
            torus_verification_fixtures(),
            torus_topology(9),
        )
    )
    write_product_boundary_json(output_dir / "gocube_product_boundary_fixtures.json", boundary)
    if args.product_output:
        product_output = Path(args.product_output)
        product_output.parent.mkdir(parents=True, exist_ok=True)
        write_product_boundary_json(product_output, boundary)


if __name__ == "__main__":
    main()
