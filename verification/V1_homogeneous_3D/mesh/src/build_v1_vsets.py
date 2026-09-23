#!/usr/bin/env python3
"""
Build self-contained PFLOTRAN vertex sets for the V1 homogeneous benchmark.

Boundary vsets are derived authoritatively from TetGen exterior-face markers.

Material-region vsets are derived from the already validated node-centered
material-ID vector.

Output:
    top.vset
    bottom.vset
    north.vset
    south.vset
    east.vset
    west.vset
    homogeneous_matrix.vset
    injector.vset

All IDs are canonical 1-based node-file row IDs, matching the UGE flow-cell
IDs and UGI mechanics-vertex IDs in the V1 workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOUNDARY_MARKERS = {
    1: "top",
    2: "bottom",
    3: "north",
    4: "south",
    5: "east",
    6: "west",
}


def data_lines(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_nodes(path: Path):
    lines = iter(data_lines(path))

    header = next(lines).split()

    count = int(header[0])
    dimension = int(header[1])

    if dimension != 3:
        raise RuntimeError(
            f"{path}: expected dimension 3, got {dimension}."
        )

    original_ids = np.empty(
        count,
        dtype=np.int64,
    )

    xyz = np.empty(
        (count, 3),
        dtype=float,
    )

    original_to_canonical = {}

    for row in range(count):
        fields = next(lines).split()

        original_id = int(fields[0])

        canonical_id = row + 1

        original_ids[row] = original_id

        xyz[row] = (
            float(fields[1]),
            float(fields[2]),
            float(fields[3]),
        )

        if original_id in original_to_canonical:
            raise RuntimeError(
                f"Duplicate TetGen node ID {original_id}."
            )

        original_to_canonical[
            original_id
        ] = canonical_id

    return (
        original_ids,
        xyz,
        original_to_canonical,
    )


def read_boundary_sets(
    face_path: Path,
    original_to_canonical: dict[int, int],
):
    lines = iter(data_lines(face_path))

    header = next(lines).split()

    face_count = int(header[0])
    marker_count = int(header[1])

    if marker_count < 1:
        raise RuntimeError(
            f"{face_path}: boundary markers are absent."
        )

    boundary_sets = {
        name: set()
        for name in BOUNDARY_MARKERS.values()
    }

    triangle_counts = {
        name: 0
        for name in BOUNDARY_MARKERS.values()
    }

    for row in range(face_count):
        fields = next(lines).split()

        if len(fields) < 5:
            raise RuntimeError(
                f"{face_path}: malformed face row {row + 1}."
            )

        node_ids = [
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
        ]

        marker = int(fields[4])

        if marker not in BOUNDARY_MARKERS:
            # Marker 7 is the internal injector surface.
            # It is intentionally not an external boundary vset.
            continue

        name = BOUNDARY_MARKERS[marker]

        triangle_counts[name] += 1

        for original_id in node_ids:
            try:
                canonical_id = (
                    original_to_canonical[
                        original_id
                    ]
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"{face_path}: face references unknown "
                    f"TetGen node ID {original_id}."
                ) from exc

            boundary_sets[name].add(
                canonical_id
            )

    return boundary_sets, triangle_counts


def read_material_ids(
    path: Path,
    expected_count: int,
):
    values = np.loadtxt(
        path,
        dtype=np.int64,
    )

    values = np.asarray(
        values,
        dtype=np.int64,
    ).reshape(-1)

    if values.size != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count:,} material IDs, "
            f"found {values.size:,}."
        )

    return values


def write_vset(
    path: Path,
    values,
):
    ids = sorted(
        {
            int(value)
            for value in values
        }
    )

    if not ids:
        raise RuntimeError(
            f"Refusing to write empty vset: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for value in ids:
            handle.write(
                f"{value}\n"
            )

    return len(ids)


def validate_boundary_coordinates(
    boundary_sets,
    xyz,
    geometry,
    atol,
):
    minimum = np.asarray(
        geometry["domain"]["min"],
        dtype=float,
    )

    maximum = np.asarray(
        geometry["domain"]["max"],
        dtype=float,
    )

    expected = {
        "top": (
            2,
            maximum[2],
        ),
        "bottom": (
            2,
            minimum[2],
        ),
        "north": (
            1,
            maximum[1],
        ),
        "south": (
            1,
            minimum[1],
        ),
        "east": (
            0,
            maximum[0],
        ),
        "west": (
            0,
            minimum[0],
        ),
    }

    for name, ids in boundary_sets.items():
        axis, value = expected[name]

        indices = np.asarray(
            sorted(ids),
            dtype=np.int64,
        ) - 1

        coordinates = xyz[
            indices,
            axis,
        ]

        if not np.allclose(
            coordinates,
            value,
            atol=atol,
            rtol=0.0,
        ):
            maximum_error = float(
                np.max(
                    np.abs(
                        coordinates
                        - value
                    )
                )
            )

            raise RuntimeError(
                f"{name}: boundary-coordinate validation failed; "
                f"maximum error={maximum_error:.6e} m."
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build V1 PFLOTRAN boundary and material-region vsets."
        )
    )

    parser.add_argument(
        "--node",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--face",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--materials",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--geometry-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--coordinate-atol",
        type=float,
        default=1.0e-8,
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading TetGen nodes: {args.node}"
    )

    (
        original_ids,
        xyz,
        original_to_canonical,
    ) = read_nodes(
        args.node
    )

    node_count = xyz.shape[0]

    print(
        f"Reading TetGen exterior faces: {args.face}"
    )

    (
        boundary_sets,
        triangle_counts,
    ) = read_boundary_sets(
        args.face,
        original_to_canonical,
    )

    print(
        f"Reading node-centered materials: {args.materials}"
    )

    materials = read_material_ids(
        args.materials,
        node_count,
    )

    print(
        f"Reading geometry metadata: {args.geometry_json}"
    )

    geometry = json.loads(
        args.geometry_json.read_text(
            encoding="utf-8"
        )
    )

    validate_boundary_coordinates(
        boundary_sets,
        xyz,
        geometry,
        args.coordinate_atol,
    )

    expected_material_ids = {
        int(
            geometry["layers"][0][
                "material_id"
            ]
        ),
        int(
            geometry[
                "refinement_targets"
            ][0][
                "material_id"
            ]
        ),
    }

    observed_material_ids = set(
        int(value)
        for value in np.unique(
            materials
        )
    )

    if (
        observed_material_ids
        != expected_material_ids
    ):
        raise RuntimeError(
            "Material-ID mismatch: "
            f"expected={sorted(expected_material_ids)}, "
            f"observed={sorted(observed_material_ids)}."
        )

    matrix_id = int(
        geometry["layers"][0][
            "material_id"
        ]
    )

    injector_id = int(
        geometry[
            "refinement_targets"
        ][0][
            "material_id"
        ]
    )

    matrix_ids = (
        np.flatnonzero(
            materials == matrix_id
        )
        + 1
    )

    injector_ids = (
        np.flatnonzero(
            materials == injector_id
        )
        + 1
    )

    if (
        matrix_ids.size
        + injector_ids.size
        != node_count
    ):
        raise RuntimeError(
            "Material-region vsets do not cover all nodes."
        )

    if np.intersect1d(
        matrix_ids,
        injector_ids,
    ).size:
        raise RuntimeError(
            "Matrix and injector vsets overlap."
        )

    print()
    print("Writing boundary vsets")

    boundary_counts = {}

    for name in (
        "top",
        "bottom",
        "north",
        "south",
        "east",
        "west",
    ):
        path = (
            args.outdir
            / f"{name}.vset"
        )

        count = write_vset(
            path,
            boundary_sets[name],
        )

        boundary_counts[name] = count

        print(
            f"  {name:8s}: "
            f"{count:6,d} nodes, "
            f"{triangle_counts[name]:6,d} triangles"
        )

    print()
    print("Writing material-region vsets")

    matrix_count = write_vset(
        args.outdir
        / "homogeneous_matrix.vset",
        matrix_ids,
    )

    injector_count = write_vset(
        args.outdir
        / "injector.vset",
        injector_ids,
    )

    print(
        f"  homogeneous_matrix : "
        f"{matrix_count:,}"
    )

    print(
        f"  injector           : "
        f"{injector_count:,}"
    )

    if injector_count != 122:
        print(
            "WARNING: injector node count differs "
            "from the current validated M2 reference of 122."
        )

    print()
    print("Cross-checks")

    print(
        f"  total nodes            : {node_count:,}"
    )

    print(
        f"  material-region total  : "
        f"{matrix_count + injector_count:,}"
    )

    print(
        "  material overlap       : 0"
    )

    print(
        "  boundary coordinates   : PASS"
    )

    print(
        "  canonical ID convention: "
        "TetGen node-file row order, 1-based"
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
