#!/usr/bin/env python3
"""Write and validate PFLOTRAN material IDs for a node-centred explicit grid.

The LANL VORONOI workflow creates one flow control volume for each TetGen
vertex.  Consequently, material row ``i`` must correspond to TetGen node-file
row ``i`` and to PFLOTRAN flow-cell ID ``i``.

This replacement keeps the historical command line::

    python3 material_h5_from_txt.py NODE_FILE H5_FILE MATERIAL_FILE

and adds optional cross-checks against the canonical geomechanics UGI and the
mesh-builder JSON sidecar::

    python3 material_h5_from_txt.py \
        bartlesville_hec_lime_v5_interfaces.1.node \
        bartlesville_hec_lime_v5_interfaces_material_ids.h5 \
        bartlesville_hec_lime_v5_interfaces_materials.txt \
        --ugi bartlesville_hec_lime_v5_interfaces.ugi \
        --geometry-json bartlesville_hec_lime_v5_interfaces_geometry.json

The HDF5 layout remains exactly:

    /Materials/Cell Ids
    /Materials/Material Ids

No compression is used, and the file is written atomically so a failed run
cannot leave a partially written production HDF5 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def iter_data_lines(path: Path) -> Iterable[str]:
    """Yield non-empty, comment-free lines."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def mesh_prefix_from_node(path: Path) -> str:
    name = path.name
    if name.endswith(".1.node"):
        return name[: -len(".1.node")]
    if name.endswith(".node"):
        return name[: -len(".node")]
    return path.stem


def read_tetgen_nodes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TetGen node IDs and coordinates in node-file row order."""
    lines = iter(iter_data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty TetGen node file: {path}") from exc

    if len(header) < 4:
        raise RuntimeError(f"Malformed TetGen node header in {path}: {' '.join(header)}")

    count = int(header[0])
    dimension = int(header[1])
    if count <= 0:
        raise RuntimeError(f"TetGen node count must be positive in {path}; got {count}.")
    if dimension != 3:
        raise RuntimeError(f"Expected a 3-D TetGen node file, got dimension={dimension} in {path}.")

    ids = np.empty(count, dtype=np.int64)
    xyz = np.empty((count, 3), dtype=np.float64)
    seen: set[int] = set()

    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{path}: expected {count} node rows, stopped at row {row}.") from exc
        if len(fields) < 4:
            raise RuntimeError(f"{path}: malformed node row {row + 1}: {' '.join(fields)}")

        node_id = int(fields[0])
        if node_id in seen:
            raise RuntimeError(f"{path}: duplicate TetGen node ID {node_id}.")
        seen.add(node_id)

        point = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            raise RuntimeError(f"{path}: node {node_id} contains non-finite coordinates {point.tolist()}.")

        ids[row] = node_id
        xyz[row] = point

    try:
        extra = next(lines)
    except StopIteration:
        extra = None
    if extra is not None:
        raise RuntimeError(f"{path}: contains extra node data after the declared {count} rows: {extra!r}")

    return ids, xyz


def read_materials(path: Path) -> np.ndarray:
    """Read exactly one positive integer material ID per data line."""
    values: list[int] = []
    for line_number, line in enumerate(iter_data_lines(path), start=1):
        fields = line.split()
        if len(fields) != 1:
            raise RuntimeError(
                f"{path}: material row {line_number} must contain exactly one integer; got {line!r}."
            )
        try:
            value = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(f"{path}: invalid material ID on row {line_number}: {fields[0]!r}") from exc
        if value <= 0:
            raise RuntimeError(f"{path}: material ID must be positive; row {line_number} has {value}.")
        values.append(value)

    if not values:
        raise RuntimeError(f"No material IDs were read from {path}.")
    return np.asarray(values, dtype=np.int64)


def read_ugi_and_validate_nodes(path: Path, expected_xyz: np.ndarray) -> tuple[int, int, float]:
    """Validate UGI counts and node coordinates against TetGen row order."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        header = handle.readline().split()
        if len(header) < 2:
            raise RuntimeError(f"Malformed UGI header in {path}.")
        element_count = int(header[0])
        node_count = int(header[1])

        if element_count <= 0 or node_count <= 0:
            raise RuntimeError(
                f"UGI counts must be positive in {path}; got elements={element_count}, nodes={node_count}."
            )
        if node_count != expected_xyz.shape[0]:
            raise RuntimeError(
                f"UGI node count {node_count} does not match TetGen node count {expected_xyz.shape[0]}."
            )

        for element_row in range(element_count):
            line = handle.readline()
            if not line:
                raise RuntimeError(
                    f"{path}: expected {element_count} element rows, stopped at row {element_row}."
                )
            fields = line.split()
            if len(fields) != 5 or fields[0] != "T":
                raise RuntimeError(
                    f"{path}: malformed tetrahedron row {element_row + 1}: {line.strip()!r}"
                )

        ugi_xyz = np.empty_like(expected_xyz)
        for node_row in range(node_count):
            line = handle.readline()
            if not line:
                raise RuntimeError(f"{path}: expected {node_count} node rows, stopped at row {node_row}.")
            fields = line.split()
            if len(fields) != 3:
                raise RuntimeError(f"{path}: malformed UGI node row {node_row + 1}: {line.strip()!r}")
            ugi_xyz[node_row] = [float(fields[0]), float(fields[1]), float(fields[2])]

        for trailing in handle:
            if trailing.strip():
                raise RuntimeError(f"{path}: unexpected trailing data after UGI node rows: {trailing.strip()!r}")

    if not np.all(np.isfinite(ugi_xyz)):
        raise RuntimeError(f"{path}: UGI contains non-finite node coordinates.")

    max_coordinate_difference = float(np.max(np.abs(ugi_xyz - expected_xyz)))
    if not np.allclose(ugi_xyz, expected_xyz, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(
            "UGI node coordinates do not match TetGen node-file row order; "
            f"maximum absolute difference is {max_coordinate_difference:.6e} m."
        )

    return element_count, node_count, max_coordinate_difference


def expected_material_ids_from_geometry(path: Path) -> set[int]:
    """Collect material IDs declared by layers, HEC, and refinement targets."""
    geometry = json.loads(path.read_text(encoding="utf-8"))
    expected: set[int] = set()

    layers = geometry.get("layers", [])
    if not isinstance(layers, list):
        raise RuntimeError(f"{path}: 'layers' must be a list.")
    for layer in layers:
        if not isinstance(layer, dict) or "material_id" not in layer:
            raise RuntimeError(f"{path}: malformed geological-layer entry: {layer!r}")
        expected.add(int(layer["material_id"]))

    hec = geometry.get("hec")
    if isinstance(hec, dict) and "material_id" in hec:
        expected.add(int(hec["material_id"]))

    targets = geometry.get("refinement_targets", [])
    if not isinstance(targets, list):
        raise RuntimeError(f"{path}: 'refinement_targets' must be a list.")
    for target in targets:
        if not isinstance(target, dict) or "material_id" not in target:
            raise RuntimeError(f"{path}: malformed refinement-target entry: {target!r}")
        expected.add(int(target["material_id"]))

    if not expected:
        raise RuntimeError(f"{path}: no material IDs were found in the geometry metadata.")
    return expected


def sha256_int64(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def write_hdf5_atomic(output: Path, cell_ids: np.ndarray, material_ids: np.ndarray) -> None:
    """Write the PFLOTRAN material HDF5 atomically and verify it by reopening."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with h5py.File(temporary, mode="w") as h5file:
            group = h5file.create_group("Materials")
            group.create_dataset("Cell Ids", data=cell_ids, dtype=np.int64)
            group.create_dataset("Material Ids", data=material_ids, dtype=np.int64)
            h5file.flush()

        with h5py.File(temporary, mode="r") as h5file:
            if "Materials/Cell Ids" not in h5file or "Materials/Material Ids" not in h5file:
                raise RuntimeError("Required material datasets are missing after HDF5 write.")
            written_cells = np.asarray(h5file["Materials/Cell Ids"], dtype=np.int64)
            written_materials = np.asarray(h5file["Materials/Material Ids"], dtype=np.int64)

        if not np.array_equal(written_cells, cell_ids):
            raise RuntimeError("Cell IDs changed during HDF5 write/read verification.")
        if not np.array_equal(written_materials, material_ids):
            raise RuntimeError("Material IDs changed during HDF5 write/read verification.")

        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate node-centred material IDs and write PFLOTRAN material HDF5."
    )
    parser.add_argument("node_file", type=Path, help="TetGen .node file")
    parser.add_argument("h5_file", type=Path, help="Output PFLOTRAN material HDF5")
    parser.add_argument("material_file", type=Path, help="One material ID per TetGen node-file row")
    parser.add_argument(
        "--ugi",
        type=Path,
        help="Optional canonical PFLOTRAN UGI; node coordinates are checked against the TetGen row order.",
    )
    parser.add_argument(
        "--geometry-json",
        type=Path,
        help="Optional mesh-builder geometry JSON; declared material IDs are checked against the material file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional validation-report path. Default: <h5 stem>_validation.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.node_file, "TetGen node file"),
        (args.material_file, "material file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    prefix = mesh_prefix_from_node(args.node_file)
    base_dir = args.node_file.parent

    ugi_path = args.ugi
    if ugi_path is None:
        candidate = base_dir / f"{prefix}.ugi"
        if candidate.is_file():
            ugi_path = candidate

    geometry_path = args.geometry_json
    if geometry_path is None:
        candidate = base_dir / f"{prefix}_geometry.json"
        if candidate.is_file():
            geometry_path = candidate

    print(f"Reading TetGen nodes: {args.node_file}")
    tetgen_ids, node_xyz = read_tetgen_nodes(args.node_file)
    node_count = node_xyz.shape[0]

    print(f"Reading material IDs: {args.material_file}")
    material_ids = read_materials(args.material_file)
    if material_ids.size != node_count:
        raise RuntimeError(
            f"Material row count {material_ids.size} does not match TetGen node count {node_count}."
        )

    cell_ids = np.arange(1, node_count + 1, dtype=np.int64)

    ugi_element_count: int | None = None
    ugi_max_coordinate_difference: float | None = None
    if ugi_path is not None:
        if not ugi_path.is_file():
            raise FileNotFoundError(f"UGI file not found: {ugi_path}")
        print(f"Validating canonical UGI: {ugi_path}")
        ugi_element_count, _, ugi_max_coordinate_difference = read_ugi_and_validate_nodes(
            ugi_path, node_xyz
        )
    else:
        print("WARNING: no UGI supplied or auto-detected; UGI/TetGen node-order check was skipped.")

    observed_ids = set(int(value) for value in np.unique(material_ids))
    expected_ids: set[int] | None = None
    if geometry_path is not None:
        if not geometry_path.is_file():
            raise FileNotFoundError(f"Geometry JSON not found: {geometry_path}")
        print(f"Validating material IDs against geometry metadata: {geometry_path}")
        expected_ids = expected_material_ids_from_geometry(geometry_path)
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        if missing or extra:
            raise RuntimeError(
                "Material-ID set does not match geometry metadata: "
                f"missing={missing}, extra={extra}."
            )
    else:
        print("WARNING: no geometry JSON supplied or auto-detected; declared material-ID check was skipped.")

    print(f"Writing material HDF5 atomically: {args.h5_file}")
    write_hdf5_atomic(args.h5_file, cell_ids, material_ids)

    distribution = Counter(int(value) for value in material_ids)
    report_path = args.report or args.h5_file.with_name(args.h5_file.stem + "_validation.txt")
    material_hash = sha256_int64(material_ids)

    report_lines = [
        "PFLOTRAN material HDF5 validation passed",
        f"node_file={args.node_file}",
        f"material_file={args.material_file}",
        f"h5_file={args.h5_file}",
        f"node_count={node_count}",
        "cell_id_mapping=tetgen_node_file_row_order_to_contiguous_1_based",
        f"tetgen_node_id_min={int(np.min(tetgen_ids))}",
        f"tetgen_node_id_max={int(np.max(tetgen_ids))}",
        f"material_ids_sha256={material_hash}",
        "dataset_cell_ids=/Materials/Cell Ids",
        "dataset_material_ids=/Materials/Material Ids",
        "hdf5_integer_dtype=int64",
        "hdf5_compression=none",
    ]
    if ugi_path is not None:
        report_lines.extend(
            [
                f"ugi_file={ugi_path}",
                f"ugi_element_count={ugi_element_count}",
                f"ugi_node_count={node_count}",
                f"ugi_max_node_coordinate_difference_m={ugi_max_coordinate_difference:.16e}",
            ]
        )
    if geometry_path is not None and expected_ids is not None:
        report_lines.append(f"geometry_json={geometry_path}")
        report_lines.append("expected_material_ids=" + ",".join(str(value) for value in sorted(expected_ids)))
    report_lines.append(
        "material_distribution="
        + ",".join(f"{material_id}:{distribution[material_id]}" for material_id in sorted(distribution))
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nMaterial HDF5 validation complete")
    print(f"  nodes / flow cells       : {node_count:,}")
    if ugi_element_count is not None:
        print(f"  geomechanics tetrahedra  : {ugi_element_count:,}")
        print(f"  max UGI/node difference  : {ugi_max_coordinate_difference:.3e} m")
    print(f"  observed material IDs    : {sorted(observed_ids)}")
    print(f"  material SHA-256         : {material_hash}")
    print("  material distribution:")
    for material_id in sorted(distribution):
        print(f"    {material_id:>2}: {distribution[material_id]:,}")
    print(f"  HDF5                     : {args.h5_file}")
    print(f"  report                   : {report_path}")


if __name__ == "__main__":
    main()
