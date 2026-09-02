#!/usr/bin/env python3
"""Validate a TetGen tetrahedral mesh and export canonical AVS and PFLOTRAN UGI files.

The AVS file used by LANL VORONOI and the PFLOTRAN geomechanics UGI file are
written from the *same* remapped, positively oriented tetrahedral connectivity.
This prevents the flow dual grid and geomechanics mesh from silently using
slightly different node numbering or element orientation.

Typical use
-----------
python3 tetgen_to_avs_ugi_canonical.py \
    bartlesville_hec_lime_v5_interfaces.1.node \
    bartlesville_hec_lime_v5_interfaces.1.ele \
    bartlesville_hec_lime_v5_interfaces.inp \
    bartlesville_hec_lime_v5_interfaces.ugi

Outputs
-------
* AVS UCD tetrahedral mesh for LANL VORONOI.
* PFLOTRAN implicit unstructured tetrahedral mesh (UGI).
* A validation report beside the AVS file.

Numbering convention
--------------------
TetGen node IDs are remapped to contiguous IDs 1..N in `.node` row order.
That row-order convention is the one used by the material, vset, and identity
flow-to-geomechanics mapping scripts in this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Node:
    tetgen_id: int
    xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class Tet:
    tetgen_id: int
    nodes: Tuple[int, int, int, int]
    attribute: int


@dataclass(frozen=True)
class CanonicalMesh:
    xyz: np.ndarray
    connectivity: np.ndarray
    attributes: np.ndarray
    reoriented_count: int
    min_abs_six_volume: float
    max_abs_six_volume: float


def data_lines(path: Path) -> Iterable[str]:
    """Yield non-empty TetGen data lines with inline comments removed."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_nodes(path: Path) -> List[Node]:
    lines = iter(data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty node file: {path}") from exc

    if len(header) < 4:
        raise RuntimeError(f"Malformed node header in {path}: {' '.join(header)}")

    count = int(float(header[0]))
    dimension = int(float(header[1]))
    if count <= 0:
        raise RuntimeError(f"Node count must be positive in {path}; received {count}.")
    if dimension != 3:
        raise RuntimeError(f"Expected 3-D TetGen nodes, got dimension={dimension} in {path}.")

    nodes: List[Node] = []
    seen: set[int] = set()
    for row_index in range(1, count + 1):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(
                f"{path}: expected {count} node rows, stopped at {row_index - 1}."
            ) from exc

        if len(fields) < 4:
            raise RuntimeError(
                f"{path}: node row {row_index} has fewer than four fields: {' '.join(fields)}"
            )

        node_id = int(float(fields[0]))
        if node_id in seen:
            raise RuntimeError(f"{path}: duplicate TetGen node ID {node_id}.")
        seen.add(node_id)

        xyz = tuple(float(value) for value in fields[1:4])
        if not np.all(np.isfinite(xyz)):
            raise RuntimeError(f"{path}: node {node_id} has non-finite coordinates {xyz}.")
        nodes.append(Node(node_id, xyz))

    # Extra non-comment lines usually indicate a stale header or malformed file.
    try:
        extra = next(lines)
    except StopIteration:
        extra = None
    if extra is not None:
        raise RuntimeError(
            f"{path}: contains more node rows than the declared count {count}; first extra row: {extra}"
        )

    return nodes


def read_tets(path: Path) -> List[Tet]:
    lines = iter(data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty element file: {path}") from exc

    if len(header) < 2:
        raise RuntimeError(f"Malformed element header in {path}: {' '.join(header)}")

    count = int(float(header[0]))
    nodes_per_element = int(float(header[1]))
    number_of_attributes = int(float(header[2])) if len(header) >= 3 else 0

    if count <= 0:
        raise RuntimeError(f"Element count must be positive in {path}; received {count}.")
    if nodes_per_element != 4:
        raise RuntimeError(
            f"Expected tetrahedra (4 nodes/element), got {nodes_per_element} in {path}."
        )
    if number_of_attributes < 0:
        raise RuntimeError(f"Invalid attribute count {number_of_attributes} in {path}.")

    tets: List[Tet] = []
    seen: set[int] = set()
    minimum_fields = 1 + nodes_per_element + number_of_attributes

    for row_index in range(1, count + 1):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(
                f"{path}: expected {count} element rows, stopped at {row_index - 1}."
            ) from exc

        if len(fields) < minimum_fields:
            raise RuntimeError(
                f"{path}: element row {row_index} has {len(fields)} fields; "
                f"expected at least {minimum_fields}: {' '.join(fields)}"
            )

        tet_id = int(float(fields[0]))
        if tet_id in seen:
            raise RuntimeError(f"{path}: duplicate TetGen element ID {tet_id}.")
        seen.add(tet_id)

        conn = tuple(int(float(value)) for value in fields[1:5])
        attribute = (
            int(round(float(fields[5])))
            if number_of_attributes > 0
            else 1
        )
        tets.append(Tet(tet_id, conn, attribute))

    try:
        extra = next(lines)
    except StopIteration:
        extra = None
    if extra is not None:
        raise RuntimeError(
            f"{path}: contains more element rows than the declared count {count}; first extra row: {extra}"
        )

    return tets


def signed_six_volume(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    return float(np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))))


def canonicalize_mesh(
    nodes: Sequence[Node],
    tets: Sequence[Tet],
    *,
    min_abs_six_volume: float,
) -> CanonicalMesh:
    """Remap IDs, validate connectivity, and enforce positive orientation."""
    if min_abs_six_volume <= 0.0:
        raise ValueError("min_abs_six_volume must be positive.")

    id_to_canonical: Dict[int, int] = {
        node.tetgen_id: row_index
        for row_index, node in enumerate(nodes, start=1)
    }
    xyz = np.asarray([node.xyz for node in nodes], dtype=float)

    canonical_conn = np.empty((len(tets), 4), dtype=np.int64)
    attributes = np.empty(len(tets), dtype=np.int64)

    missing: List[Tuple[int, Tuple[int, int, int, int]]] = []
    repeated: List[Tuple[int, Tuple[int, int, int, int]]] = []
    degenerate: List[Tuple[int, Tuple[int, int, int, int], float]] = []

    reoriented = 0
    min_six = float("inf")
    max_six = 0.0

    for row_index, tet in enumerate(tets):
        if any(node_id not in id_to_canonical for node_id in tet.nodes):
            missing.append((tet.tetgen_id, tet.nodes))
            continue

        conn = tuple(id_to_canonical[node_id] for node_id in tet.nodes)
        if len(set(conn)) != 4:
            repeated.append((tet.tetgen_id, conn))
            continue

        p0, p1, p2, p3 = (xyz[node_id - 1] for node_id in conn)
        six_volume = signed_six_volume(p0, p1, p2, p3)
        abs_six_volume = abs(six_volume)

        if (
            not np.isfinite(abs_six_volume)
            or abs_six_volume <= min_abs_six_volume
        ):
            degenerate.append((tet.tetgen_id, conn, six_volume))
            continue

        if six_volume < 0.0:
            # Swap the final two vertices so both AVS and UGI use the same
            # positively oriented tetrahedron.
            conn = (conn[0], conn[1], conn[3], conn[2])
            reoriented += 1

        canonical_conn[row_index, :] = conn
        attributes[row_index] = tet.attribute
        min_six = min(min_six, abs_six_volume)
        max_six = max(max_six, abs_six_volume)

    if missing or repeated or degenerate:
        details: List[str] = [
            "Canonical TetGen validation failed:",
            f"  missing node references : {len(missing)}",
            f"  repeated vertices       : {len(repeated)}",
            f"  degenerate tetrahedra   : {len(degenerate)}",
        ]
        for tet_id, conn in missing[:10]:
            details.append(f"  missing tet={tet_id} connectivity={conn}")
        for tet_id, conn in repeated[:10]:
            details.append(f"  repeated tet={tet_id} connectivity={conn}")
        for tet_id, conn, six_volume in degenerate[:10]:
            details.append(
                f"  degenerate tet={tet_id} connectivity={conn} "
                f"signed_six_volume={six_volume:.16e}"
            )
        raise RuntimeError("\n".join(details))

    if not np.all(canonical_conn > 0):
        raise RuntimeError("Internal error: canonical connectivity contains unset or non-positive IDs.")

    return CanonicalMesh(
        xyz=xyz,
        connectivity=canonical_conn,
        attributes=attributes,
        reoriented_count=reoriented,
        min_abs_six_volume=min_six,
        max_abs_six_volume=max_six,
    )


def connectivity_sha256(connectivity: np.ndarray) -> str:
    canonical_bytes = np.asarray(connectivity, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(canonical_bytes).hexdigest()


def write_avs(mesh: CanonicalMesh, output: Path) -> None:
    """Write AVS UCD input for LANL VORONOI."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{mesh.xyz.shape[0]} {mesh.connectivity.shape[0]} 0 0 0\n")
        for node_id, point in enumerate(mesh.xyz, start=1):
            handle.write(
                f"{node_id} {point[0]:.12e} {point[1]:.12e} {point[2]:.12e}\n"
            )

        # LANL VORONOI only needs a valid integer material flag here. The
        # PFLOTRAN flow materials are supplied separately through material HDF5.
        for element_id, conn in enumerate(mesh.connectivity, start=1):
            handle.write(
                f"{element_id} 1 tet "
                f"{conn[0]} {conn[1]} {conn[2]} {conn[3]}\n"
            )


def write_ugi(mesh: CanonicalMesh, output: Path) -> None:
    """Write PFLOTRAN tetrahedral implicit unstructured geomechanics mesh."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{mesh.connectivity.shape[0]} {mesh.xyz.shape[0]}\n")
        for conn in mesh.connectivity:
            handle.write(f"T {conn[0]} {conn[1]} {conn[2]} {conn[3]}\n")
        for point in mesh.xyz:
            handle.write(f"{point[0]:.12e} {point[1]:.12e} {point[2]:.12e}\n")


def write_report(
    output: Path,
    *,
    node_file: Path,
    ele_file: Path,
    avs_file: Path,
    ugi_file: Path,
    mesh: CanonicalMesh,
    min_abs_six_volume_threshold: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    unique_attributes, attribute_counts = np.unique(
        mesh.attributes,
        return_counts=True,
    )
    attribute_text = ",".join(
        f"{int(attribute)}:{int(count)}"
        for attribute, count in zip(unique_attributes, attribute_counts)
    )

    output.write_text(
        "Canonical TetGen export passed\n"
        f"node_file={node_file}\n"
        f"element_file={ele_file}\n"
        f"avs_file={avs_file}\n"
        f"ugi_file={ugi_file}\n"
        f"node_count={mesh.xyz.shape[0]}\n"
        f"tetrahedron_count={mesh.connectivity.shape[0]}\n"
        "node_mapping=tetgen_node_file_row_order_to_contiguous_1_based\n"
        "connectivity_source=single_canonical_array_shared_by_avs_and_ugi\n"
        f"connectivity_sha256={connectivity_sha256(mesh.connectivity)}\n"
        f"reoriented_tetrahedra={mesh.reoriented_count}\n"
        f"minimum_absolute_six_volume={mesh.min_abs_six_volume:.16e}\n"
        f"maximum_absolute_six_volume={mesh.max_abs_six_volume:.16e}\n"
        f"degenerate_threshold_absolute_six_volume={min_abs_six_volume_threshold:.16e}\n"
        f"element_attribute_counts={attribute_text}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a TetGen tetrahedral mesh and write canonical AVS and "
            "PFLOTRAN UGI files from identical connectivity."
        )
    )
    parser.add_argument("node_file", type=Path, help="TetGen .node file")
    parser.add_argument("ele_file", type=Path, help="TetGen .ele file")
    parser.add_argument("avs_file", type=Path, help="Output AVS UCD file")
    parser.add_argument("ugi_file", type=Path, help="Output PFLOTRAN UGI file")
    parser.add_argument(
        "--min-abs-six-volume",
        type=float,
        default=1.0e-12,
        help=(
            "Reject tetrahedra whose absolute signed six-volume is at or below "
            "this value. Default: 1e-12."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path. Default: <avs_stem>_canonical_export.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path, label in (
        (args.node_file, "node file"),
        (args.ele_file, "element file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if args.avs_file.resolve() == args.ugi_file.resolve():
        raise ValueError("AVS and UGI outputs must be different files.")

    print(f"Reading nodes: {args.node_file}")
    nodes = read_nodes(args.node_file)
    print(f"Reading tetrahedra: {args.ele_file}")
    tets = read_tets(args.ele_file)

    print("Canonicalizing connectivity and enforcing positive orientation...")
    mesh = canonicalize_mesh(
        nodes,
        tets,
        min_abs_six_volume=args.min_abs_six_volume,
    )

    print(f"Writing AVS: {args.avs_file}")
    write_avs(mesh, args.avs_file)
    print(f"Writing UGI: {args.ugi_file}")
    write_ugi(mesh, args.ugi_file)

    report = args.report or args.avs_file.with_name(
        args.avs_file.stem + "_canonical_export.txt"
    )
    write_report(
        report,
        node_file=args.node_file,
        ele_file=args.ele_file,
        avs_file=args.avs_file,
        ugi_file=args.ugi_file,
        mesh=mesh,
        min_abs_six_volume_threshold=args.min_abs_six_volume,
    )

    print("\nCanonical export complete")
    print(f"  nodes                  : {mesh.xyz.shape[0]:,}")
    print(f"  tetrahedra             : {mesh.connectivity.shape[0]:,}")
    print(f"  reoriented tetrahedra  : {mesh.reoriented_count:,}")
    print(f"  min |six-volume|       : {mesh.min_abs_six_volume:.6e}")
    print(f"  max |six-volume|       : {mesh.max_abs_six_volume:.6e}")
    print(f"  connectivity SHA-256   : {connectivity_sha256(mesh.connectivity)}")
    print(f"  report                 : {report}")


if __name__ == "__main__":
    main()
