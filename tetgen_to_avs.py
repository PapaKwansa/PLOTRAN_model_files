#!/usr/bin/env python3
"""Robust TetGen-to-AVS converter for LANL VORONOI.

This replacement is deliberately strict:
  * skips comments and blank lines;
  * remaps TetGen node IDs to contiguous AVS IDs 1..N in node-file row order;
  * validates every tetrahedron before it is written;
  * enforces positive tetrahedron orientation;
  * writes an AVS UCD file with no node/cell attributes.

The row-order remapping is important because the accompanying material and vset
scripts use the row number in <mesh>.1.node as the PFLOTRAN/VORONOI cell ID.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                yield line


def read_nodes(path: Path) -> List[Node]:
    lines = iter(data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty node file: {path}") from exc
    if len(header) < 4:
        raise RuntimeError(f"Malformed node header in {path}: {' '.join(header)}")

    count = int(header[0])
    dim = int(header[1])
    if dim != 3:
        raise RuntimeError(f"Expected 3-D TetGen nodes, got dimension={dim} in {path}")

    nodes: List[Node] = []
    seen: set[int] = set()
    for row_index in range(1, count + 1):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{path}: expected {count} node rows, stopped at {row_index - 1}") from exc
        if len(fields) < 4:
            raise RuntimeError(f"{path}: node row {row_index} has fewer than four fields: {' '.join(fields)}")
        node_id = int(fields[0])
        if node_id in seen:
            raise RuntimeError(f"{path}: duplicate TetGen node ID {node_id}")
        seen.add(node_id)
        xyz = tuple(float(value) for value in fields[1:4])
        if not np.all(np.isfinite(xyz)):
            raise RuntimeError(f"{path}: node {node_id} has non-finite coordinates {xyz}")
        nodes.append(Node(node_id, xyz))

    return nodes


def read_tets(path: Path) -> List[Tet]:
    lines = iter(data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty element file: {path}") from exc
    if len(header) < 2:
        raise RuntimeError(f"Malformed element header in {path}: {' '.join(header)}")

    count = int(header[0])
    nodes_per_element = int(header[1])
    number_of_attributes = int(header[2]) if len(header) >= 3 else 0
    if nodes_per_element != 4:
        raise RuntimeError(
            f"Expected tetrahedra (4 nodes/element), got {nodes_per_element} in {path}."
        )

    tets: List[Tet] = []
    seen: set[int] = set()
    for row_index in range(1, count + 1):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{path}: expected {count} element rows, stopped at {row_index - 1}") from exc
        if len(fields) < 5:
            raise RuntimeError(f"{path}: element row {row_index} has fewer than five fields: {' '.join(fields)}")
        tet_id = int(fields[0])
        if tet_id in seen:
            raise RuntimeError(f"{path}: duplicate TetGen element ID {tet_id}")
        seen.add(tet_id)
        conn = tuple(int(value) for value in fields[1:5])
        attribute = int(round(float(fields[5]))) if number_of_attributes > 0 and len(fields) >= 6 else 1
        tets.append(Tet(tet_id, conn, attribute))

    return tets


def signed_six_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))))


def write_avs(nodes: List[Node], tets: List[Tet], output: Path) -> Dict[str, float | int]:
    # VORONOI uses AVS vertex IDs directly as PFLOTRAN cell IDs.  We number
    # vertices by the .node row order so all generated vsets remain consistent.
    id_to_avs: Dict[int, int] = {node.tetgen_id: row + 1 for row, node in enumerate(nodes)}
    xyz = np.asarray([node.xyz for node in nodes], dtype=float)

    bad_missing: List[Tuple[int, Tuple[int, int, int, int]]] = []
    bad_repeated: List[Tuple[int, Tuple[int, int, int, int]]] = []
    bad_degenerate: List[Tuple[int, Tuple[int, int, int, int], float]] = []
    repaired_orientation = 0
    avs_tets: List[Tuple[int, int, int, int]] = []
    min_abs_six_volume = float("inf")
    max_abs_six_volume = 0.0

    for tet in tets:
        if any(vertex not in id_to_avs for vertex in tet.nodes):
            bad_missing.append((tet.tetgen_id, tet.nodes))
            continue
        conn = tuple(id_to_avs[vertex] for vertex in tet.nodes)
        if len(set(conn)) != 4:
            bad_repeated.append((tet.tetgen_id, conn))
            continue

        p0, p1, p2, p3 = (xyz[index - 1] for index in conn)
        six_volume = signed_six_volume(p0, p1, p2, p3)
        abs_six_volume = abs(six_volume)
        if not np.isfinite(abs_six_volume) or abs_six_volume <= 1.0e-12:
            bad_degenerate.append((tet.tetgen_id, conn, six_volume))
            continue
        if six_volume < 0.0:
            # Positive orientation avoids sign inconsistencies in downstream
            # tetrahedral-volume calculations; geometry is otherwise unchanged.
            conn = (conn[0], conn[1], conn[3], conn[2])
            repaired_orientation += 1
        avs_tets.append(conn)
        min_abs_six_volume = min(min_abs_six_volume, abs_six_volume)
        max_abs_six_volume = max(max_abs_six_volume, abs_six_volume)

    if bad_missing or bad_repeated or bad_degenerate:
        report = output.with_name(output.stem + "_invalid_connectivity.txt")
        with report.open("w", encoding="utf-8") as handle:
            handle.write("Invalid TetGen-to-AVS connectivity report\n")
            handle.write(f"tetgen_nodes={len(nodes)}\n")
            handle.write(f"tetgen_tetrahedra={len(tets)}\n")
            handle.write(f"missing_node_references={len(bad_missing)}\n")
            handle.write(f"repeated_vertices={len(bad_repeated)}\n")
            handle.write(f"degenerate_tetrahedra={len(bad_degenerate)}\n")
            for tet_id, conn in bad_missing[:25]:
                handle.write(f"missing_node_reference tet={tet_id} connectivity={conn}\n")
            for tet_id, conn in bad_repeated[:25]:
                handle.write(f"repeated_vertex tet={tet_id} connectivity={conn}\n")
            for tet_id, conn, sixvol in bad_degenerate[:25]:
                handle.write(f"degenerate tet={tet_id} connectivity={conn} signed_six_volume={sixvol:.16e}\n")
        raise RuntimeError(
            "TetGen connectivity validation failed before AVS export. "
            f"Details: {report.name}"
        )

    if len(avs_tets) != len(tets):
        raise RuntimeError("Internal error: tetrahedron count changed during validation.")

    with output.open("w", encoding="utf-8") as handle:
        # AVS UCD header: nodes, cells, node-data, cell-data, model-data.
        handle.write(f"{len(nodes)} {len(avs_tets)} 0 0 0\n")
        for avs_id, point in enumerate(xyz, start=1):
            handle.write(f"{avs_id} {point[0]:.12e} {point[1]:.12e} {point[2]:.12e}\n")
        # VORONOI reads: element_id, integer material flag, 'tet', n1 n2 n3 n4.
        # The material flag is not used by the PFLOTRAN writer, so use 1.
        for element_id, conn in enumerate(avs_tets, start=1):
            handle.write(f"{element_id} 1 tet {conn[0]} {conn[1]} {conn[2]} {conn[3]}\n")

    report = output.with_name(output.stem + "_avs_validation.txt")
    report.write_text(
        "AVS validation passed\n"
        f"node_count={len(nodes)}\n"
        f"tetrahedron_count={len(avs_tets)}\n"
        f"id_mapping=node_file_row_order_to_contiguous_1_based\n"
        f"reoriented_tetrahedra={repaired_orientation}\n"
        f"minimum_absolute_six_volume={min_abs_six_volume:.16e}\n"
        f"maximum_absolute_six_volume={max_abs_six_volume:.16e}\n",
        encoding="utf-8",
    )
    return {
        "nodes": len(nodes),
        "tets": len(avs_tets),
        "reoriented": repaired_orientation,
        "min_abs_six_volume": min_abs_six_volume,
        "max_abs_six_volume": max_abs_six_volume,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TetGen mesh and write VORONOI-safe AVS UCD.")
    parser.add_argument("node_file", type=Path)
    parser.add_argument("ele_file", type=Path)
    parser.add_argument("avs_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = read_nodes(args.node_file)
    tets = read_tets(args.ele_file)
    result = write_avs(nodes, tets, args.avs_file)
    print("--> Wrote VORONOI-safe AVS mesh")
    print(f"    AVS file                   : {args.avs_file}")
    print(f"    nodes                      : {result['nodes']:,}")
    print(f"    tetrahedra                 : {result['tets']:,}")
    print(f"    reoriented tetrahedra      : {result['reoriented']:,}")
    print(f"    min |six-volume|           : {result['min_abs_six_volume']:.6e}")
    print(f"    validation report           : {args.avs_file.with_name(args.avs_file.stem + '_avs_validation.txt')}")


if __name__ == "__main__":
    main()
