#!/usr/bin/env python3
"""TetGen mesh quality report for Bartlesville / North Avant meshes.

Reads TetGen .node and .ele files, computes simple tetrahedron quality
measures, writes a CSV summary, and exports an unstructured VTK file that
ParaView can color by quality metrics.

Metrics included
----------------
* volume_m3
* edge_ratio = max_edge / min_edge
* radius_edge_ratio = circumradius / shortest_edge
* min_dihedral_deg (approximate; computed from face normals)

Usage
-----
    python tetgen_quality_report.py bartlesville_hec
or
    python tetgen_quality_report.py bartlesville_hec.1.node bartlesville_hec.1.ele

Outputs
-------
    <prefix>_quality_summary.csv
    <prefix>_quality_histogram.csv
    <prefix>_quality.vtu
    <prefix>_quality_bad_tets.vtu   (only if bad tets are found)

The VTU file is cell-centered: each tetrahedron carries its own quality data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_tetgen_node(path: Path) -> Dict[int, np.ndarray]:
    """Parse a TetGen .node file into a node-id -> xyz map."""
    nodes: Dict[int, np.ndarray] = {}
    header_read = False

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if not header_read:
                header_read = True
                if len(fields) < 4:
                    raise ValueError(f"Invalid TetGen node header in {path}")
                continue
            if len(fields) < 4:
                continue
            node_id = int(fields[0])
            nodes[node_id] = np.array([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)

    if not nodes:
        raise RuntimeError(f"No nodes parsed from {path}")
    return nodes


def parse_tetgen_ele(path: Path) -> List[Tuple[int, int, int, int]]:
    """Parse a TetGen .ele file into a list of tetrahedra."""
    elements: List[Tuple[int, int, int, int]] = []
    header_read = False

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if not header_read:
                header_read = True
                if len(fields) < 2:
                    raise ValueError(f"Invalid TetGen element header in {path}")
                continue
            if len(fields) < 5:
                continue
            # TetGen element lines are: id n1 n2 n3 n4 [region]
            elements.append(tuple(int(v) for v in fields[1:5]))

    if not elements:
        raise RuntimeError(f"No elements parsed from {path}")
    return elements


def tetra_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return abs(float(np.dot(b - a, np.cross(c - a, d - a)))) / 6.0


def edge_lengths(points: Sequence[np.ndarray]) -> List[float]:
    lengths: List[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            lengths.append(float(np.linalg.norm(points[i] - points[j])))
    return lengths


def tetra_circumradius(points: Sequence[np.ndarray]) -> float:
    """Return the circumradius of a tetrahedron.

    Uses a linear solve based on differences from one vertex. Returns inf for
    degenerate tets.
    """
    a, b, c, d = points
    m = np.vstack([b - a, c - a, d - a]).T
    rhs = 0.5 * np.array([
        float(np.dot(b, b) - np.dot(a, a)),
        float(np.dot(c, c) - np.dot(a, a)),
        float(np.dot(d, d) - np.dot(a, a)),
    ])
    try:
        center = np.linalg.solve(m, rhs)
    except np.linalg.LinAlgError:
        return math.inf
    return float(np.linalg.norm(center - a))


def triangle_normal(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = np.cross(q - p, r - p)
    norm = float(np.linalg.norm(n))
    if norm <= 0.0:
        return np.zeros(3, dtype=float)
    return n / norm


def min_dihedral_angle_deg(points: Sequence[np.ndarray]) -> float:
    """Compute the minimum dihedral angle in degrees.

    This uses face-normal angles. For a well-shaped tet this is a good proxy.
    """
    a, b, c, d = points
    faces = [
        triangle_normal(a, b, c),
        triangle_normal(a, d, b),
        triangle_normal(a, c, d),
        triangle_normal(b, d, c),
    ]
    if any(float(np.linalg.norm(n)) == 0.0 for n in faces):
        return 0.0

    # Opposite-face pairs for the 4 vertices
    pairs = [
        (faces[0], faces[1]),
        (faces[0], faces[2]),
        (faces[0], faces[3]),
        (faces[1], faces[2]),
        (faces[1], faces[3]),
        (faces[2], faces[3]),
    ]
    angles = []
    for n1, n2 in pairs:
        cosang = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        ang = math.degrees(math.acos(abs(cosang)))
        angles.append(ang)
    return min(angles) if angles else 0.0


def quality_bin(edge_ratio: float, min_dihedral: float) -> str:
    if edge_ratio > 50.0 or min_dihedral < 5.0:
        return "bad"
    if edge_ratio > 20.0 or min_dihedral < 10.0:
        return "poor"
    if edge_ratio > 10.0 or min_dihedral < 18.0:
        return "fair"
    return "good"


def vtk_cell_type_tet() -> int:
    return 10


def write_vtu(path: Path, nodes: Dict[int, np.ndarray], elements: List[Tuple[int, int, int, int]], qualities: List[dict]) -> None:
    """Write a minimal ASCII VTU file."""
    # Build a compact 0-based point array and remap node ids.
    node_ids = sorted(nodes)
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    points = np.vstack([nodes[nid] for nid in node_ids])

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(elements)}">\n')

        # Point data
        f.write('      <PointData/>\n')

        # Cell data
        f.write('      <CellData Scalars="edge_ratio">\n')
        for name in ("volume_m3", "edge_ratio", "radius_edge_ratio", "min_dihedral_deg", "quality_class"):
            pass
        # numeric arrays
        arrays = {
            "volume_m3": [q["volume_m3"] for q in qualities],
            "edge_ratio": [q["edge_ratio"] for q in qualities],
            "radius_edge_ratio": [q["radius_edge_ratio"] for q in qualities],
            "min_dihedral_deg": [q["min_dihedral_deg"] for q in qualities],
            "quality_code": [q["quality_code"] for q in qualities],
        }
        for name, vals in arrays.items():
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            f.write("          " + " ".join(f"{float(v):.16e}" for v in vals) + "\n")
            f.write("        </DataArray>\n")
        f.write('      </CellData>\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in points:
            f.write(f"          {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Cells
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for tet in elements:
            idxs = [id_to_idx[n] for n in tet]
            f.write("          " + " ".join(str(i) for i in idxs) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        off = 0
        for _ in elements:
            off += 4
            f.write(f"          {off}\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in elements:
            f.write(f"          {vtk_cell_type_tet()}\n")
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')

        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


def write_bad_tets_vtu(path: Path, nodes: Dict[int, np.ndarray], elements: List[Tuple[int, int, int, int]], qualities: List[dict], mask: List[bool]) -> None:
    bad_indices = [i for i, is_bad in enumerate(mask) if is_bad]
    if not bad_indices:
        return
    bad_elements = [elements[i] for i in bad_indices]
    bad_qualities = [qualities[i] for i in bad_indices]
    write_vtu(path, nodes, bad_elements, bad_qualities)


def write_summary_csv(path: Path, qualities: List[dict]) -> None:
    n = len(qualities)
    vols = np.array([q["volume_m3"] for q in qualities], dtype=float)
    ratios = np.array([q["edge_ratio"] for q in qualities], dtype=float)
    min_dihedrals = np.array([q["min_dihedral_deg"] for q in qualities], dtype=float)
    rratios = np.array([q["radius_edge_ratio"] for q in qualities], dtype=float)

    def pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p))

    rows = [
        ("count", n),
        ("volume_min_m3", float(np.min(vols))),
        ("volume_p5_m3", pct(vols, 5)),
        ("volume_median_m3", float(np.median(vols))),
        ("volume_p95_m3", pct(vols, 95)),
        ("volume_max_m3", float(np.max(vols))),
        ("edge_ratio_min", float(np.min(ratios))),
        ("edge_ratio_p5", pct(ratios, 5)),
        ("edge_ratio_median", float(np.median(ratios))),
        ("edge_ratio_p95", pct(ratios, 95)),
        ("edge_ratio_max", float(np.max(ratios))),
        ("radius_edge_ratio_min", float(np.min(rratios))),
        ("radius_edge_ratio_p5", pct(rratios, 5)),
        ("radius_edge_ratio_median", float(np.median(rratios))),
        ("radius_edge_ratio_p95", pct(rratios, 95)),
        ("radius_edge_ratio_max", float(np.max(rratios))),
        ("min_dihedral_min_deg", float(np.min(min_dihedrals))),
        ("min_dihedral_p5_deg", pct(min_dihedrals, 5)),
        ("min_dihedral_median_deg", float(np.median(min_dihedrals))),
        ("min_dihedral_p95_deg", pct(min_dihedrals, 95)),
        ("min_dihedral_max_deg", float(np.max(min_dihedrals))),
        ("bad_tets", sum(1 for q in qualities if q["quality_class"] == "bad")),
        ("poor_tets", sum(1 for q in qualities if q["quality_class"] == "poor")),
        ("fair_tets", sum(1 for q in qualities if q["quality_class"] == "fair")),
        ("good_tets", sum(1 for q in qualities if q["quality_class"] == "good")),
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in rows:
            writer.writerow([metric, value])


def write_histogram_csv(path: Path, qualities: List[dict], field: str, bins: List[float]) -> None:
    values = np.array([q[field] for q in qualities], dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "count"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([left, right, int(count)])


def build_quality_records(nodes: Dict[int, np.ndarray], elements: List[Tuple[int, int, int, int]]) -> Tuple[List[dict], List[bool]]:
    qualities: List[dict] = []
    bad_mask: List[bool] = []

    for tet in elements:
        pts = [nodes[n] for n in tet]
        vol = tetra_volume(*pts)
        lens = edge_lengths(pts)
        min_edge = min(lens)
        max_edge = max(lens)
        edge_ratio = math.inf if min_edge <= 0.0 else max_edge / min_edge
        circum = tetra_circumradius(pts)
        radius_edge_ratio = math.inf if min_edge <= 0.0 or not math.isfinite(circum) else circum / min_edge
        min_dih = min_dihedral_angle_deg(pts)

        qclass = quality_bin(edge_ratio, min_dih)
        is_bad = qclass in {"bad", "poor"}

        rec = {
            "volume_m3": vol,
            "edge_ratio": edge_ratio if math.isfinite(edge_ratio) else 1.0e30,
            "radius_edge_ratio": radius_edge_ratio if math.isfinite(radius_edge_ratio) else 1.0e30,
            "min_dihedral_deg": min_dih,
            "quality_class": qclass,
            "quality_code": {"good": 0.0, "fair": 1.0, "poor": 2.0, "bad": 3.0}[qclass],
        }
        qualities.append(rec)
        bad_mask.append(is_bad)

    return qualities, bad_mask



def resolve_inputs(prefix: str | None, node_arg: str | None, ele_arg: str | None) -> Tuple[Path, Path, str]:
    """Resolve input paths.

    Accepted forms:
      * prefix only -> prefix.1.node / prefix.1.ele
      * explicit --node and --ele -> use those paths
    """
    if node_arg or ele_arg:
        if not (node_arg and ele_arg):
            raise SystemExit("Provide both --node and --ele, or just a prefix.")
        node = Path(node_arg)
        ele = Path(ele_arg)
        mesh_prefix = prefix or node.name.replace(".1.node", "").replace(".node", "")
    else:
        if not prefix:
            raise SystemExit("Provide a mesh prefix, or both --node and --ele.")
        mesh_prefix = prefix
        node = Path(f"{mesh_prefix}.1.node")
        ele = Path(f"{mesh_prefix}.1.ele")

    if not node.exists():
        raise FileNotFoundError(node)
    if not ele.exists():
        raise FileNotFoundError(ele)
    return node, ele, mesh_prefix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TetGen tetrahedron quality and export CSV/VTK files."
    )
    parser.add_argument("prefix", nargs="?", help="Mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("--node", help="TetGen .node file")
    parser.add_argument("--ele", help="TetGen .ele file")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    node, ele, mesh_prefix = resolve_inputs(args.prefix, args.node, args.ele)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {node.name} and {ele.name}...")
    nodes = parse_tetgen_node(node)
    elements = parse_tetgen_ele(ele)
    qualities, bad_mask = build_quality_records(nodes, elements)

    summary_csv = outdir / f"{mesh_prefix}_quality_summary.csv"
    hist_csv = outdir / f"{mesh_prefix}_quality_histogram.csv"
    vtu = outdir / f"{mesh_prefix}_quality.vtu"
    bad_vtu = outdir / f"{mesh_prefix}_quality_bad_tets.vtu"

    write_summary_csv(summary_csv, qualities)
    write_histogram_csv(hist_csv, qualities, "edge_ratio", bins=[0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000])
    write_vtu(vtu, nodes, elements, qualities)
    if any(bad_mask):
        write_bad_tets_vtu(bad_vtu, nodes, elements, qualities, bad_mask)

    vols = np.array([q["volume_m3"] for q in qualities], dtype=float)
    ratios = np.array([q["edge_ratio"] for q in qualities], dtype=float)
    dih = np.array([q["min_dihedral_deg"] for q in qualities], dtype=float)

    print(f"nodes: {len(nodes):,}")
    print(f"elements: {len(elements):,}")
    print(f"min volume: {vols.min():.6e}")
    print(f"median edge ratio: {np.median(ratios):.3f}")
    print(f"max edge ratio: {ratios.max():.3f}")
    print(f"min dihedral angle: {dih.min():.3f} deg")
    print(f"poor/bad tets: {sum(bad_mask):,}")
    print(f"Wrote {summary_csv.name}")
    print(f"Wrote {hist_csv.name}")
    print(f"Wrote {vtu.name}")
    if any(bad_mask):
        print(f"Wrote {bad_vtu.name}")


if __name__ == "__main__":
    main()
