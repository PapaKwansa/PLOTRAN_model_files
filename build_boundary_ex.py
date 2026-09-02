#!/usr/bin/env python3
"""Build validated PFLOTRAN boundary ``.ex`` files for a node-centred median dual.

This utility is designed for the North Avant / Bartlesville workflow in which:

* TetGen supplies the primal tetrahedral mesh;
* LANL VORONOI with ``-cv median`` supplies one flow cell per TetGen vertex;
* the UGE cell ID equals the TetGen node-file row number (1 based);
* TetGen external-face markers identify top, bottom, north, south, east, west.

For a triangular exterior face, the barycentric/median boundary dual assigns one
third of the triangle area to each of its three vertices.  The script therefore
accumulates ``triangle_area / 3`` at each incident boundary vertex.  The sum of
all nodal boundary areas is required to equal the geometric area of the model
face.

A boundary flow cell is centred at the primal boundary vertex.  To avoid a zero
cell-to-boundary distance in PFLOTRAN, the .ex coordinate is moved a small
amount along the outward normal:

    epsilon_i = max(epsilon_min, epsilon_fraction * h_i)

where ``h_i`` is the shortest positive UGE centre-to-centre connection distance
for cell i.  The default epsilon fraction (1e-3) follows the scale used by LANL
DFNWorks boundary conversion, but the boundary area here is the actual median
nodal boundary area rather than a hard-coded constant.

Outputs (by default in the current directory):

    top.ex, bottom.ex, north.ex, south.ex, east.ex, west.ex
    boundary_ex_validation.csv
    boundary_ex_validation.json

The existing top.vset, ..., west.vset files are checked automatically when they
are present.  A set mismatch is a hard error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np


class BoundaryExError(RuntimeError):
    """Raised when boundary geometry or numbering is inconsistent."""


@dataclass(frozen=True)
class FaceSpec:
    name: str
    marker: int
    axis: int
    side: str
    normal: np.ndarray
    pressure_boundary: bool


# Marker convention written by build_poly_layers4.py.
DEFAULT_FACE_SPECS: tuple[FaceSpec, ...] = (
    FaceSpec("top", 1, 2, "max", np.array([0.0, 0.0, 1.0]), False),
    FaceSpec("bottom", 2, 2, "min", np.array([0.0, 0.0, -1.0]), False),
    FaceSpec("north", 3, 1, "max", np.array([0.0, 1.0, 0.0]), True),
    FaceSpec("south", 4, 1, "min", np.array([0.0, -1.0, 0.0]), True),
    FaceSpec("east", 5, 0, "max", np.array([1.0, 0.0, 0.0]), True),
    FaceSpec("west", 6, 0, "min", np.array([-1.0, 0.0, 0.0]), True),
)


def iter_data_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    domain = data.get("domain")
    if not isinstance(domain, Mapping):
        raise BoundaryExError(f"{path}: missing domain object")
    minimum = np.asarray(domain.get("min"), dtype=float)
    maximum = np.asarray(domain.get("max"), dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise BoundaryExError(f"{path}: domain.min/domain.max must each contain 3 values")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise BoundaryExError(f"{path}: non-finite domain bounds")
    if np.any(maximum <= minimum):
        raise BoundaryExError(f"{path}: invalid domain bounds")

    raw_markers = data.get("boundary_markers", {})
    markers = {str(k): int(v) for k, v in raw_markers.items()} if isinstance(raw_markers, Mapping) else {}
    return minimum, maximum, markers


def read_tetgen_nodes(path: Path) -> tuple[np.ndarray, Dict[int, int]]:
    lines = iter_data_lines(path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise BoundaryExError(f"Empty TetGen node file: {path}") from exc
    if len(header) < 4:
        raise BoundaryExError(f"Malformed node header in {path}")
    count = int(header[0])
    dimension = int(header[1])
    if count <= 0 or dimension != 3:
        raise BoundaryExError(f"{path}: expected positive 3-D node file")

    xyz = np.empty((count, 3), dtype=float)
    id_to_row: Dict[int, int] = {}
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise BoundaryExError(f"{path}: expected {count} node rows, stopped at {row}") from exc
        if len(fields) < 4:
            raise BoundaryExError(f"{path}: malformed node row {row + 1}")
        node_id = int(fields[0])
        if node_id in id_to_row:
            raise BoundaryExError(f"{path}: duplicate node ID {node_id}")
        point = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)
        if not np.all(np.isfinite(point)):
            raise BoundaryExError(f"{path}: non-finite coordinates for node {node_id}")
        id_to_row[node_id] = row
        xyz[row] = point
    return xyz, id_to_row


def parse_count_header(line: str, keyword: str, path: Path) -> int:
    fields = line.split()
    if len(fields) != 2 or fields[0].upper() != keyword:
        raise BoundaryExError(f"{path}: expected '{keyword} <integer>', got {line!r}")
    try:
        value = int(fields[1])
    except ValueError as exc:
        raise BoundaryExError(f"{path}: invalid {keyword} count") from exc
    if value < 0:
        raise BoundaryExError(f"{path}: negative {keyword} count")
    return value


def read_uge_and_local_spacing(
    path: Path,
    expected_xyz: np.ndarray,
    center_atol: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read UGE cell centres and compute shortest connected-centre distance per cell."""
    lines = iter_data_lines(path)
    try:
        cell_header = next(lines)
    except StopIteration as exc:
        raise BoundaryExError(f"Empty UGE file: {path}") from exc
    ncell = parse_count_header(cell_header, "CELLS", path)
    if ncell != expected_xyz.shape[0]:
        raise BoundaryExError(
            f"{path}: UGE cell count {ncell} != TetGen node count {expected_xyz.shape[0]}"
        )

    centers = np.empty((ncell, 3), dtype=float)
    seen = np.zeros(ncell, dtype=bool)
    for row in range(ncell):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise BoundaryExError(f"{path}: stopped in CELLS block at row {row}") from exc
        if len(fields) != 5:
            raise BoundaryExError(f"{path}: malformed cell row {row + 1}: {' '.join(fields)}")
        cell_id = int(fields[0])
        if cell_id < 1 or cell_id > ncell or seen[cell_id - 1]:
            raise BoundaryExError(f"{path}: invalid or duplicate cell ID {cell_id}")
        seen[cell_id - 1] = True
        centers[cell_id - 1] = [float(fields[1]), float(fields[2]), float(fields[3])]
    if not np.all(seen):
        raise BoundaryExError(f"{path}: missing cell IDs")

    max_center_difference = float(np.max(np.abs(centers - expected_xyz)))
    if max_center_difference > center_atol:
        raise BoundaryExError(
            f"{path}: max UGE/node coordinate mismatch {max_center_difference:.6e} m "
            f"> tolerance {center_atol:.6e} m"
        )

    try:
        connection_header = next(lines)
    except StopIteration as exc:
        raise BoundaryExError(f"{path}: missing CONNECTIONS block") from exc
    nconn = parse_count_header(connection_header, "CONNECTIONS", path)

    min_distance = np.full(ncell, np.inf, dtype=float)
    max_a_over_d = 0.0
    for row in range(nconn):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise BoundaryExError(f"{path}: stopped in CONNECTIONS block at row {row}") from exc
        if len(fields) != 6:
            raise BoundaryExError(f"{path}: malformed connection row {row + 1}")
        cell_a = int(fields[0])
        cell_b = int(fields[1])
        area = float(fields[5])
        if not (1 <= cell_a <= ncell and 1 <= cell_b <= ncell and cell_a != cell_b):
            raise BoundaryExError(f"{path}: invalid connection cells {cell_a}, {cell_b}")
        if not math.isfinite(area) or area <= 0.0:
            raise BoundaryExError(f"{path}: nonpositive/nonfinite area at connection row {row + 1}")
        distance = float(np.linalg.norm(centers[cell_a - 1] - centers[cell_b - 1]))
        if not math.isfinite(distance) or distance <= 0.0:
            raise BoundaryExError(f"{path}: nonpositive centre distance at row {row + 1}")
        if distance < min_distance[cell_a - 1]:
            min_distance[cell_a - 1] = distance
        if distance < min_distance[cell_b - 1]:
            min_distance[cell_b - 1] = distance
        max_a_over_d = max(max_a_over_d, area / distance)

    if not np.all(np.isfinite(min_distance)):
        missing = np.where(~np.isfinite(min_distance))[0] + 1
        raise BoundaryExError(f"{path}: {missing.size} cells have no positive-area neighbour")
    return centers, min_distance, max_a_over_d, nconn


def classify_external_triangle(
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    atol: float,
) -> str | None:
    matches: list[str] = []
    tests = (
        ("west", 0, minimum[0]),
        ("east", 0, maximum[0]),
        ("south", 1, minimum[1]),
        ("north", 1, maximum[1]),
        ("bottom", 2, minimum[2]),
        ("top", 2, maximum[2]),
    )
    for name, axis, value in tests:
        if np.all(np.isclose(points[:, axis], value, atol=atol, rtol=0.0)):
            matches.append(name)
    if len(matches) > 1:
        raise BoundaryExError(f"Degenerate triangle lies on multiple external planes: {matches}")
    return matches[0] if matches else None


def read_boundary_faces_and_accumulate_areas(
    face_path: Path,
    xyz: np.ndarray,
    id_to_row: Mapping[int, int],
    minimum: np.ndarray,
    maximum: np.ndarray,
    specs: Mapping[str, FaceSpec],
    coordinate_atol: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, float]]:
    lines = iter_data_lines(face_path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise BoundaryExError(f"Empty TetGen face file: {face_path}") from exc
    if len(header) < 2:
        raise BoundaryExError(f"Malformed face header in {face_path}")
    nfaces = int(header[0])
    marker_count = int(header[1])

    nodal_areas = {name: np.zeros(xyz.shape[0], dtype=float) for name in specs}
    triangle_counts = {name: 0 for name in specs}
    triangle_area_sums = {name: 0.0 for name in specs}

    for row in range(nfaces):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise BoundaryExError(f"{face_path}: expected {nfaces} face rows, stopped at {row}") from exc
        if len(fields) < 4:
            raise BoundaryExError(f"{face_path}: malformed face row {row + 1}")
        tetgen_ids = [int(fields[1]), int(fields[2]), int(fields[3])]
        try:
            rows = [id_to_row[node_id] for node_id in tetgen_ids]
        except KeyError as exc:
            raise BoundaryExError(f"{face_path}: unknown node ID {exc.args[0]}") from exc
        points = xyz[rows]
        face_name = classify_external_triangle(points, minimum, maximum, coordinate_atol)
        if face_name is None:
            continue

        spec = specs[face_name]
        if marker_count > 0 and len(fields) >= 5:
            marker = int(fields[4])
            # Marker 0 means unmarked; any nonzero conflicting marker is a hard error.
            if marker not in (0, spec.marker):
                raise BoundaryExError(
                    f"{face_path}: external {face_name} triangle row {row + 1} has marker "
                    f"{marker}, expected {spec.marker}"
                )

        cross = np.cross(points[1] - points[0], points[2] - points[0])
        area = 0.5 * float(np.linalg.norm(cross))
        if not math.isfinite(area) or area <= 0.0:
            raise BoundaryExError(f"{face_path}: nonpositive triangle area at row {row + 1}")

        share = area / 3.0
        for node_row in rows:
            nodal_areas[face_name][node_row] += share
        triangle_counts[face_name] += 1
        triangle_area_sums[face_name] += area

    return nodal_areas, triangle_counts, triangle_area_sums


def expected_face_area(name: str, minimum: np.ndarray, maximum: np.ndarray) -> float:
    dx, dy, dz = maximum - minimum
    if name in {"top", "bottom"}:
        return float(dx * dy)
    if name in {"north", "south"}:
        return float(dx * dz)
    if name in {"east", "west"}:
        return float(dy * dz)
    raise KeyError(name)


def read_vset(path: Path) -> set[int]:
    values: set[int] = set()
    for line in iter_data_lines(path):
        fields = line.split()
        if len(fields) != 1:
            raise BoundaryExError(f"{path}: expected one vertex ID per line")
        value = int(fields[0])
        if value in values:
            raise BoundaryExError(f"{path}: duplicate vertex ID {value}")
        values.add(value)
    return values


def ex_text(
    ids: np.ndarray,
    centers: np.ndarray,
    nodal_area: np.ndarray,
    local_spacing: np.ndarray,
    normal: np.ndarray,
    epsilon_fraction: float,
    epsilon_min: float,
    area_multiplier: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    spacings = local_spacing[ids - 1]
    epsilons = np.maximum(epsilon_min, epsilon_fraction * spacings)
    coordinates = centers[ids - 1] + epsilons[:, None] * normal[None, :]
    areas = nodal_area[ids - 1] * area_multiplier

    lines = [f"CONNECTIONS {ids.size}\n"]
    for cell_id, point, area in zip(ids, coordinates, areas):
        lines.append(
            f"{int(cell_id)} "
            f"{point[0]:.12e} {point[1]:.12e} {point[2]:.12e} "
            f"{area:.12e}\n"
        )
    return "".join(lines), epsilons, areas, coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate area-consistent PFLOTRAN .ex files from TetGen exterior faces."
    )
    parser.add_argument("prefix", help="Mesh prefix, e.g. bartlesville_hec_lime_v5_interfaces")
    parser.add_argument("--node", type=Path, help="TetGen .node file; default <prefix>.1.node")
    parser.add_argument("--face", type=Path, help="TetGen .face file; default <prefix>.1.face")
    parser.add_argument("--uge", type=Path, required=True, help="Validated median-control-volume UGE")
    parser.add_argument(
        "--geometry-json", type=Path, help="Geometry metadata; default <prefix>_geometry.json"
    )
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument(
        "--epsilon-fraction",
        type=float,
        default=1.0e-3,
        help="Boundary offset as fraction of shortest connected-centre distance; default 1e-3",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=1.0e-6,
        help="Minimum outward offset in metres; default 1e-6",
    )
    parser.add_argument(
        "--pressure-area-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for north/south/east/west areas; default 1 (physical area)",
    )
    parser.add_argument(
        "--noflow-area-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for top/bottom areas; default 1",
    )
    parser.add_argument(
        "--coordinate-atol", type=float, default=1.0e-6, help="Coordinate tolerance [m]"
    )
    parser.add_argument(
        "--area-closure-rtol",
        type=float,
        default=1.0e-10,
        help="Maximum relative exterior-face area mismatch; default 1e-10",
    )
    parser.add_argument(
        "--skip-vset-check",
        action="store_true",
        help="Do not compare generated boundary cell IDs with existing <face>.vset files",
    )
    args = parser.parse_args()

    for name in (
        "epsilon_fraction",
        "epsilon_min",
        "pressure_area_multiplier",
        "noflow_area_multiplier",
        "coordinate_atol",
        "area_closure_rtol",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.epsilon_fraction == 0.0 and args.epsilon_min == 0.0:
        parser.error("epsilon fraction and minimum cannot both be zero")
    if args.pressure_area_multiplier <= 0.0 or args.noflow_area_multiplier <= 0.0:
        parser.error("area multipliers must be strictly positive")
    return args


def main() -> None:
    args = parse_args()
    prefix = args.prefix.removesuffix(".poly")
    node_path = args.node or Path(f"{prefix}.1.node")
    face_path = args.face or Path(f"{prefix}.1.face")
    geometry_path = args.geometry_json or Path(f"{prefix}_geometry.json")

    for path in (node_path, face_path, args.uge, geometry_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Reading geometry metadata: {geometry_path}")
    minimum, maximum, marker_overrides = read_geometry(geometry_path)

    specs: Dict[str, FaceSpec] = {}
    for default in DEFAULT_FACE_SPECS:
        marker = marker_overrides.get(default.name, default.marker)
        specs[default.name] = FaceSpec(
            default.name,
            marker,
            default.axis,
            default.side,
            default.normal,
            default.pressure_boundary,
        )

    print(f"Reading TetGen nodes: {node_path}")
    xyz, id_to_row = read_tetgen_nodes(node_path)

    print(f"Reading validated median UGE: {args.uge}")
    centers, local_spacing, max_internal_a_over_d, connection_count = read_uge_and_local_spacing(
        args.uge, xyz, args.coordinate_atol
    )

    print(f"Accumulating median-dual boundary areas from: {face_path}")
    nodal_areas, triangle_counts, triangle_area_sums = read_boundary_faces_and_accumulate_areas(
        face_path,
        xyz,
        id_to_row,
        minimum,
        maximum,
        specs,
        args.coordinate_atol,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []

    for name in ("top", "bottom", "north", "south", "east", "west"):
        spec = specs[name]
        areas_raw = nodal_areas[name]
        ids = np.where(areas_raw > 0.0)[0].astype(np.int64) + 1
        if ids.size == 0:
            raise BoundaryExError(f"No external triangles/cells found for {name}")

        geometric_area = float(np.sum(areas_raw[ids - 1], dtype=np.float64))
        triangle_area = float(triangle_area_sums[name])
        expected_area = expected_face_area(name, minimum, maximum)
        rel_error = abs(geometric_area - expected_area) / expected_area
        triangle_vs_nodal = abs(triangle_area - geometric_area) / expected_area
        if rel_error > args.area_closure_rtol:
            raise BoundaryExError(
                f"{name}: nodal boundary area {geometric_area:.16e} differs from expected "
                f"{expected_area:.16e} by {rel_error:.6e}"
            )
        if triangle_vs_nodal > args.area_closure_rtol:
            raise BoundaryExError(f"{name}: triangle-area and nodal-area sums do not close")

        vset_status = "not_checked"
        vset_path = Path(f"{name}.vset")
        if not args.skip_vset_check and vset_path.is_file():
            expected_ids = read_vset(vset_path)
            generated_ids = set(int(value) for value in ids)
            if expected_ids != generated_ids:
                missing = sorted(generated_ids - expected_ids)[:10]
                extra = sorted(expected_ids - generated_ids)[:10]
                raise BoundaryExError(
                    f"{vset_path}: boundary-node set mismatch for {name}; "
                    f"missing_in_vset={missing}, extra_in_vset={extra}"
                )
            vset_status = "matched"
        elif not args.skip_vset_check:
            vset_status = "file_missing"

        multiplier = (
            args.pressure_area_multiplier
            if spec.pressure_boundary
            else args.noflow_area_multiplier
        )
        text, epsilons, output_areas, _ = ex_text(
            ids,
            centers,
            areas_raw,
            local_spacing,
            spec.normal,
            args.epsilon_fraction,
            args.epsilon_min,
            multiplier,
        )
        output_path = args.outdir / f"{name}.ex"
        atomic_write_text(output_path, text)

        a_over_eps = output_areas / epsilons
        min_boundary_ratio = float(np.min(a_over_eps) / max_internal_a_over_d)
        report_rows.append(
            {
                "face": name,
                "marker": spec.marker,
                "boundary_type": "pressure" if spec.pressure_boundary else "noflow",
                "triangle_count": triangle_counts[name],
                "cell_count": int(ids.size),
                "expected_area_m2": expected_area,
                "geometric_nodal_area_m2": geometric_area,
                "output_area_m2": float(np.sum(output_areas, dtype=np.float64)),
                "area_relative_error": rel_error,
                "area_multiplier": multiplier,
                "nodal_area_min_m2": float(np.min(areas_raw[ids - 1])),
                "nodal_area_median_m2": float(np.median(areas_raw[ids - 1])),
                "nodal_area_max_m2": float(np.max(areas_raw[ids - 1])),
                "epsilon_min_m": float(np.min(epsilons)),
                "epsilon_median_m": float(np.median(epsilons)),
                "epsilon_max_m": float(np.max(epsilons)),
                "min_boundary_A_over_epsilon_m": float(np.min(a_over_eps)),
                "max_internal_A_over_distance_m": max_internal_a_over_d,
                "min_boundary_to_max_internal_coefficient_ratio": min_boundary_ratio,
                "vset_status": vset_status,
                "output_file": str(output_path),
            }
        )
        print(
            f"  {name:6s}: triangles={triangle_counts[name]:,}, cells={ids.size:,}, "
            f"area={geometric_area:.9e} m^2, epsilon="
            f"[{np.min(epsilons):.3e}, {np.max(epsilons):.3e}] m, "
            f"vset={vset_status}"
        )

    csv_path = args.outdir / "boundary_ex_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    report = {
        "status": "passed",
        "prefix": prefix,
        "node_file": str(node_path),
        "face_file": str(face_path),
        "uge_file": str(args.uge),
        "geometry_json": str(geometry_path),
        "node_count": int(xyz.shape[0]),
        "uge_connection_count": int(connection_count),
        "domain_min_m": minimum.tolist(),
        "domain_max_m": maximum.tolist(),
        "epsilon_fraction": args.epsilon_fraction,
        "epsilon_min_m": args.epsilon_min,
        "pressure_area_multiplier": args.pressure_area_multiplier,
        "noflow_area_multiplier": args.noflow_area_multiplier,
        "max_internal_A_over_distance_m": max_internal_a_over_d,
        "faces": report_rows,
    }
    json_path = args.outdir / "boundary_ex_validation.json"
    atomic_write_text(json_path, json.dumps(report, indent=2) + "\n")

    print("\nBoundary .ex generation and validation passed")
    print(f"  flow cells                  : {xyz.shape[0]:,}")
    print(f"  internal UGE connections    : {connection_count:,}")
    print(f"  max internal A/d            : {max_internal_a_over_d:.6e} m")
    print(f"  validation CSV              : {csv_path}")
    print(f"  validation JSON             : {json_path}")


if __name__ == "__main__":
    main()
