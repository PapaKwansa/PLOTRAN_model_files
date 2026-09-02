#!/usr/bin/env python3
"""Validate/sanitize a LANL-VORONOI PFLOTRAN UGE and write a mapping.

This utility is intended for a node-centred Delaunay/Voronoi workflow in which
LANL VORONOI creates one PFLOTRAN flow cell per TetGen vertex and the
geomechanics UGI uses those same vertices in the same row order.

Why zero-area connections are handled specially
-------------------------------------------------
LANL VORONOI writes one connection record for each represented Delaunay edge.
For co-spherical or otherwise degenerate point configurations, an edge can have
an exactly collapsed Voronoi dual face and therefore an area of zero. Such a
record contributes no transmissive face and should not be passed to PFLOTRAN as
an active connection. This script drops only areas less than or equal to the
user-selected threshold (zero by default), writes a sanitized UGE, and then
requires the remaining positive-area graph to be connected and free of isolated
cells.

Hard validation checks
----------------------
* UGE cell count equals the TetGen/UGI vertex count.
* UGE cell IDs are exactly 1..N with no duplicates or gaps.
* UGE cell centres match TetGen node-file row order.
* UGI vertices match the same TetGen node-file row order.
* All cell volumes are finite and strictly positive.
* Connection face points are finite and cell references are valid/distinct.
* Negative or non-finite connection areas are rejected.
* Connections with area <= --drop-area-le are removed from the active UGE.
* All retained connection areas and centre distances are strictly positive.
* No duplicate retained undirected cell-pair connections exist unless allowed.
* The retained connection graph is connected and contains no isolated cells.
* Optional material-HDF5 arrays use the same 1..N cell IDs and row count.

Outputs are written only after all hard checks pass:
* sanitized UGE (when zero/negligible connections exist, or when requested),
* <mesh>.mapping      -- flow-cell ID -> geomechanics-vertex ID,
* <mesh>_all.vset     -- all geomechanics vertex IDs,
* <mesh>_uge_validation.txt,
* <mesh>_uge_validation.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class ValidationError(RuntimeError):
    """Raised when a mesh consistency check fails."""


@dataclass(frozen=True)
class UGEData:
    centers: np.ndarray
    volumes: np.ndarray
    cell_a: np.ndarray
    cell_b: np.ndarray
    areas: np.ndarray
    center_distances: np.ndarray
    area_over_distance: np.ndarray
    bisector_relative_error: np.ndarray
    degree: np.ndarray
    component_count: int
    duplicate_pair_count: int
    original_connection_count: int
    retained_connection_count: int
    dropped_connection_count: int
    drop_area_le_m2: float
    keep_mask: np.ndarray
    dropped_samples: tuple[dict[str, object], ...]


def iter_data_lines(path: Path) -> Iterator[str]:
    """Yield non-empty, non-comment lines with inline comments removed."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def parse_header(line: str, keyword: str, path: Path) -> int:
    fields = line.split()
    if len(fields) != 2 or fields[0].upper() != keyword:
        raise ValidationError(
            f"{path}: expected '{keyword} <integer>', received {line!r}."
        )
    try:
        count = int(fields[1])
    except ValueError as exc:
        raise ValidationError(f"{path}: invalid {keyword} count in {line!r}.") from exc
    if count < 0:
        raise ValidationError(f"{path}: negative {keyword} count {count}.")
    return count


def read_tetgen_nodes(path: Path) -> np.ndarray:
    lines = iter_data_lines(path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise ValidationError(f"Empty TetGen node file: {path}") from exc
    if len(header) < 4:
        raise ValidationError(f"Malformed TetGen node header in {path}.")

    count = int(header[0])
    dimension = int(header[1])
    if count <= 0 or dimension != 3:
        raise ValidationError(
            f"{path}: expected a positive 3-D node file; "
            f"count={count}, dimension={dimension}."
        )

    xyz = np.empty((count, 3), dtype=np.float64)
    seen_ids: set[int] = set()
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise ValidationError(
                f"{path}: expected {count} node rows, stopped at {row}."
            ) from exc
        if len(fields) < 4:
            raise ValidationError(f"{path}: malformed node row {row + 1}.")
        node_id = int(fields[0])
        if node_id in seen_ids:
            raise ValidationError(f"{path}: duplicate TetGen node ID {node_id}.")
        seen_ids.add(node_id)
        xyz[row] = (float(fields[1]), float(fields[2]), float(fields[3]))

    if not np.isfinite(xyz).all():
        raise ValidationError(f"{path}: non-finite node coordinates detected.")
    return xyz


def read_ugi_vertices(path: Path) -> tuple[int, np.ndarray]:
    lines = iter_data_lines(path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise ValidationError(f"Empty UGI file: {path}") from exc
    if len(header) != 2:
        raise ValidationError(f"Malformed UGI header in {path}: {' '.join(header)!r}")

    element_count = int(header[0])
    node_count = int(header[1])
    if element_count <= 0 or node_count <= 0:
        raise ValidationError(
            f"{path}: non-positive UGI counts: "
            f"elements={element_count}, nodes={node_count}."
        )

    for row in range(element_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise ValidationError(
                f"{path}: expected {element_count} element rows, stopped at {row}."
            ) from exc
        if not fields or fields[0].upper() not in {"T", "P", "W", "H"}:
            raise ValidationError(f"{path}: malformed UGI element row {row + 1}.")

    xyz = np.empty((node_count, 3), dtype=np.float64)
    for row in range(node_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise ValidationError(
                f"{path}: expected {node_count} vertex rows, stopped at {row}."
            ) from exc
        if len(fields) != 3:
            raise ValidationError(f"{path}: malformed UGI vertex row {row + 1}.")
        xyz[row] = tuple(float(value) for value in fields)

    if not np.isfinite(xyz).all():
        raise ValidationError(f"{path}: non-finite UGI vertex coordinates detected.")
    return element_count, xyz


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def union(self, first: int, second: int) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        rank = self.rank
        parent = self.parent
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    def component_count(self) -> int:
        roots = {self.find(index) for index in range(self.parent.size)}
        return len(roots)


def read_uge(
    path: Path,
    allow_duplicate_connections: bool,
    drop_area_le_m2: float,
    sample_limit: int,
) -> UGEData:
    """Read a UGE, remove zero/negligible-area records in memory, and validate."""
    lines = iter_data_lines(path)
    try:
        cell_header = next(lines)
    except StopIteration as exc:
        raise ValidationError(f"Empty UGE file: {path}") from exc
    cell_count = parse_header(cell_header, "CELLS", path)
    if cell_count <= 0:
        raise ValidationError(f"{path}: UGE must contain at least one cell.")

    centers = np.empty((cell_count, 3), dtype=np.float64)
    volumes = np.empty(cell_count, dtype=np.float64)
    seen = np.zeros(cell_count, dtype=bool)

    for row in range(cell_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise ValidationError(
                f"{path}: expected {cell_count} cell rows, stopped at {row}."
            ) from exc
        if len(fields) != 5:
            raise ValidationError(
                f"{path}: UGE cell row {row + 1} must have 5 fields, "
                f"got {len(fields)}."
            )
        cell_id = int(fields[0])
        if not 1 <= cell_id <= cell_count:
            raise ValidationError(
                f"{path}: cell ID {cell_id} outside valid range 1..{cell_count}."
            )
        index = cell_id - 1
        if seen[index]:
            raise ValidationError(f"{path}: duplicate cell ID {cell_id}.")
        seen[index] = True
        centers[index] = (float(fields[1]), float(fields[2]), float(fields[3]))
        volumes[index] = float(fields[4])

    if not seen.all():
        missing = np.where(~seen)[0][:20] + 1
        raise ValidationError(
            f"{path}: missing cell IDs; first missing IDs: {missing.tolist()}"
        )
    if not np.isfinite(centers).all():
        raise ValidationError(f"{path}: non-finite cell centers detected.")
    if not np.isfinite(volumes).all() or np.any(volumes <= 0.0):
        bad = np.where(~np.isfinite(volumes) | (volumes <= 0.0))[0][:20] + 1
        raise ValidationError(
            f"{path}: cell volumes must be finite and positive; "
            f"first bad cell IDs: {bad.tolist()}"
        )

    try:
        connection_header = next(lines)
    except StopIteration as exc:
        raise ValidationError(f"{path}: missing CONNECTIONS header.") from exc
    original_count = parse_header(connection_header, "CONNECTIONS", path)
    if original_count <= 0:
        raise ValidationError(f"{path}: UGE must contain at least one connection.")

    keep_mask = np.ones(original_count, dtype=bool)
    cell_a_all = np.empty(original_count, dtype=np.int32)
    cell_b_all = np.empty(original_count, dtype=np.int32)
    areas_all = np.empty(original_count, dtype=np.float64)
    distances_all = np.empty(original_count, dtype=np.float64)
    geometry_factor_all = np.empty(original_count, dtype=np.float64)
    bisector_error_all = np.empty(original_count, dtype=np.float64)
    degree = np.zeros(cell_count, dtype=np.int64)
    union_find = UnionFind(cell_count)
    dropped_samples: list[dict[str, object]] = []

    for row in range(original_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise ValidationError(
                f"{path}: expected {original_count} connection rows, "
                f"stopped at {row}."
            ) from exc
        if len(fields) != 6:
            raise ValidationError(
                f"{path}: connection row {row + 1} must have 6 fields, "
                f"got {len(fields)}."
            )

        a_id = int(fields[0])
        b_id = int(fields[1])
        if not (1 <= a_id <= cell_count and 1 <= b_id <= cell_count):
            raise ValidationError(
                f"{path}: connection {row + 1} references IDs ({a_id}, {b_id}) "
                f"outside 1..{cell_count}."
            )
        if a_id == b_id:
            raise ValidationError(
                f"{path}: self-connection at row {row + 1}, cell {a_id}."
            )

        face_center = np.array(
            [float(fields[2]), float(fields[3]), float(fields[4])],
            dtype=np.float64,
        )
        area = float(fields[5])
        if not np.isfinite(face_center).all():
            raise ValidationError(
                f"{path}: non-finite face point at connection row {row + 1}."
            )
        if not math.isfinite(area):
            raise ValidationError(
                f"{path}: non-finite connection area at row {row + 1}: {area}."
            )
        if area < 0.0:
            raise ValidationError(
                f"{path}: negative connection area at row {row + 1}: {area}."
            )

        a_index = a_id - 1
        b_index = b_id - 1
        separation = float(np.linalg.norm(centers[b_index] - centers[a_index]))
        if not math.isfinite(separation) or separation <= 0.0:
            raise ValidationError(
                f"{path}: coincident/non-finite connected cell centers "
                f"at row {row + 1}."
            )

        distance_a = float(np.linalg.norm(face_center - centers[a_index]))
        distance_b = float(np.linalg.norm(face_center - centers[b_index]))
        scale = max(distance_a, distance_b, separation, np.finfo(np.float64).tiny)

        cell_a_all[row] = a_id
        cell_b_all[row] = b_id
        areas_all[row] = area
        distances_all[row] = separation
        geometry_factor_all[row] = area / separation
        bisector_error_all[row] = abs(distance_a - distance_b) / scale

        if area <= drop_area_le_m2:
            keep_mask[row] = False
            if len(dropped_samples) < sample_limit:
                dropped_samples.append(
                    {
                        "connection_row_1based": row + 1,
                        "cell_a": a_id,
                        "cell_b": b_id,
                        "face_point_m": face_center.tolist(),
                        "area_m2": area,
                        "center_distance_m": separation,
                    }
                )
            continue

        degree[a_index] += 1
        degree[b_index] += 1
        union_find.union(a_index, b_index)

    try:
        extra = next(lines)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValidationError(f"{path}: unexpected data after final connection: {extra!r}")

    retained_count = int(np.count_nonzero(keep_mask))
    dropped_count = original_count - retained_count
    if retained_count <= 0:
        raise ValidationError(
            f"{path}: no positive-area connections remain after applying "
            f"area <= {drop_area_le_m2:.6e} m^2 filter."
        )

    cell_a = cell_a_all[keep_mask]
    cell_b = cell_b_all[keep_mask]
    areas = areas_all[keep_mask]
    distances = distances_all[keep_mask]
    geometry_factor = geometry_factor_all[keep_mask]
    bisector_error = bisector_error_all[keep_mask]

    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValidationError(
            f"{path}: internal error: retained connection areas are not all positive."
        )

    isolated = np.where(degree == 0)[0]
    if isolated.size:
        raise ValidationError(
            f"{path}: after dropping {dropped_count} zero/negligible-area records, "
            f"{isolated.size} cells are isolated; first IDs: "
            f"{(isolated[:20] + 1).tolist()}"
        )

    component_count = union_find.component_count()
    if component_count != 1:
        raise ValidationError(
            f"{path}: positive-area connection graph has "
            f"{component_count} disconnected components after dropping "
            f"{dropped_count} records."
        )

    low = np.minimum(cell_a, cell_b).astype(np.int64)
    high = np.maximum(cell_a, cell_b).astype(np.int64)
    pair_keys = low * np.int64(cell_count + 1) + high
    pair_keys.sort()
    duplicate_pair_count = int(np.count_nonzero(pair_keys[1:] == pair_keys[:-1]))
    if duplicate_pair_count and not allow_duplicate_connections:
        raise ValidationError(
            f"{path}: found {duplicate_pair_count} duplicate retained undirected "
            "cell-pair connections. Rerun with --allow-duplicate-connections only "
            "after confirming they are intentional."
        )

    return UGEData(
        centers=centers,
        volumes=volumes,
        cell_a=cell_a,
        cell_b=cell_b,
        areas=areas,
        center_distances=distances,
        area_over_distance=geometry_factor,
        bisector_relative_error=bisector_error,
        degree=degree,
        component_count=component_count,
        duplicate_pair_count=duplicate_pair_count,
        original_connection_count=original_count,
        retained_connection_count=retained_count,
        dropped_connection_count=dropped_count,
        drop_area_le_m2=drop_area_le_m2,
        keep_mask=keep_mask,
        dropped_samples=tuple(dropped_samples),
    )


def validate_material_h5(path: Path, expected_count: int) -> dict[str, object]:
    try:
        import h5py
    except ImportError as exc:
        raise ValidationError("h5py is required when --material-h5 is supplied.") from exc

    with h5py.File(path, "r") as handle:
        for dataset in ("/Materials/Cell Ids", "/Materials/Material Ids"):
            if dataset not in handle:
                raise ValidationError(f"{path}: missing dataset {dataset}.")
        cell_ids = np.asarray(handle["/Materials/Cell Ids"])
        material_ids = np.asarray(handle["/Materials/Material Ids"])

    if cell_ids.ndim != 1 or material_ids.ndim != 1:
        raise ValidationError(f"{path}: material datasets must be one-dimensional.")
    if cell_ids.size != expected_count or material_ids.size != expected_count:
        raise ValidationError(
            f"{path}: expected {expected_count} material rows, got "
            f"cell_ids={cell_ids.size}, material_ids={material_ids.size}."
        )
    expected_ids = np.arange(1, expected_count + 1, dtype=cell_ids.dtype)
    if not np.array_equal(cell_ids, expected_ids):
        raise ValidationError(f"{path}: /Materials/Cell Ids is not exactly 1..N.")
    unique, counts = np.unique(material_ids.astype(np.int64), return_counts=True)
    return {
        "path": str(path),
        "material_distribution": {
            str(int(material)): int(count)
            for material, count in zip(unique, counts)
        },
    }


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def percentile_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_filtered_uge(source: Path, target: Path, keep_mask: np.ndarray) -> None:
    """Copy a UGE while removing connection rows whose keep mask is false."""
    if source.resolve() == target.resolve():
        raise ValidationError(
            "Refusing to overwrite the source UGE in place. Choose a distinct "
            "--clean-uge path."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent), text=True
    )
    try:
        lines = iter_data_lines(source)
        cell_header = next(lines)
        cell_count = parse_header(cell_header, "CELLS", source)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"CELLS {cell_count}\n")
            for _ in range(cell_count):
                handle.write(next(lines) + "\n")

            connection_header = next(lines)
            original_count = parse_header(connection_header, "CONNECTIONS", source)
            if original_count != keep_mask.size:
                raise ValidationError(
                    "Internal error: UGE connection count changed between read and write."
                )
            retained_count = int(np.count_nonzero(keep_mask))
            handle.write(f"CONNECTIONS {retained_count}\n")
            for row in range(original_count):
                line = next(lines)
                if bool(keep_mask[row]):
                    handle.write(line + "\n")

            try:
                extra = next(lines)
            except StopIteration:
                extra = None
            if extra is not None:
                raise ValidationError(
                    f"{source}: unexpected data after final connection: {extra!r}"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def mapping_text(count: int) -> str:
    return "".join(f"{index} {index}\n" for index in range(1, count + 1))


def vset_text(count: int) -> str:
    return "".join(f"{index}\n" for index in range(1, count + 1))


def build_report(
    args: argparse.Namespace,
    uge: UGEData,
    effective_uge: Path,
    node_xyz: np.ndarray,
    ugi_element_count: int,
    max_center_difference: float,
    max_ugi_difference: float,
    material_info: dict[str, object] | None,
    domain_info: dict[str, object] | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "status": "passed",
        "input_uge_file": str(args.uge),
        "effective_uge_file": str(effective_uge),
        "node_file": str(args.node),
        "ugi_file": str(args.ugi),
        "mapping_file": str(args.mapping),
        "all_vset_file": str(args.all_vset),
        "cell_count": int(uge.centers.shape[0]),
        "original_connection_count": int(uge.original_connection_count),
        "retained_connection_count": int(uge.retained_connection_count),
        "dropped_connection_count": int(uge.dropped_connection_count),
        "drop_area_le_m2": float(uge.drop_area_le_m2),
        "dropped_connection_samples": list(uge.dropped_samples),
        "geomechanics_element_count": int(ugi_element_count),
        "geomechanics_vertex_count": int(node_xyz.shape[0]),
        "max_uge_center_vs_node_difference_m": float(max_center_difference),
        "max_ugi_vertex_vs_node_difference_m": float(max_ugi_difference),
        "cell_volume_m3": percentile_stats(uge.volumes),
        "retained_connection_area_m2": percentile_stats(uge.areas),
        "center_distance_m": percentile_stats(uge.center_distances),
        "area_over_distance_m": percentile_stats(uge.area_over_distance),
        "bisector_relative_error": percentile_stats(uge.bisector_relative_error),
        "retained_connection_degree": percentile_stats(
            uge.degree.astype(np.float64)
        ),
        "connection_graph_components": int(uge.component_count),
        "duplicate_connection_pairs": int(uge.duplicate_pair_count),
        "total_cell_volume_m3": float(np.sum(uge.volumes, dtype=np.float64)),
        "input_uge_sha256": file_sha256(args.uge),
        "effective_uge_sha256": file_sha256(effective_uge),
        "node_sha256": file_sha256(args.node),
        "ugi_sha256": file_sha256(args.ugi),
    }
    if material_info is not None:
        report["material_h5"] = material_info
    if domain_info is not None:
        report["domain"] = domain_info
    return report


def text_report(report: dict[str, object]) -> str:
    volume = report["cell_volume_m3"]
    area = report["retained_connection_area_m2"]
    distance = report["center_distance_m"]
    factor = report["area_over_distance_m"]
    bisector = report["bisector_relative_error"]
    degree = report["retained_connection_degree"]

    lines = [
        "PFLOTRAN UGE / geomechanics mapping validation passed",
        f"input_uge_file={report['input_uge_file']}",
        f"effective_uge_file={report['effective_uge_file']}",
        f"node_file={report['node_file']}",
        f"ugi_file={report['ugi_file']}",
        f"mapping_file={report['mapping_file']}",
        f"all_vset_file={report['all_vset_file']}",
        f"cell_count={report['cell_count']}",
        f"original_connection_count={report['original_connection_count']}",
        f"retained_connection_count={report['retained_connection_count']}",
        f"dropped_connection_count={report['dropped_connection_count']}",
        f"drop_area_le_m2={float(report['drop_area_le_m2']):.16e}",
        f"geomechanics_element_count={report['geomechanics_element_count']}",
        f"geomechanics_vertex_count={report['geomechanics_vertex_count']}",
        f"max_uge_center_vs_node_difference_m={report['max_uge_center_vs_node_difference_m']:.16e}",
        f"max_ugi_vertex_vs_node_difference_m={report['max_ugi_vertex_vs_node_difference_m']:.16e}",
        f"connection_graph_components={report['connection_graph_components']}",
        f"duplicate_connection_pairs={report['duplicate_connection_pairs']}",
        f"total_cell_volume_m3={report['total_cell_volume_m3']:.16e}",
    ]

    for prefix, stats in (
        ("cell_volume_m3", volume),
        ("retained_connection_area_m2", area),
        ("center_distance_m", distance),
        ("area_over_distance_m", factor),
        ("bisector_relative_error", bisector),
        ("retained_connection_degree", degree),
    ):
        for key in ("min", "p01", "p05", "median", "p95", "p99", "max"):
            lines.append(f"{prefix}_{key}={float(stats[key]):.16e}")

    samples = report.get("dropped_connection_samples", [])
    if isinstance(samples, list):
        for index, sample in enumerate(samples, start=1):
            lines.append(
                "dropped_sample_{}={}".format(
                    index,
                    json.dumps(sample, sort_keys=True, separators=(",", ":")),
                )
            )

    domain = report.get("domain")
    if isinstance(domain, dict):
        lines.extend(
            [
                f"expected_domain_volume_m3={float(domain['expected_volume_m3']):.16e}",
                f"domain_volume_relative_error={float(domain['relative_error']):.16e}",
            ]
        )

    material = report.get("material_h5")
    if isinstance(material, dict):
        distribution = material.get("material_distribution", {})
        if isinstance(distribution, dict):
            compact = ",".join(
                f"{key}:{value}" for key, value in distribution.items()
            )
            lines.append(f"material_distribution={compact}")

    lines.extend(
        [
            f"input_uge_sha256={report['input_uge_sha256']}",
            f"effective_uge_sha256={report['effective_uge_sha256']}",
            f"node_sha256={report['node_sha256']}",
            f"ugi_sha256={report['ugi_sha256']}",
            "mapping_contract=flow_cell_i_maps_to_geomechanics_vertex_i",
            "mapping_ids=contiguous_1_based",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a LANL-VORONOI PFLOTRAN UGE, remove exact zero-area "
            "dual connections, and write a verified identity mapping."
        )
    )
    parser.add_argument("uge", type=Path, help="Original PFLOTRAN ASCII .uge file")
    parser.add_argument("node", type=Path, help="TetGen .node file")
    parser.add_argument("ugi", type=Path, help="Canonical PFLOTRAN geomechanics .ugi file")
    parser.add_argument(
        "--clean-uge",
        type=Path,
        help=(
            "Sanitized output UGE. Required when connections are dropped; "
            "recommended name: <stem>_clean.uge"
        ),
    )
    parser.add_argument(
        "--drop-area-le",
        type=float,
        default=0.0,
        help=(
            "Drop connection records with area <= this value [m^2]. "
            "Default 0 removes exact zero-area records only."
        ),
    )
    parser.add_argument(
        "--dropped-sample-limit",
        type=int,
        default=20,
        help="Maximum dropped-connection samples written to reports; default 20",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Output mapping file; default: <effective UGE stem>.mapping",
    )
    parser.add_argument(
        "--all-vset",
        type=Path,
        help="Output all-vertex vset; default: <effective UGE stem>_all.vset",
    )
    parser.add_argument(
        "--geometry-json",
        type=Path,
        help="Optional geometry JSON used to check total domain-volume closure",
    )
    parser.add_argument(
        "--material-h5",
        type=Path,
        help="Optional material HDF5 to validate against UGE cell IDs/count",
    )
    parser.add_argument(
        "--center-atol",
        type=float,
        default=1.0e-6,
        help="Maximum allowed UGE/UGI coordinate mismatch [m]; default 1e-6",
    )
    parser.add_argument(
        "--domain-volume-warn-rtol",
        type=float,
        default=1.0e-6,
        help="Relative domain-volume mismatch that triggers a warning; default 1e-6",
    )
    parser.add_argument(
        "--domain-volume-fail-rtol",
        type=float,
        default=1.0e-3,
        help="Relative domain-volume mismatch that fails validation; default 1e-3",
    )
    parser.add_argument(
        "--allow-duplicate-connections",
        action="store_true",
        help="Report rather than fail on duplicate retained undirected cell pairs",
    )
    parser.add_argument(
        "--report-prefix",
        type=Path,
        help="Output report prefix; default: <effective UGE stem>_uge_validation",
    )
    args = parser.parse_args()

    if args.drop_area_le < 0.0 or not math.isfinite(args.drop_area_le):
        parser.error("--drop-area-le must be finite and non-negative")
    if args.dropped_sample_limit < 0:
        parser.error("--dropped-sample-limit must be non-negative")
    if args.center_atol < 0.0:
        parser.error("--center-atol must be non-negative")
    if not (0.0 <= args.domain_volume_warn_rtol <= args.domain_volume_fail_rtol):
        parser.error(
            "require 0 <= --domain-volume-warn-rtol <= --domain-volume-fail-rtol"
        )
    return args


def main() -> None:
    args = parse_args()
    for path in (args.uge, args.node, args.ugi):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.geometry_json is not None and not args.geometry_json.is_file():
        raise FileNotFoundError(args.geometry_json)
    if args.material_h5 is not None and not args.material_h5.is_file():
        raise FileNotFoundError(args.material_h5)

    print(f"Reading TetGen nodes: {args.node}")
    node_xyz = read_tetgen_nodes(args.node)

    print(f"Reading canonical UGI: {args.ugi}")
    ugi_element_count, ugi_xyz = read_ugi_vertices(args.ugi)
    if ugi_xyz.shape != node_xyz.shape:
        raise ValidationError(
            f"UGI/node count mismatch: UGI={ugi_xyz.shape[0]}, "
            f"TetGen={node_xyz.shape[0]}."
        )
    max_ugi_difference = float(np.max(np.linalg.norm(ugi_xyz - node_xyz, axis=1)))
    if max_ugi_difference > args.center_atol:
        raise ValidationError(
            "UGI vertices differ from TetGen row-order coordinates by as much as "
            f"{max_ugi_difference:.6e} m "
            f"(tolerance {args.center_atol:.6e} m)."
        )

    print(f"Reading and validating UGE: {args.uge}")
    uge = read_uge(
        args.uge,
        args.allow_duplicate_connections,
        args.drop_area_le,
        args.dropped_sample_limit,
    )
    if uge.centers.shape[0] != node_xyz.shape[0]:
        raise ValidationError(
            f"UGE/node count mismatch: UGE={uge.centers.shape[0]}, "
            f"TetGen={node_xyz.shape[0]}."
        )
    max_center_difference = float(
        np.max(np.linalg.norm(uge.centers - node_xyz, axis=1))
    )
    if max_center_difference > args.center_atol:
        worst = int(np.argmax(np.linalg.norm(uge.centers - node_xyz, axis=1)))
        raise ValidationError(
            "UGE cell centers do not follow TetGen node-file row order. "
            f"Maximum difference is {max_center_difference:.6e} m at cell "
            f"{worst + 1} (tolerance {args.center_atol:.6e} m)."
        )

    print(
        "Connection-area scan: "
        f"original={uge.original_connection_count:,}, "
        f"retained={uge.retained_connection_count:,}, "
        f"dropped={uge.dropped_connection_count:,} "
        f"(area <= {uge.drop_area_le_m2:.6e} m^2)"
    )
    for sample in uge.dropped_samples[:5]:
        print(
            "  dropped row {connection_row_1based}: cells "
            "{cell_a}-{cell_b}, area={area_m2:.6e} m^2".format(**sample)
        )

    if uge.dropped_connection_count > 0:
        if args.clean_uge is None:
            suggested = args.uge.with_name(args.uge.stem + "_clean.uge")
            raise ValidationError(
                f"Found {uge.dropped_connection_count} zero/negligible-area "
                "connection records. The positive-area graph is otherwise valid, "
                "but PFLOTRAN should use a sanitized UGE. Rerun with "
                f"--clean-uge {suggested}."
            )
        print(f"Writing sanitized UGE atomically: {args.clean_uge}")
        write_filtered_uge(args.uge, args.clean_uge, uge.keep_mask)
        effective_uge = args.clean_uge
    else:
        if args.clean_uge is not None:
            print(
                "No connection rows met the drop criterion; writing an identical "
                f"validated copy: {args.clean_uge}"
            )
            write_filtered_uge(args.uge, args.clean_uge, uge.keep_mask)
            effective_uge = args.clean_uge
        else:
            effective_uge = args.uge

    if args.mapping is None:
        args.mapping = effective_uge.with_suffix(".mapping")
    if args.all_vset is None:
        args.all_vset = effective_uge.with_name(effective_uge.stem + "_all.vset")
    if args.report_prefix is None:
        args.report_prefix = effective_uge.with_name(
            effective_uge.stem + "_uge_validation"
        )

    material_info = None
    if args.material_h5 is not None:
        print(f"Validating material HDF5: {args.material_h5}")
        material_info = validate_material_h5(args.material_h5, node_xyz.shape[0])

    domain_info = None
    if args.geometry_json is not None:
        geometry = json.loads(args.geometry_json.read_text(encoding="utf-8"))
        domain = geometry.get("domain")
        if not isinstance(domain, dict) or "min" not in domain or "max" not in domain:
            raise ValidationError(
                f"{args.geometry_json}: missing domain.min/domain.max arrays."
            )
        minimum = np.asarray(domain["min"], dtype=np.float64)
        maximum = np.asarray(domain["max"], dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
            raise ValidationError(f"{args.geometry_json}: invalid domain bounds.")
        expected_volume = float(np.prod(maximum - minimum))
        total_volume = float(np.sum(uge.volumes, dtype=np.float64))
        relative_error = abs(total_volume - expected_volume) / expected_volume
        domain_info = {
            "minimum_m": minimum.tolist(),
            "maximum_m": maximum.tolist(),
            "expected_volume_m3": expected_volume,
            "uge_total_volume_m3": total_volume,
            "relative_error": relative_error,
        }
        if relative_error > args.domain_volume_fail_rtol:
            raise ValidationError(
                "UGE total volume differs from the declared domain by "
                f"{relative_error:.6e}, above failure tolerance "
                f"{args.domain_volume_fail_rtol:.6e}."
            )
        if relative_error > args.domain_volume_warn_rtol:
            print(
                "WARNING: domain-volume relative mismatch is "
                f"{relative_error:.6e} (warning tolerance "
                f"{args.domain_volume_warn_rtol:.6e})."
            )

    report = build_report(
        args=args,
        uge=uge,
        effective_uge=effective_uge,
        node_xyz=node_xyz,
        ugi_element_count=ugi_element_count,
        max_center_difference=max_center_difference,
        max_ugi_difference=max_ugi_difference,
        material_info=material_info,
        domain_info=domain_info,
    )

    print("All hard checks passed; writing identity mapping atomically.")
    atomic_write_text(args.mapping, mapping_text(node_xyz.shape[0]))
    atomic_write_text(args.all_vset, vset_text(node_xyz.shape[0]))

    text_path = Path(str(args.report_prefix) + ".txt")
    json_path = Path(str(args.report_prefix) + ".json")
    atomic_write_text(text_path, text_report(report))
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\nUGE sanitization / mapping validation complete")
    print(f"  flow cells                  : {node_xyz.shape[0]:,}")
    print(f"  original connections        : {uge.original_connection_count:,}")
    print(f"  retained connections        : {uge.retained_connection_count:,}")
    print(f"  dropped zero/negligible     : {uge.dropped_connection_count:,}")
    print(f"  geomechanics tetrahedra     : {ugi_element_count:,}")
    print(f"  max UGE/node difference     : {max_center_difference:.3e} m")
    print(f"  max UGI/node difference     : {max_ugi_difference:.3e} m")
    print(f"  minimum cell volume         : {np.min(uge.volumes):.6e} m^3")
    print(f"  minimum retained area       : {np.min(uge.areas):.6e} m^2")
    print(f"  minimum center distance     : {np.min(uge.center_distances):.6e} m")
    print(f"  maximum A/d factor          : {np.max(uge.area_over_distance):.6e} m")
    print(f"  graph components            : {uge.component_count}")
    print(f"  duplicate retained pairs    : {uge.duplicate_pair_count}")
    print(f"  effective UGE               : {effective_uge}")
    print(f"  mapping                     : {args.mapping}")
    print(f"  all-vertex vset             : {args.all_vset}")
    print(f"  text report                 : {text_path}")
    print(f"  JSON report                 : {json_path}")
    if domain_info is not None:
        print(
            "  domain-volume rel. error    : "
            f"{float(domain_info['relative_error']):.3e}"
        )


if __name__ == "__main__":
    main()
