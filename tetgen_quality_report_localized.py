#!/usr/bin/env python3
"""TetGen mesh-quality report with spatial localization of critical tetrahedra.

This is a drop-in diagnostic companion for the North Avant / Bartlesville mesh
workflow.  It reads TetGen ``.node`` and ``.ele`` files, computes tetrahedron
quality measures, exports ParaView VTU files, and writes CSV tables that locate
where the worst tetrahedra occur.

The script does not modify the mesh.

Outputs
-------
``<prefix>_quality_summary.csv``
    Global quality statistics.
``<prefix>_quality_histogram.csv``
    Edge-ratio histogram, retained for compatibility with the older report.
``<prefix>_quality_zone_summary.csv``
    Counts, trigger-specific counts, and extrema grouped by target, nested HEC
    refinement zone, or geological layer.
``<prefix>_quality_trigger_summary.csv``
    Overall counts for each critical-quality trigger and their combinations.
``<prefix>_quality_zbin_summary.csv``
    Depth-binned critical counts and extrema.
``<prefix>_quality_worst_tets.csv``
    Union of the worst tetrahedra by minimum dihedral angle, radius-edge ratio,
    edge ratio, normalized volume, and volume.  Includes centroid coordinates,
    element/material attribute, node IDs, and nearest target information.
``<prefix>_quality.vtu``
    Full tetrahedral mesh with cell-centered quality fields.
``<prefix>_quality_critical_tets.vtu``
    Only tetrahedra that meet one or more critical thresholds.

Examples
--------
    python3 tetgen_quality_report_localized.py bartlesville_hec_lime_v3_noq \
        --outdir quality_limestone_v3_localized

    python3 tetgen_quality_report_localized.py \
        --node mesh.1.node --ele mesh.1.ele --geometry mesh_geometry.json
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TetElement:
    element_id: int
    nodes: Tuple[int, int, int, int]
    attribute: int


def iter_data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def parse_tetgen_node(path: Path) -> Dict[int, np.ndarray]:
    lines = iter(iter_data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty TetGen node file: {path}") from exc
    if len(header) < 4:
        raise RuntimeError(f"Malformed TetGen node header in {path}: {' '.join(header)}")

    count = int(float(header[0]))
    dimension = int(float(header[1]))
    if dimension != 3:
        raise RuntimeError(f"Expected a 3-D node file; received dimension={dimension} in {path}")

    nodes: Dict[int, np.ndarray] = {}
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{path}: expected {count} node rows, stopped at {row}") from exc
        if len(fields) < 4:
            raise RuntimeError(f"{path}: malformed node row {row + 1}: {' '.join(fields)}")
        node_id = int(float(fields[0]))
        if node_id in nodes:
            raise RuntimeError(f"{path}: duplicate node ID {node_id}")
        xyz = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)
        if not np.all(np.isfinite(xyz)):
            raise RuntimeError(f"{path}: node {node_id} has non-finite coordinates {xyz}")
        nodes[node_id] = xyz

    return nodes


def parse_tetgen_ele(path: Path) -> List[TetElement]:
    lines = iter(iter_data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty TetGen element file: {path}") from exc
    if len(header) < 2:
        raise RuntimeError(f"Malformed TetGen element header in {path}: {' '.join(header)}")

    count = int(float(header[0]))
    nodes_per_element = int(float(header[1]))
    number_of_attributes = int(float(header[2])) if len(header) >= 3 else 0
    if nodes_per_element != 4:
        raise RuntimeError(f"Expected 4-node tetrahedra, got {nodes_per_element} in {path}")

    elements: List[TetElement] = []
    seen: set[int] = set()
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{path}: expected {count} element rows, stopped at {row}") from exc
        if len(fields) < 5:
            raise RuntimeError(f"{path}: malformed element row {row + 1}: {' '.join(fields)}")
        element_id = int(float(fields[0]))
        if element_id in seen:
            raise RuntimeError(f"{path}: duplicate element ID {element_id}")
        seen.add(element_id)
        nodes = tuple(int(float(value)) for value in fields[1:5])
        attribute = 1
        if number_of_attributes > 0 and len(fields) >= 6:
            attribute = int(round(float(fields[5])))
        elements.append(TetElement(element_id, nodes, attribute))

    return elements


def signed_six_volume(points: Sequence[np.ndarray]) -> float:
    a, b, c, d = points
    return float(np.dot(b - a, np.cross(c - a, d - a)))


def tetra_volume(points: Sequence[np.ndarray]) -> float:
    return abs(signed_six_volume(points)) / 6.0


def edge_lengths(points: Sequence[np.ndarray]) -> List[float]:
    return [
        float(np.linalg.norm(points[i] - points[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    ]


def tetra_circumradius(points: Sequence[np.ndarray]) -> float:
    """Return the tetrahedron circumradius using a row-wise linear system."""
    a, b, c, d = points
    matrix = np.vstack([b - a, c - a, d - a])
    rhs = 0.5 * np.asarray(
        [
            float(np.dot(b, b) - np.dot(a, a)),
            float(np.dot(c, c) - np.dot(a, a)),
            float(np.dot(d, d) - np.dot(a, a)),
        ],
        dtype=float,
    )
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return math.inf
    radius = float(np.linalg.norm(center - a))
    return radius if math.isfinite(radius) else math.inf


def outward_face_normal(
    points: Sequence[np.ndarray],
    face: Tuple[int, int, int],
    opposite: int,
) -> np.ndarray:
    i, j, k = face
    normal = np.cross(points[j] - points[i], points[k] - points[i])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return np.zeros(3, dtype=float)
    # If the normal points toward the opposite vertex, reverse it so it is outward.
    if float(np.dot(normal, points[opposite] - points[i])) > 0.0:
        normal = -normal
    return normal / float(np.linalg.norm(normal))


def dihedral_angles_deg(points: Sequence[np.ndarray]) -> List[float]:
    """Return the six internal dihedral angles of a tetrahedron in degrees."""
    # Each edge is shared by two faces.  The internal dihedral is pi minus the
    # angle between the corresponding outward face normals.
    edge_faces = [
        ((0, 1), (0, 1, 2), 3, (0, 1, 3), 2),
        ((0, 2), (0, 2, 1), 3, (0, 2, 3), 1),
        ((0, 3), (0, 3, 1), 2, (0, 3, 2), 1),
        ((1, 2), (1, 2, 0), 3, (1, 2, 3), 0),
        ((1, 3), (1, 3, 0), 2, (1, 3, 2), 0),
        ((2, 3), (2, 3, 0), 1, (2, 3, 1), 0),
    ]
    angles: List[float] = []
    for _edge, face_a, opposite_a, face_b, opposite_b in edge_faces:
        normal_a = outward_face_normal(points, face_a, opposite_a)
        normal_b = outward_face_normal(points, face_b, opposite_b)
        if float(np.linalg.norm(normal_a)) == 0.0 or float(np.linalg.norm(normal_b)) == 0.0:
            return [0.0] * 6
        cosine = float(np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))
        normal_angle = math.acos(cosine)
        internal = math.pi - normal_angle
        angles.append(math.degrees(internal))
    return angles


def quality_bin(edge_ratio: float, min_dihedral: float) -> str:
    # Retain the original project thresholds so old and new reports compare.
    if edge_ratio > 50.0 or min_dihedral < 5.0:
        return "bad"
    if edge_ratio > 20.0 or min_dihedral < 10.0:
        return "poor"
    if edge_ratio > 10.0 or min_dihedral < 18.0:
        return "fair"
    return "good"


def load_geometry(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def layer_name_at_z(z_value: float, geometry: Optional[Mapping[str, Any]]) -> str:
    if not geometry:
        return "unclassified"
    for layer in geometry.get("layers", []):
        lower = float(layer["z_min"])
        upper = float(layer["z_max"])
        if lower <= z_value <= upper:
            return str(layer.get("name", f"material_{layer.get('material_id', 'unknown')}"))
    return "outside_layers"


def target_proximity(
    centroid: np.ndarray,
    geometry: Optional[Mapping[str, Any]],
) -> Tuple[str, float]:
    """Return nearest active target and a normalized proximity measure."""
    if not geometry:
        return "none", math.inf

    best_name = "none"
    best_measure = math.inf
    for target in geometry.get("refinement_targets", []):
        name = str(target.get("name", "target"))
        center = np.asarray(target.get("center_xyz_m", target.get("center", [0, 0, 0])), dtype=float)
        shape = str(target.get("tag_shape", ""))

        if shape == "sphere":
            radii = [float(target.get("tag_radius_m", target.get("radius_m", 0.0)))]
            radii.extend(float(value) for value in target.get("geodesic_shell_radii_m", []))
            outer = max(radii) if radii else 1.0
            measure = float(np.linalg.norm(centroid - center)) / max(outer, 1.0e-12)
        elif shape == "vertical_cylinder":
            shells = target.get("tube_shells", [])
            radii = [float(target.get("tag_radius_m", target.get("radius_m", 0.0)))]
            radii.extend(float(shell.get("radius_m", 0.0)) for shell in shells)
            paddings = [0.0]
            paddings.extend(float(shell.get("endpoint_padding_m", 0.0)) for shell in shells)
            outer = max(radii) if radii else 1.0
            padding = max(paddings)
            radial = float(np.linalg.norm(centroid[:2] - center[:2]))
            z_min = float(target.get("tag_z_min_m", center[2])) - padding
            z_max = float(target.get("tag_z_max_m", center[2])) + padding
            axial_excess = max(z_min - float(centroid[2]), float(centroid[2]) - z_max, 0.0)
            measure = math.hypot(radial / max(outer, 1.0e-12), axial_excess / max(padding, 1.0))
        else:
            measure = float(np.linalg.norm(centroid - center))

        if measure < best_measure:
            best_measure = measure
            best_name = name

    return best_name, best_measure


def _normalized_axis(value: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    axis = np.asarray(value if value is not None else fallback, dtype=float)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        axis = np.asarray(fallback, dtype=float)
        norm = float(np.linalg.norm(axis))
    return axis / norm


def hec_local_coordinates(
    point: np.ndarray,
    geometry: Optional[Mapping[str, Any]],
) -> Tuple[float, float, float]:
    """Return HEC-local length, width, and vertical coordinates in metres."""
    if not geometry or not isinstance(geometry.get("hec"), Mapping):
        return math.nan, math.nan, math.nan
    hec = geometry["hec"]
    center = np.asarray(
        hec.get("center_xyz_m", hec.get("center", [0.0, 0.0, 0.0])),
        dtype=float,
    )
    axes = hec.get("axes", {})
    length_axis = _normalized_axis(axes.get("length"), [1.0, 0.0, 0.0])
    width_axis = _normalized_axis(axes.get("width"), [0.0, 1.0, 0.0])
    up_axis = _normalized_axis(
        axes.get("up", axes.get("normal_up")),
        [0.0, 0.0, 1.0],
    )
    relative = np.asarray(point, dtype=float) - center
    return (
        float(np.dot(relative, length_axis)),
        float(np.dot(relative, width_axis)),
        float(np.dot(relative, up_axis)),
    )


def _hec_refinement_zones(
    geometry: Optional[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    if not geometry:
        return []
    meshing = geometry.get("meshing", {})
    if not isinstance(meshing, Mapping):
        return []
    zones = meshing.get("hec_refinement_zones", [])
    if not isinstance(zones, list):
        return []
    clean = [zone for zone in zones if isinstance(zone, Mapping)]
    return sorted(clean, key=lambda zone: float(zone.get("xy_spacing_m", math.inf)))


def _zone_vertical_envelope(zone: Mapping[str, Any]) -> Tuple[float, float]:
    values = sorted(float(value) for value in zone.get("vertical_levels_m", []))
    if not values:
        return -math.inf, math.inf
    spacing = float(zone.get("xy_spacing_m", 0.0))
    gaps = [right - left for left, right in zip(values[:-1], values[1:])]
    transition_scale = max([spacing, *gaps, 1.0])
    padding = 0.5 * transition_scale
    return values[0] - padding, values[-1] + padding


def classify_hec_refinement_zone(
    centroid: np.ndarray,
    geometry: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], float, float, float]:
    """Classify a centroid using refinement-zone metadata written by the builder.

    Zones are checked from finest to coarsest.  This makes the diagnostic valid
    for both the single-zone V3 mesh and the nested V4/V5 layouts instead of
    relying on the former hard-coded 650 m by 650 m V3 envelope.
    """
    u_m, v_m, w_m = hec_local_coordinates(centroid, geometry)
    zones = _hec_refinement_zones(geometry)
    if not zones or not all(math.isfinite(value) for value in (u_m, v_m, w_m)):
        return None, u_m, v_m, w_m

    z_value = float(centroid[2])
    for zone in zones:
        x_half = float(zone.get("x_half_extent_m", 0.0))
        y_half = float(zone.get("y_half_extent_m", 0.0))
        spacing = float(zone.get("xy_spacing_m", 0.0))
        # Include cells that bridge from the last point row to the next zone.
        plan_padding = 0.75 * max(spacing, 1.0)
        z_min, z_max = _zone_vertical_envelope(zone)
        if (
            abs(u_m) <= x_half + plan_padding
            and abs(v_m) <= y_half + plan_padding
            and z_min <= z_value <= z_max
        ):
            return f"hec_zone:{zone.get('name', 'unnamed')}", u_m, v_m, w_m

    # Keep one explicit label for tetrahedra immediately outside the outermost
    # configured zone; these are often the transition cells we need to repair.
    outer = zones[-1]
    x_half = float(outer.get("x_half_extent_m", 0.0))
    y_half = float(outer.get("y_half_extent_m", 0.0))
    spacing = float(outer.get("xy_spacing_m", 0.0))
    z_min, z_max = _zone_vertical_envelope(outer)
    if (
        abs(u_m) <= x_half + 2.0 * max(spacing, 1.0)
        and abs(v_m) <= y_half + 2.0 * max(spacing, 1.0)
        and z_min - max(spacing, 1.0) <= float(centroid[2]) <= z_max + max(spacing, 1.0)
    ):
        return "hec_zone:outer_bridge", u_m, v_m, w_m

    return None, u_m, v_m, w_m


def spatial_zone(
    centroid: np.ndarray,
    geometry: Optional[Mapping[str, Any]],
) -> Tuple[str, str, float, float, float, float, str]:
    target_name, target_measure = target_proximity(centroid, geometry)
    layer = layer_name_at_z(float(centroid[2]), geometry)
    if target_measure <= 1.35:
        u_m, v_m, w_m = hec_local_coordinates(centroid, geometry)
        return (
            f"target:{target_name}",
            target_name,
            target_measure,
            u_m,
            v_m,
            w_m,
            layer,
        )

    hec_zone, u_m, v_m, w_m = classify_hec_refinement_zone(centroid, geometry)
    if hec_zone is not None:
        return hec_zone, target_name, target_measure, u_m, v_m, w_m, layer

    return (
        f"layer:{layer}",
        target_name,
        target_measure,
        u_m,
        v_m,
        w_m,
        layer,
    )


def build_quality_records(
    nodes: Dict[int, np.ndarray],
    elements: Sequence[TetElement],
    geometry: Optional[Mapping[str, Any]],
    critical_min_dihedral: float,
    critical_radius_edge: float,
    critical_edge_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    qualities: List[Dict[str, Any]] = []
    critical_mask: List[bool] = []

    for element in elements:
        if len(set(element.nodes)) != 4:
            raise RuntimeError(f"Element {element.element_id} has repeated node IDs: {element.nodes}")
        try:
            points = [nodes[node_id] for node_id in element.nodes]
        except KeyError as exc:
            raise RuntimeError(f"Element {element.element_id} references missing node {exc.args[0]}") from exc

        signed6 = signed_six_volume(points)
        volume = abs(signed6) / 6.0
        lengths = edge_lengths(points)
        minimum_edge = min(lengths)
        maximum_edge = max(lengths)
        edge_ratio = math.inf if minimum_edge <= 0.0 else maximum_edge / minimum_edge
        circumradius = tetra_circumradius(points)
        radius_edge_ratio = (
            math.inf
            if minimum_edge <= 0.0 or not math.isfinite(circumradius)
            else circumradius / minimum_edge
        )
        dihedrals = dihedral_angles_deg(points)
        minimum_dihedral = min(dihedrals)
        maximum_dihedral = max(dihedrals)
        centroid = np.mean(np.vstack(points), axis=0)
        normalized_volume = (
            0.0
            if maximum_edge <= 0.0
            else 6.0 * math.sqrt(2.0) * volume / (maximum_edge ** 3)
        )

        quality_class = quality_bin(edge_ratio, minimum_dihedral)
        trigger_min_dihedral = minimum_dihedral <= critical_min_dihedral
        trigger_radius_edge = radius_edge_ratio >= critical_radius_edge
        trigger_edge_ratio = edge_ratio >= critical_edge_ratio
        critical = bool(
            trigger_min_dihedral
            or trigger_radius_edge
            or trigger_edge_ratio
        )
        reason_names = []
        reason_code = 0
        if trigger_min_dihedral:
            reason_names.append("min_dihedral")
            reason_code |= 1
        if trigger_radius_edge:
            reason_names.append("radius_edge")
            reason_code |= 2
        if trigger_edge_ratio:
            reason_names.append("edge_ratio")
            reason_code |= 4
        (
            zone, nearest_target, target_measure,
            hec_u_m, hec_v_m, hec_w_m, layer_name,
        ) = spatial_zone(centroid, geometry)

        record: Dict[str, Any] = {
            "element_id": element.element_id,
            "attribute": element.attribute,
            "node_1": element.nodes[0],
            "node_2": element.nodes[1],
            "node_3": element.nodes[2],
            "node_4": element.nodes[3],
            "centroid_x_m": float(centroid[0]),
            "centroid_y_m": float(centroid[1]),
            "centroid_z_m": float(centroid[2]),
            "signed_six_volume_m3": signed6,
            "volume_m3": volume,
            "minimum_edge_m": minimum_edge,
            "maximum_edge_m": maximum_edge,
            "edge_ratio": edge_ratio if math.isfinite(edge_ratio) else 1.0e30,
            "circumradius_m": circumradius if math.isfinite(circumradius) else 1.0e30,
            "radius_edge_ratio": radius_edge_ratio if math.isfinite(radius_edge_ratio) else 1.0e30,
            "normalized_volume": normalized_volume,
            "min_dihedral_deg": minimum_dihedral,
            "max_dihedral_deg": maximum_dihedral,
            "quality_class": quality_class,
            "quality_code": {"good": 0.0, "fair": 1.0, "poor": 2.0, "bad": 3.0}[quality_class],
            "critical_flag": 1.0 if critical else 0.0,
            "critical_reason_code": float(reason_code),
            "critical_reasons": ";".join(reason_names) if reason_names else "none",
            "trigger_min_dihedral": 1.0 if trigger_min_dihedral else 0.0,
            "trigger_radius_edge": 1.0 if trigger_radius_edge else 0.0,
            "trigger_edge_ratio": 1.0 if trigger_edge_ratio else 0.0,
            "spatial_zone": zone,
            "geological_layer": layer_name,
            "hec_u_m": hec_u_m,
            "hec_v_m": hec_v_m,
            "hec_w_m": hec_w_m,
            "nearest_target": nearest_target,
            "nearest_target_normalized_distance": target_measure,
        }
        qualities.append(record)
        critical_mask.append(critical)

    return qualities, critical_mask


def percentile(values: Sequence[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), value))


def write_summary_csv(path: Path, qualities: Sequence[Mapping[str, Any]]) -> None:
    def values(name: str) -> np.ndarray:
        return np.asarray([float(record[name]) for record in qualities], dtype=float)

    volumes = values("volume_m3")
    edge_ratios = values("edge_ratio")
    radius_ratios = values("radius_edge_ratio")
    min_dihedrals = values("min_dihedral_deg")
    max_dihedrals = values("max_dihedral_deg")
    normalized = values("normalized_volume")

    rows: List[Tuple[str, float | int]] = [("count", len(qualities))]
    for label, array in (
        ("volume_m3", volumes),
        ("edge_ratio", edge_ratios),
        ("radius_edge_ratio", radius_ratios),
        ("min_dihedral_deg", min_dihedrals),
        ("max_dihedral_deg", max_dihedrals),
        ("normalized_volume", normalized),
    ):
        rows.extend(
            [
                (f"{label}_min", float(np.min(array))),
                (f"{label}_p1", percentile(array, 1)),
                (f"{label}_p5", percentile(array, 5)),
                (f"{label}_median", float(np.median(array))),
                (f"{label}_p95", percentile(array, 95)),
                (f"{label}_p99", percentile(array, 99)),
                (f"{label}_max", float(np.max(array))),
            ]
        )

    rows.extend(
        [
            ("inverted_tets", sum(float(q["signed_six_volume_m3"]) < 0.0 for q in qualities)),
            ("critical_tets", sum(float(q["critical_flag"]) > 0.5 for q in qualities)),
            ("bad_tets", sum(q["quality_class"] == "bad" for q in qualities)),
            ("poor_tets", sum(q["quality_class"] == "poor" for q in qualities)),
            ("fair_tets", sum(q["quality_class"] == "fair" for q in qualities)),
            ("good_tets", sum(q["quality_class"] == "good" for q in qualities)),
        ]
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_histogram_csv(path: Path, qualities: Sequence[Mapping[str, Any]]) -> None:
    values = np.asarray([float(q["edge_ratio"]) for q in qualities], dtype=float)
    bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, math.inf]
    counts, edges = np.histogram(values, bins=bins)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "count"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([left, right, int(count)])


def write_zone_summary_csv(path: Path, qualities: Sequence[Mapping[str, Any]]) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in qualities:
        grouped[str(record["spatial_zone"])].append(record)

    fieldnames = [
        "spatial_zone",
        "tetrahedra",
        "critical_tetrahedra",
        "critical_fraction",
        "trigger_min_dihedral_count",
        "trigger_radius_edge_count",
        "trigger_edge_ratio_count",
        "multiple_trigger_count",
        "min_dihedral_min_deg",
        "min_dihedral_p1_deg",
        "radius_edge_p99",
        "radius_edge_max",
        "edge_ratio_p99",
        "edge_ratio_max",
        "volume_min_m3",
        "centroid_z_min_m",
        "centroid_z_max_m",
        "abs_hec_u_p95_m",
        "abs_hec_v_p95_m",
    ]
    rows: List[Dict[str, Any]] = []
    for zone, records in grouped.items():
        min_dih = np.asarray([float(q["min_dihedral_deg"]) for q in records])
        radius = np.asarray([float(q["radius_edge_ratio"]) for q in records])
        edge = np.asarray([float(q["edge_ratio"]) for q in records])
        volume = np.asarray([float(q["volume_m3"]) for q in records])
        zvals = np.asarray([float(q["centroid_z_m"]) for q in records])
        uvals = np.asarray([abs(float(q["hec_u_m"])) for q in records])
        vvals = np.asarray([abs(float(q["hec_v_m"])) for q in records])
        finite_u = uvals[np.isfinite(uvals)]
        finite_v = vvals[np.isfinite(vvals)]
        critical = sum(float(q["critical_flag"]) > 0.5 for q in records)
        trigger_dih = sum(float(q["trigger_min_dihedral"]) > 0.5 for q in records)
        trigger_radius = sum(float(q["trigger_radius_edge"]) > 0.5 for q in records)
        trigger_edge = sum(float(q["trigger_edge_ratio"]) > 0.5 for q in records)
        multiple = sum(int(float(q["critical_reason_code"])) not in {0, 1, 2, 4} for q in records)
        rows.append(
            {
                "spatial_zone": zone,
                "tetrahedra": len(records),
                "critical_tetrahedra": critical,
                "critical_fraction": critical / len(records),
                "trigger_min_dihedral_count": trigger_dih,
                "trigger_radius_edge_count": trigger_radius,
                "trigger_edge_ratio_count": trigger_edge,
                "multiple_trigger_count": multiple,
                "min_dihedral_min_deg": float(np.min(min_dih)),
                "min_dihedral_p1_deg": float(np.percentile(min_dih, 1)),
                "radius_edge_p99": float(np.percentile(radius, 99)),
                "radius_edge_max": float(np.max(radius)),
                "edge_ratio_p99": float(np.percentile(edge, 99)),
                "edge_ratio_max": float(np.max(edge)),
                "volume_min_m3": float(np.min(volume)),
                "centroid_z_min_m": float(np.min(zvals)),
                "centroid_z_max_m": float(np.max(zvals)),
                "abs_hec_u_p95_m": float(np.percentile(finite_u, 95)) if finite_u.size else math.nan,
                "abs_hec_v_p95_m": float(np.percentile(finite_v, 95)) if finite_v.size else math.nan,
            }
        )
    rows.sort(key=lambda row: (int(row["critical_tetrahedra"]), float(row["radius_edge_max"])), reverse=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_trigger_summary_csv(path: Path, qualities: Sequence[Mapping[str, Any]]) -> None:
    counts: Dict[int, int] = defaultdict(int)
    for record in qualities:
        counts[int(float(record["critical_reason_code"]))] += 1
    labels = {
        0: "none",
        1: "min_dihedral_only",
        2: "radius_edge_only",
        3: "min_dihedral_and_radius_edge",
        4: "edge_ratio_only",
        5: "min_dihedral_and_edge_ratio",
        6: "radius_edge_and_edge_ratio",
        7: "all_three",
    }
    total = len(qualities)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason_code", "reason", "tetrahedra", "fraction"])
        for code in range(8):
            count = counts.get(code, 0)
            writer.writerow([code, labels[code], count, count / total if total else 0.0])


def write_zbin_summary_csv(
    path: Path,
    qualities: Sequence[Mapping[str, Any]],
    bin_size_m: float,
) -> None:
    if bin_size_m <= 0.0:
        raise ValueError("z-bin size must be positive")
    grouped: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for record in qualities:
        z_value = float(record["centroid_z_m"])
        lower = math.floor(z_value / bin_size_m) * bin_size_m
        grouped[lower].append(record)
    fieldnames = [
        "z_bin_lower_m", "z_bin_upper_m", "tetrahedra", "critical_tetrahedra",
        "critical_fraction", "trigger_min_dihedral_count", "trigger_radius_edge_count",
        "trigger_edge_ratio_count", "min_dihedral_min_deg", "radius_edge_max",
        "edge_ratio_max", "volume_min_m3",
    ]
    rows: List[Dict[str, Any]] = []
    for lower, records in grouped.items():
        rows.append({
            "z_bin_lower_m": lower,
            "z_bin_upper_m": lower + bin_size_m,
            "tetrahedra": len(records),
            "critical_tetrahedra": sum(float(q["critical_flag"]) > 0.5 for q in records),
            "critical_fraction": sum(float(q["critical_flag"]) > 0.5 for q in records) / len(records),
            "trigger_min_dihedral_count": sum(float(q["trigger_min_dihedral"]) > 0.5 for q in records),
            "trigger_radius_edge_count": sum(float(q["trigger_radius_edge"]) > 0.5 for q in records),
            "trigger_edge_ratio_count": sum(float(q["trigger_edge_ratio"]) > 0.5 for q in records),
            "min_dihedral_min_deg": min(float(q["min_dihedral_deg"]) for q in records),
            "radius_edge_max": max(float(q["radius_edge_ratio"]) for q in records),
            "edge_ratio_max": max(float(q["edge_ratio"]) for q in records),
            "volume_min_m3": min(float(q["volume_m3"]) for q in records),
        })
    rows.sort(key=lambda row: float(row["z_bin_lower_m"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_worst_tets_csv(
    path: Path,
    qualities: Sequence[Mapping[str, Any]],
    top_n: int,
) -> None:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    selected: Dict[int, set[str]] = defaultdict(set)
    index_by_id = {int(record["element_id"]): record for record in qualities}

    rankings = {
        "min_dihedral": heapq.nsmallest(top_n, qualities, key=lambda q: float(q["min_dihedral_deg"])),
        "max_radius_edge": heapq.nlargest(top_n, qualities, key=lambda q: float(q["radius_edge_ratio"])),
        "max_edge_ratio": heapq.nlargest(top_n, qualities, key=lambda q: float(q["edge_ratio"])),
        "min_normalized_volume": heapq.nsmallest(top_n, qualities, key=lambda q: float(q["normalized_volume"])),
        "min_volume": heapq.nsmallest(top_n, qualities, key=lambda q: float(q["volume_m3"])),
    }
    for ranking_name, records in rankings.items():
        for record in records:
            selected[int(record["element_id"])].add(ranking_name)

    rows: List[Dict[str, Any]] = []
    for element_id, reasons in selected.items():
        record = dict(index_by_id[element_id])
        record["worst_list_membership"] = ";".join(sorted(reasons))
        rows.append(record)

    rows.sort(
        key=lambda q: (
            float(q["min_dihedral_deg"]),
            -float(q["radius_edge_ratio"]),
            -float(q["edge_ratio"]),
        )
    )
    fieldnames = ["worst_list_membership", *index_by_id[next(iter(index_by_id))].keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_vtu(
    path: Path,
    nodes: Dict[int, np.ndarray],
    elements: Sequence[TetElement],
    qualities: Sequence[Mapping[str, Any]],
) -> None:
    node_ids = sorted(nodes)
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    points = np.vstack([nodes[node_id] for node_id in node_ids])

    numeric_arrays = {
        "element_id": [q["element_id"] for q in qualities],
        "attribute": [q["attribute"] for q in qualities],
        "volume_m3": [q["volume_m3"] for q in qualities],
        "normalized_volume": [q["normalized_volume"] for q in qualities],
        "minimum_edge_m": [q["minimum_edge_m"] for q in qualities],
        "maximum_edge_m": [q["maximum_edge_m"] for q in qualities],
        "edge_ratio": [q["edge_ratio"] for q in qualities],
        "radius_edge_ratio": [q["radius_edge_ratio"] for q in qualities],
        "min_dihedral_deg": [q["min_dihedral_deg"] for q in qualities],
        "max_dihedral_deg": [q["max_dihedral_deg"] for q in qualities],
        "quality_code": [q["quality_code"] for q in qualities],
        "critical_flag": [q["critical_flag"] for q in qualities],
        "critical_reason_code": [q["critical_reason_code"] for q in qualities],
        "trigger_min_dihedral": [q["trigger_min_dihedral"] for q in qualities],
        "trigger_radius_edge": [q["trigger_radius_edge"] for q in qualities],
        "trigger_edge_ratio": [q["trigger_edge_ratio"] for q in qualities],
        "hec_u_m": [q["hec_u_m"] for q in qualities],
        "hec_v_m": [q["hec_v_m"] for q in qualities],
        "hec_w_m": [q["hec_w_m"] for q in qualities],
    }

    with path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        handle.write("  <UnstructuredGrid>\n")
        handle.write(f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(elements)}">\n')
        handle.write("      <PointData/>\n")
        handle.write('      <CellData Scalars="min_dihedral_deg">\n')
        for name, values in numeric_arrays.items():
            handle.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            handle.write("          " + " ".join(f"{float(value):.16e}" for value in values) + "\n")
            handle.write("        </DataArray>\n")
        handle.write("      </CellData>\n")
        handle.write("      <Points>\n")
        handle.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for point in points:
            handle.write(f"          {point[0]:.16e} {point[1]:.16e} {point[2]:.16e}\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")
        handle.write("      <Cells>\n")
        handle.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for element in elements:
            handle.write("          " + " ".join(str(id_to_index[node]) for node in element.nodes) + "\n")
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for offset in range(4, 4 * len(elements) + 1, 4):
            handle.write(f"          {offset}\n")
        handle.write("        </DataArray>\n")
        handle.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in elements:
            handle.write("          10\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Cells>\n")
        handle.write("    </Piece>\n")
        handle.write("  </UnstructuredGrid>\n")
        handle.write("</VTKFile>\n")


def write_subset_vtu(
    path: Path,
    nodes: Dict[int, np.ndarray],
    elements: Sequence[TetElement],
    qualities: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
) -> None:
    indices = [index for index, selected in enumerate(mask) if selected]
    if not indices:
        return
    write_vtu(
        path,
        nodes,
        [elements[index] for index in indices],
        [qualities[index] for index in indices],
    )


def resolve_inputs(
    prefix: Optional[str],
    node_argument: Optional[str],
    element_argument: Optional[str],
    geometry_argument: Optional[str],
) -> Tuple[Path, Path, Optional[Path], str]:
    if node_argument or element_argument:
        if not (node_argument and element_argument):
            raise SystemExit("Provide both --node and --ele, or provide only a mesh prefix.")
        node_path = Path(node_argument)
        element_path = Path(element_argument)
        mesh_prefix = prefix or node_path.name.replace(".1.node", "").replace(".node", "")
    else:
        if not prefix:
            raise SystemExit("Provide a mesh prefix, or both --node and --ele.")
        mesh_prefix = prefix
        node_path = Path(f"{mesh_prefix}.1.node")
        element_path = Path(f"{mesh_prefix}.1.ele")

    if not node_path.is_file():
        raise FileNotFoundError(node_path)
    if not element_path.is_file():
        raise FileNotFoundError(element_path)

    if geometry_argument:
        geometry_path: Optional[Path] = Path(geometry_argument)
        if not geometry_path.is_file():
            raise FileNotFoundError(geometry_path)
    else:
        candidate = Path(f"{mesh_prefix}_geometry.json")
        geometry_path = candidate if candidate.is_file() else None

    return node_path, element_path, geometry_path, mesh_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and spatially localize TetGen tetrahedron quality metrics."
    )
    parser.add_argument("prefix", nargs="?", help="Mesh prefix, e.g. bartlesville_hec_lime_v3_noq")
    parser.add_argument("--node", help="TetGen .node file")
    parser.add_argument("--ele", help="TetGen .ele file")
    parser.add_argument("--geometry", help="Optional geometry JSON; default <prefix>_geometry.json")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--top-n", type=int, default=500, help="Worst elements retained per ranking")
    parser.add_argument("--critical-min-dihedral", type=float, default=1.0, help="Critical minimum dihedral threshold [deg]")
    parser.add_argument("--critical-radius-edge", type=float, default=20.0, help="Critical radius-edge threshold")
    parser.add_argument("--critical-edge-ratio", type=float, default=100.0, help="Critical max/min edge threshold")
    parser.add_argument("--z-bin-size", type=float, default=5.0, help="Depth-bin size for the z summary [m]")
    parser.add_argument("--print-worst", type=int, default=12, help="Number of worst tetrahedra printed to the terminal")
    parser.add_argument("--csv-only", action="store_true", help="Skip VTU files when only CSV/terminal diagnostics are needed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_path, element_path, geometry_path, mesh_prefix = resolve_inputs(
        args.prefix,
        args.node,
        args.ele,
        args.geometry,
    )
    output_directory = Path(args.outdir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Reading {node_path.name} and {element_path.name}...")
    if geometry_path:
        print(f"Using geometry metadata: {geometry_path.name}")
    else:
        print("Geometry metadata not found; zone labels will be unclassified.")

    nodes = parse_tetgen_node(node_path)
    elements = parse_tetgen_ele(element_path)
    geometry = load_geometry(geometry_path)
    qualities, critical_mask = build_quality_records(
        nodes,
        elements,
        geometry,
        args.critical_min_dihedral,
        args.critical_radius_edge,
        args.critical_edge_ratio,
    )

    summary_path = output_directory / f"{mesh_prefix}_quality_summary.csv"
    histogram_path = output_directory / f"{mesh_prefix}_quality_histogram.csv"
    zone_path = output_directory / f"{mesh_prefix}_quality_zone_summary.csv"
    trigger_path = output_directory / f"{mesh_prefix}_quality_trigger_summary.csv"
    zbin_path = output_directory / f"{mesh_prefix}_quality_zbin_summary.csv"
    worst_path = output_directory / f"{mesh_prefix}_quality_worst_tets.csv"
    vtu_path = output_directory / f"{mesh_prefix}_quality.vtu"
    critical_vtu_path = output_directory / f"{mesh_prefix}_quality_critical_tets.vtu"

    write_summary_csv(summary_path, qualities)
    write_histogram_csv(histogram_path, qualities)
    write_zone_summary_csv(zone_path, qualities)
    write_trigger_summary_csv(trigger_path, qualities)
    write_zbin_summary_csv(zbin_path, qualities, args.z_bin_size)
    write_worst_tets_csv(worst_path, qualities, args.top_n)
    if not args.csv_only:
        write_vtu(vtu_path, nodes, elements, qualities)
        write_subset_vtu(critical_vtu_path, nodes, elements, qualities, critical_mask)

    minimum_dihedral = min(float(q["min_dihedral_deg"]) for q in qualities)
    maximum_radius_edge = max(float(q["radius_edge_ratio"]) for q in qualities)
    maximum_edge_ratio = max(float(q["edge_ratio"]) for q in qualities)
    critical_count = sum(critical_mask)

    print(f"nodes: {len(nodes):,}")
    print(f"elements: {len(elements):,}")
    print(f"critical tets: {critical_count:,} ({100.0 * critical_count / len(elements):.3f}%)")
    print(f"minimum dihedral angle: {minimum_dihedral:.6g} deg")
    print(f"maximum radius-edge ratio: {maximum_radius_edge:.6g}")
    print(f"maximum edge ratio: {maximum_edge_ratio:.6g}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {zone_path}")
    print(f"Wrote {trigger_path}")
    print(f"Wrote {zbin_path}")
    print(f"Wrote {worst_path}")
    if not args.csv_only:
        print(f"Wrote {vtu_path}")
        if critical_count:
            print(f"Wrote {critical_vtu_path}")

    # Show the most problematic zones directly in the terminal.
    zone_rows: List[Tuple[int, str]] = []
    with zone_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            zone_rows.append((int(row["critical_tetrahedra"]), row["spatial_zone"]))
    print("\nTop zones by critical-tetrahedron count:")
    for count, zone in sorted(zone_rows, reverse=True)[:10]:
        print(f"  {zone:40s} {count:10,d}")

    trigger_counts = {
        "min_dihedral": sum(float(q["trigger_min_dihedral"]) > 0.5 for q in qualities),
        "radius_edge": sum(float(q["trigger_radius_edge"]) > 0.5 for q in qualities),
        "edge_ratio": sum(float(q["trigger_edge_ratio"]) > 0.5 for q in qualities),
    }
    print("\nCritical-trigger counts (overlap is allowed):")
    for name, count in trigger_counts.items():
        print(f"  {name:20s} {count:10,d}")

    print("\nWorst tetrahedra by minimum dihedral angle:")
    worst_records = heapq.nsmallest(
        max(0, args.print_worst),
        qualities,
        key=lambda q: float(q["min_dihedral_deg"]),
    )
    for record in worst_records:
        print(
            f"  tet={int(record['element_id']):8d} "
            f"zone={str(record['spatial_zone']):34s} "
            f"z={float(record['centroid_z_m']):9.3f} "
            f"u={float(record['hec_u_m']):9.3f} "
            f"v={float(record['hec_v_m']):9.3f} "
            f"dih={float(record['min_dihedral_deg']):9.5f} "
            f"R/e={float(record['radius_edge_ratio']):12.4g} "
            f"edge={float(record['edge_ratio']):10.4g}"
        )


if __name__ == "__main__":
    main()
