#!/usr/bin/env python3
"""Build the North Avant / Bartlesville PLC with a tag-only HEC only.

This replacement keeps:
  * the 10 km x 10 km x 750 m layered domain,
  * the 5 degree east-of-north rotated HEC proxy,
  * the PFLOTRAN-ready layered region structure,

and removes:
  * the expensive borehole mesh-zone lattices,
  * all strainmeter borehole construction,
  * all borehole surface markers / local shells.

The injection well is now handled as a small PFLOTRAN source region:
  * TetGen writes the mesh from this PLC,
  * make_wellbore_vset.py is used later in workflow.py to build wellbore.vset
    from the TetGen .node file,
  * the PFLOTRAN SOURCE_SINK block can remain unchanged.

Run:
    python3 build_poly_layers4_hec_only.py bartlesville_hec /path/to/tetgen
or:
    python3 build_poly_layers4_hec_only.py bartlesville_hec --write-only
"""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Domain and geological layers.
# -----------------------------------------------------------------------------
DOMAIN_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
DOMAIN_MAX = np.array([10000.0, 10000.0, 750.0], dtype=float)
DOMAIN_SIZE = DOMAIN_MAX - DOMAIN_MIN


@dataclass(frozen=True)
class GeologicalLayer:
    number: int
    material_id: int
    name: str
    z_min: float
    z_max: float

# 1) replace the layer stack near the top with this
LAYERS = (
    GeologicalLayer(1, 4, "underburden", 0.0, 200.0),
    GeologicalLayer(2, 3, "basal_layer", 200.0, 500.0),
    GeologicalLayer(3, 2, "bartlesville_sand", 500.0, 535.0),
    GeologicalLayer(4, 1, "overburden", 535.0, 750.0),
)

# -----------------------------------------------------------------------------
# Tag-only HEC proxy geometry.
# -----------------------------------------------------------------------------
HEC_NAME = "bartlesville_hec"
HEC_MATERIAL_ID = 5
HEC_HOST_MATERIAL_ID = 2
HEC_CENTER = np.array([5000.0, 5000.0, 530.0], dtype=float)
HEC_LENGTH_M = 580.0
HEC_WIDTH_M = 300.0
HEC_THICKNESS_M = 5.0
HEC_AZIMUTH_EAST_OF_NORTH_DEG = 5.0

# -----------------------------------------------------------------------------
# Matrix grid used to create the layered PLC.
# -----------------------------------------------------------------------------
MATRIX_COARSE_STEP_M = 250.0
ROTATED_TAG_LATTICE_STEP_M = 20.0
X_FINE_START, X_FINE_END = 4800.0, 5200.0
Y_FINE_START, Y_FINE_END = 4680.0, 5320.0
ROTATION_INNER_HALF_X = 220.0
ROTATION_INNER_HALF_Y = 360.0
ROTATION_OUTER_HALF_X = 800.0
ROTATION_OUTER_HALF_Y = 1000.0


# Local refinement windows: injector, HEC neighborhood, and strainmeters.
# Replace the strainmeter boxes with your actual observation-point coordinates.
REFINEMENT_WINDOWS = (
    ("injector_core", 4987.5, 5012.5, 4987.5, 5012.5, 502.0, 533.0, 2.5, 2.5, 2.5),
    ("avn31", 5330.0, 5370.0, 4710.0, 4750.0, 503.0, 532.0, 10.0, 10.0, 5.0),
    ("avn2",  5140.0, 5180.0, 5175.0, 5195.0, 10.0, 60.0, 20.0, 10.0, 10.0),
    ("avn87", 5440.0, 5480.0, 5175.0, 5195.0, 10.0, 60.0, 20.0, 10.0, 10.0),
    ("central_bulk", 462.5, 562.5, 462.5, 562.5, 462.5, 562.5, 25.0, 25.0, 25.0),
)
# Vertical grading. The matrix levels include the HEC support planes z=527.5,
# 530.0 and 532.5 so the tag-only HEC can be represented cleanly as a rotated
# prism-like material zone without introducing internal PLC facets.
@dataclass(frozen=True)
class VerticalBand:
    label: str
    z_min: float
    z_max: float
    geological_layer: int
    note: str


VERTICAL_BANDS: Tuple[VerticalBand, ...] = (
    VerticalBand("L1_bottom_coarse", 0.0, 100.0, 1, "largest cells near lower external boundary"),
    VerticalBand("L1_lower_transition", 100.0, 160.0, 1, "transition toward refined beds"),
    VerticalBand("L1_mid_transition", 160.0, 185.0, 1, "transition toward refined beds"),
    VerticalBand("L1_upper_transition", 185.0, 200.0, 1, "matches fine spacing near layer 2"),
    VerticalBand("L2_lower_refined", 200.0, 210.0, 2, "refined near layer 3"),
    VerticalBand("L2_upper_refined", 210.0, 220.0, 2, "refined near layer 3"),
    VerticalBand("L3_lower_refined", 220.0, 235.0, 3, "layer 3 refined band"),
    VerticalBand("L3_upper_refined", 235.0, 250.0, 3, "layer 3 refined band"),
    VerticalBand("L4_lower_refined", 250.0, 350.0, 4, "coarsening upward"),
    VerticalBand("L4_transition_1", 350.0, 500.0, 4, "coarsening upward"),
    VerticalBand("L4_hec_lower_matrix", 500.0, 525.0, 4, "matrix plane below 5 m HEC support"),
    VerticalBand("L4_hec_center_matrix", 525.0, 530.0, 4, "matrix plane at HEC centre"),
    VerticalBand("L4_hec_upper_matrix", 530.0, 535.0, 4, "matrix plane above 5 m HEC support"),
    VerticalBand("L4_transition_2", 535.0, 650.0, 4, "coarsening upward"),
    VerticalBand("L4_top_transition", 650.0, 750.0, 4, "larger cells near upper external boundary"),
)
Z_LEVELS: Tuple[float, ...] = tuple([VERTICAL_BANDS[0].z_min] + [band.z_max for band in VERTICAL_BANDS])

BOUNDARY_MARKERS: Dict[str, int] = {
    "top": 1,
    "bottom": 2,
    "north": 3,
    "south": 4,
    "east": 5,
    "west": 6,
}
DEFAULT_TETGEN_FLAGS = "-pnAef"


@dataclass(frozen=True)
class Facet:
    point_ids: Tuple[int, int, int]
    marker: int


@dataclass(frozen=True)
class Region:
    point: np.ndarray
    attribute: int
    label: str


class PointRegistry:
    """Unique 3-D Part-1 point store with TetGen one-based IDs."""

    def __init__(self) -> None:
        self.points: List[np.ndarray] = []
        self._ids: Dict[Tuple[float, float, float], int] = {}

    @staticmethod
    def _key(point: Sequence[float]) -> Tuple[float, float, float]:
        return tuple(round(float(v), 10) for v in point)

    def add(self, point: Sequence[float]) -> int:
        xyz = np.asarray(point, dtype=float)
        key = self._key(xyz)
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        self.points.append(xyz)
        point_id = len(self.points)
        self._ids[key] = point_id
        return point_id

    def xyz(self, point_id: int) -> np.ndarray:
        return self.points[point_id - 1]


def unit(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(values))
    if magnitude <= 0.0:
        raise ValueError("Cannot normalise a zero vector.")
    return values / magnitude


def hec_axes() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(HEC_AZIMUTH_EAST_OF_NORTH_DEG)
    length_axis = unit([math.sin(angle), math.cos(angle), 0.0])
    width_axis = unit([math.cos(angle), -math.sin(angle), 0.0])
    return length_axis, width_axis, np.array([0.0, 0.0, 1.0], dtype=float)


def hec_corners() -> np.ndarray:
    length_axis, width_axis, up_axis = hec_axes()
    corners: List[np.ndarray] = []
    for dz in (-0.5 * HEC_THICKNESS_M, +0.5 * HEC_THICKNESS_M):
        for su, sv in ((+1.0, +1.0), (+1.0, -1.0), (-1.0, -1.0), (-1.0, +1.0)):
            corners.append(
                HEC_CENTER
                + su * 0.5 * HEC_LENGTH_M * length_axis
                + sv * 0.5 * HEC_WIDTH_M * width_axis
                + dz * up_axis
            )
    return np.asarray(corners, dtype=float)


def rotation_weight(value: float, inner_half: float, outer_half: float) -> float:
    distance = abs(value)
    if distance <= inner_half:
        return 1.0
    if distance >= outer_half:
        return 0.0
    fraction = (distance - inner_half) / (outer_half - inner_half)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def rotate_matrix_xy(base_x: float, base_y: float) -> Tuple[float, float]:
    dx, dy = base_x - HEC_CENTER[0], base_y - HEC_CENTER[1]
    weight = min(
        rotation_weight(dx, ROTATION_INNER_HALF_X, ROTATION_OUTER_HALF_X),
        rotation_weight(dy, ROTATION_INNER_HALF_Y, ROTATION_OUTER_HALF_Y),
    )
    angle = -math.radians(HEC_AZIMUTH_EAST_OF_NORTH_DEG) * weight
    c, s = math.cos(angle), math.sin(angle)
    return HEC_CENTER[0] + c * dx - s * dy, HEC_CENTER[1] + s * dx + c * dy


def unique_sorted(values: Iterable[float]) -> np.ndarray:
    return np.asarray(sorted({round(float(v), 9) for v in values}), dtype=float)

def endpoint_values(start: float, stop: float, step: float) -> List[float]:
    values = [float(start)]
    current = float(start)
    while current + step < stop - 1.0e-9:
        current += step
        values.append(current)
    if not math.isclose(values[-1], stop, abs_tol=1.0e-9):
        values.append(float(stop))
    return values


def add_box_lattice(
    registry: PointRegistry,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    dx: float, dy: float, dz: float,
) -> int:
    xs = endpoint_values(x0, x1, dx)
    ys = endpoint_values(y0, y1, dy)
    zs = endpoint_values(z0, z1, dz)
    count = 0
    for z in zs:
        for x in xs:
            for y in ys:
                registry.add((x, y, z))
                count += 1
    return count


def add_hierarchical_refinement(registry: PointRegistry) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, x0, x1, y0, y1, z0, z1, dx, dy, dz in REFINEMENT_WINDOWS:
        counts[name] = add_box_lattice(registry, x0, x1, y0, y1, z0, z1, dx, dy, dz)
    return counts


def inclusive_values(start: float, stop: float, step: float) -> List[float]:
    count = int(round((stop - start) / step))
    if not math.isclose(start + count * step, stop, abs_tol=1.0e-9):
        raise ValueError("[{:.3f}, {:.3f}] is not divisible by {:.3f}".format(start, stop, step))
    return [start + i * step for i in range(count + 1)]


def make_x_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4600.0]
    values += inclusive_values(X_FINE_START, X_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5400.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    return axis


def make_y_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4500.0]
    values += inclusive_values(Y_FINE_START, Y_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5500.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    return axis


def add_oriented_triangle(
    registry: PointRegistry,
    facets: List[Facet],
    ids: Sequence[int],
    normal: Sequence[float],
    marker: int,
) -> None:
    value = list(ids)
    a, b, c = (registry.xyz(i) for i in value)
    if float(np.dot(np.cross(b - a, c - a), np.asarray(normal, dtype=float))) < 0.0:
        value[1], value[2] = value[2], value[1]
    facets.append(Facet(tuple(value), marker))


def build_matrix_surface_plc(registry: PointRegistry, facets: List[Facet]) -> Tuple[np.ndarray, np.ndarray]:
    """Build the layered PLC for the matrix + tag-only HEC."""
    x_values, y_values = make_x_axis(), make_y_axis()
    nx, ny = len(x_values) - 1, len(y_values) - 1
    nodes: Dict[Tuple[int, int, int], int] = {}

    for k, z_value in enumerate(Z_LEVELS):
        for i, base_x in enumerate(x_values):
            for j, base_y in enumerate(y_values):
                x_value, y_value = rotate_matrix_xy(float(base_x), float(base_y))
                nodes[(i, j, k)] = registry.add((x_value, y_value, z_value))

    min_area = math.inf
    for i in range(nx):
        for j in range(ny):
            p00 = registry.xyz(nodes[(i, j, 0)])[:2]
            p10 = registry.xyz(nodes[(i + 1, j, 0)])[:2]
            p11 = registry.xyz(nodes[(i + 1, j + 1, 0)])[:2]
            p01 = registry.xyz(nodes[(i, j + 1, 0)])[:2]
            a1 = float((p10[0] - p00[0]) * (p11[1] - p00[1]) - (p10[1] - p00[1]) * (p11[0] - p00[0]))
            a2 = float((p11[0] - p00[0]) * (p01[1] - p00[1]) - (p11[1] - p00[1]) * (p01[0] - p00[0]))
            min_area = min(min_area, a1, a2)
            if a1 <= 1.0e-8 or a2 <= 1.0e-8:
                raise RuntimeError("Warped plan-view grid folded at ({}, {}).".format(i, j))
    print("    minimum warped plan-view triangle area: {:.6g} m^2".format(min_area))

    for k in range(len(Z_LEVELS)):
        if k == 0:
            marker, normal = BOUNDARY_MARKERS["bottom"], (0.0, 0.0, -1.0)
        elif k == len(Z_LEVELS) - 1:
            marker, normal = BOUNDARY_MARKERS["top"], (0.0, 0.0, 1.0)
        else:
            marker, normal = 0, (0.0, 0.0, 1.0)
        for i in range(nx):
            for j in range(ny):
                p00, p10 = nodes[(i, j, k)], nodes[(i + 1, j, k)]
                p11, p01 = nodes[(i + 1, j + 1, k)], nodes[(i, j + 1, k)]
                add_oriented_triangle(registry, facets, (p00, p10, p11), normal, marker)
                add_oriented_triangle(registry, facets, (p00, p11, p01), normal, marker)

    for k in range(len(Z_LEVELS) - 1):
        for i in range(nx):
            p00, p10 = nodes[(i, 0, k)], nodes[(i + 1, 0, k)]
            p11, p01 = nodes[(i + 1, 0, k + 1)], nodes[(i, 0, k + 1)]
            add_oriented_triangle(registry, facets, (p00, p10, p11), (0.0, -1.0, 0.0), BOUNDARY_MARKERS["south"])
            add_oriented_triangle(registry, facets, (p00, p11, p01), (0.0, -1.0, 0.0), BOUNDARY_MARKERS["south"])
            p00, p10 = nodes[(i, ny, k)], nodes[(i + 1, ny, k)]
            p11, p01 = nodes[(i + 1, ny, k + 1)], nodes[(i, ny, k + 1)]
            add_oriented_triangle(registry, facets, (p00, p10, p11), (0.0, 1.0, 0.0), BOUNDARY_MARKERS["north"])
            add_oriented_triangle(registry, facets, (p00, p11, p01), (0.0, 1.0, 0.0), BOUNDARY_MARKERS["north"])
        for j in range(ny):
            p00, p10 = nodes[(0, j, k)], nodes[(0, j + 1, k)]
            p11, p01 = nodes[(0, j + 1, k + 1)], nodes[(0, j, k + 1)]
            add_oriented_triangle(registry, facets, (p00, p10, p11), (-1.0, 0.0, 0.0), BOUNDARY_MARKERS["west"])
            add_oriented_triangle(registry, facets, (p00, p11, p01), (-1.0, 0.0, 0.0), BOUNDARY_MARKERS["west"])
            p00, p10 = nodes[(nx, j, k)], nodes[(nx, j + 1, k)]
            p11, p01 = nodes[(nx, j + 1, k + 1)], nodes[(nx, j, k + 1)]
            add_oriented_triangle(registry, facets, (p00, p10, p11), (1.0, 0.0, 0.0), BOUNDARY_MARKERS["east"])
            add_oriented_triangle(registry, facets, (p00, p11, p01), (1.0, 0.0, 0.0), BOUNDARY_MARKERS["east"])

    return x_values, y_values


def write_poly(path: Path, registry: PointRegistry, facets: Sequence[Facet], regions: Sequence[Region]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# North Avant / Bartlesville matrix PLC with tag-only HEC\n")
        handle.write("# The injection well is no longer meshed as a cylindrical borehole.\n")
        handle.write("# It is handled later as a small PFLOTRAN source region via wellbore.vset.\n\n")
        handle.write("# Part 1 - node list\n")
        handle.write("{} 3 0 0\n".format(len(registry.points)))
        for point_id, xyz in enumerate(registry.points, start=1):
            handle.write("{:d} {:.10f} {:.10f} {:.10f}\n".format(point_id, xyz[0], xyz[1], xyz[2]))
        handle.write("\n# Part 2 - facet list\n")
        handle.write("{} 1\n".format(len(facets)))
        for facet in facets:
            handle.write("1 0 {}\n".format(facet.marker))
            handle.write("3 {} {} {}\n".format(facet.point_ids[0], facet.point_ids[1], facet.point_ids[2]))
        handle.write("\n# Part 3 - hole list\n0\n")
        handle.write("\n# Part 4 - region list\n")
        handle.write("{}\n".format(len(regions)))
        for index, region in enumerate(regions, start=1):
            x_value, y_value, z_value = region.point
            handle.write("{:d} {:.10f} {:.10f} {:.10f} {:d}\n".format(index, x_value, y_value, z_value, region.attribute))


def write_sidecars(mesh_prefix: str, point_count: int, facet_count: int, x_values: np.ndarray, y_values: np.ndarray) -> None:
    length_axis, width_axis, up_axis = hec_axes()
    tag_u = np.arange(-280.0, 280.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)
    tag_v = np.arange(-140.0, 140.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)

    geometry = {
        "domain": {"min": DOMAIN_MIN.tolist(), "max": DOMAIN_MAX.tolist(), "size": DOMAIN_SIZE.tolist()},
        "layers": [asdict(layer) for layer in LAYERS],
        "vertical_bands": [asdict(band) for band in VERTICAL_BANDS],
        "z_levels_m": list(Z_LEVELS),
        "meshing": {
            "strategy": "matrix-only layered PLC with tag-only HEC",
            "base_x_axis_m": x_values.tolist(),
            "base_y_axis_m": y_values.tolist(),
            "rotated_hec_tag_lattice_step_m": ROTATED_TAG_LATTICE_STEP_M,
            "tetgen_flags": DEFAULT_TETGEN_FLAGS,
        },
        "hec": {
            "name": HEC_NAME,
            "material_id": HEC_MATERIAL_ID,
            "host_material_id": HEC_HOST_MATERIAL_ID,
            "representation": "tag_only_on_rotated_matrix_lattice",
            "center": HEC_CENTER.tolist(),
            "length_m": HEC_LENGTH_M,
            "width_m": HEC_WIDTH_M,
            "thickness_m": HEC_THICKNESS_M,
            "bottom_z_m": HEC_CENTER[2] - 0.5 * HEC_THICKNESS_M,
            "top_z_m": HEC_CENTER[2] + 0.5 * HEC_THICKNESS_M,
            "azimuth_deg_east_of_north": HEC_AZIMUTH_EAST_OF_NORTH_DEG,
            "dip_deg": 0.0,
            "axes": {
                "length": length_axis.tolist(),
                "width": width_axis.tolist(),
                "normal_up": up_axis.tolist(),
            },
            "tagging": {
                "method": "strict_z530_vertex_centres_inside_exact_oriented_rectangle",
                "expected_vertical_dual_support_m": [527.5, 532.5],
                "expected_tagged_vertex_count": int(tag_u.size * tag_v.size),
            },
        },
        "plc": {
            "point_count": point_count,
            "facet_count": facet_count,
            "holes": 0,
            "regions": 4,
            "contains_hec_facets": False,
            "contains_hec_region": False,
            "contains_borehole_facets": False,
            "contains_borehole_regions": False,
        },
        "boundary_markers": BOUNDARY_MARKERS,
    }

    Path("{}_geometry.json".format(mesh_prefix)).write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")

    with Path("{}_hec_tag_geometry.xyz".format(mesh_prefix)).open("w", encoding="utf-8") as handle:
        handle.write("# Exact HEC prism corners; diagnostic only, not PLC facets.\n# id x_m y_m z_m\n")
        for index, corner in enumerate(hec_corners(), start=1):
            handle.write("{:d} {:.10f} {:.10f} {:.10f}\n".format(index, corner[0], corner[1], corner[2]))

    with Path("{}_vertical_grading.csv".format(mesh_prefix)).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "z_min_m", "z_max_m", "vertical_thickness_m", "geological_layer", "note"])
        writer.writeheader()
        for band in VERTICAL_BANDS:
            writer.writerow({
                "label": band.label,
                "z_min_m": "{:.6f}".format(band.z_min),
                "z_max_m": "{:.6f}".format(band.z_max),
                "vertical_thickness_m": "{:.6f}".format(band.z_max - band.z_min),
                "geological_layer": band.geological_layer,
                "note": band.note,
            })


def build_geometry(mesh_prefix: str) -> Path:
    registry = PointRegistry()
    facets: List[Facet] = []

    x_values, y_values = build_matrix_surface_plc(registry, facets)
    refinement_counts = add_hierarchical_refinement(registry)
    print("    local refinement counts:", refinement_counts)
    regions = (
        Region(np.array([5000.0, 5000.0, 500.0]), 1, "overburden"),
        Region(np.array([5000.0, 5000.0, 210.0]), 2, "bartlesville_sand"),
        Region(np.array([5000.0, 5000.0, 235.0]), 3, "basal_layer"),
        Region(np.array([5000.0, 5000.0, 100.0]), 4, "underburden"),
    )

    poly_path = Path("{}.poly".format(mesh_prefix))
    write_poly(poly_path, registry, facets, regions)
    write_sidecars(mesh_prefix, len(registry.points), len(facets), x_values, y_values)

    print("\n--> Wrote Bartlesville PLC with tag-only HEC")
    print("    poly file            : {}".format(poly_path))
    print("    geometry JSON        : {}_geometry.json".format(mesh_prefix))
    print("    HEC diagnostic       : {}_hec_tag_geometry.xyz".format(mesh_prefix))
    print("    vertical profile     : {}_vertical_grading.csv".format(mesh_prefix))
    print("    points               : {}".format(len(registry.points)))
    print("    facets               : {}".format(len(facets)))
    print("    regions              : {}".format(len(regions)))

    return poly_path


def run_tetgen(tetgen_exe: str, poly_path: Path, diagnose: bool) -> None:
    flags = os.environ.get("BARTLESVILLE_TETGEN_FLAGS", DEFAULT_TETGEN_FLAGS).strip()
    if not flags:
        raise ValueError("BARTLESVILLE_TETGEN_FLAGS cannot be empty.")
    if "a" in flags:
        raise ValueError("Do not use TetGen -a in this source-region workflow.")
    if diagnose and "d" not in flags:
        flags += "d"

    command = [tetgen_exe, *shlex.split(flags), str(poly_path)]
    print("\n--> Running TetGen")
    print("    HEC PLC entities       : 0 points, 0 facets, 0 regions (tag-only)")
    print("    borehole PLC entities : removed; handled later as wellbore.vset")
    print("    TetGen flags          :", flags)
    print("CMD:", " ".join(shlex.quote(token) for token in command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bartlesville PLC with tag-only HEC and no borehole mesh zones.")
    parser.add_argument("mesh_prefix", help="Output mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("tetgen_exe", nargs="?", help="TetGen executable path unless --write-only is used")
    parser.add_argument("--write-only", action="store_true", help="Write .poly and sidecars without TetGen")
    parser.add_argument("--diagnose", action="store_true", help="Append TetGen -d diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix[:-5] if args.mesh_prefix.endswith(".poly") else args.mesh_prefix
    poly_path = build_geometry(prefix)
    if args.write_only:
        print("\n--> --write-only selected; TetGen was not run.\n")
        return
    if not args.tetgen_exe:
        raise SystemExit("ERROR: provide <tetgen_exe>, or use --write-only.")
    run_tetgen(args.tetgen_exe, poly_path, args.diagnose)
    print("\n--> TetGen completed successfully.\n")


if __name__ == "__main__":
    main()
