#!/usr/bin/env python3
"""Build the North Avant / Bartlesville TetGen PLC with a tag-only HEC and
five locally resolved, solid borehole *mesh zones* with smooth cylindrical refinement halos.

Run through workflow.py:
    python3 workflow.py

Important representation choice
--------------------------------
The HEC and boreholes are deliberately NOT closed internal PLC shells.  The
previous shell approach intersected the layered planar facets and TetGen
stopped with a self-intersection. Here the .poly adds ordinary, high-density
matrix points along five cylindrical borehole lattices. After TetGen, the
corresponding existing mesh vertices are assigned materials 6--10.

The injection borehole reaches the HEC top. The four green strainmeters are
well away from the HEC footprint in plan view and only occupy the upper 100 m
of the model (z=650--750 m).

Therefore the boreholes are:
  * solid material-tagged mesh zones, never voids;
  * not Part-3 TetGen holes;
  * not internal PLC surface/region entities;
  * explicitly resolved by local points written in Part 1 of the .poly.

This is the robust analogue of the working tag-only rotated HEC method.
"""

from __future__ import annotations

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


LAYERS: Tuple[GeologicalLayer, ...] = (
    GeologicalLayer(1, 4, "underburden", 0.0, 200.0),
    GeologicalLayer(2, 2, "bartlesville_sand", 200.0, 220.0),
    GeologicalLayer(3, 3, "basal_layer", 220.0, 250.0),
    GeologicalLayer(4, 1, "overburden", 250.0, 750.0),
)

# -----------------------------------------------------------------------------
# Tag-only HEC. x=east and y=north. The HEC is horizontal and its 580 m axis is
# 5 degrees east (+x) of north (+y).
# -----------------------------------------------------------------------------
HEC_NAME = "bartlesville_hec"
HEC_MATERIAL_ID = 5
HEC_HOST_MATERIAL_ID = 1
HEC_CENTER = np.array([5000.0, 5000.0, 530.0], dtype=float)
HEC_LENGTH_M = 580.0
HEC_WIDTH_M = 300.0
HEC_THICKNESS_M = 5.0
HEC_AZIMUTH_EAST_OF_NORTH_DEG = 5.0

# -----------------------------------------------------------------------------
# Matrix grid. This is the already-working rotated central lattice that gives
# the HEC a visibly 5-degree rotated plan-view footprint.
# -----------------------------------------------------------------------------
MATRIX_COARSE_STEP_M = 500.0
MATRIX_TRANSITION_STEP_M = 200.0
ROTATED_TAG_LATTICE_STEP_M = 20.0
X_FINE_START, X_FINE_END = 4800.0, 5200.0
Y_FINE_START, Y_FINE_END = 4680.0, 5320.0
ROTATION_INNER_HALF_X = 220.0
ROTATION_INNER_HALF_Y = 360.0
ROTATION_OUTER_HALF_X = 800.0
ROTATION_OUTER_HALF_Y = 1000.0

# -----------------------------------------------------------------------------
# Five borehole mesh-zone definitions.
# The grey injection borehole reaches the HEC top at z=532.5 m and extends to
# the domain top. The four green strainmeters are completely outside the HEC
# footprint in plan view, and only extend 100 m downward from the model top:
# z=650--750 m. They therefore do NOT touch or intersect the HEC.
#
# All target tags reach the model top (z=750 m). To avoid putting unconstrained
# extra points directly on the already-triangulated top boundary, the final
# local point ring is at z=749 m. For strainmeters, the first local point ring
# is at z=651 m so that it does not lie exactly on the z=650 m matrix-plane
# facet; its material-tag interval still begins at z=650 m.
#
# Borehole refinement is deliberately much denser than the former three-ring
# lattice.  Each cylinder gets several concentric radial rings and a smaller
# vertical increment.  Adjacent z-planes are azimuthally staggered so TetGen
# does not build long, visually obvious radial wedges around the wells.
# -----------------------------------------------------------------------------
INJECTION_RADIUS_M = 5.0
STRAINMETER_RADIUS_M = 2.0
HEC_TOP_Z_M = HEC_CENTER[2] + 0.5 * HEC_THICKNESS_M
INJECTION_BOTTOM_Z_M = HEC_TOP_Z_M
STRAINMETER_BOTTOM_Z_M = 650.0
BOREHOLE_TAG_TOP_Z_M = 750.0
# Do not add unconstrained local point rings directly on the top boundary or
# on the z=650 m grading plane. The tags still represent the closed intervals.
BOREHOLE_LATTICE_TOP_Z_M = 749.0
INJECTION_LATTICE_BOTTOM_Z_M = INJECTION_BOTTOM_Z_M
STRAINMETER_LATTICE_BOTTOM_Z_M = 651.0

# Borehole local refinement is now a continuous, concentric cylindrical-shell
# point field.  The prior sparse polar collars had independent coarse z steps
# in their outer rings, which allowed long star-shaped tetrahedra to bridge to
# the ambient matrix.  Here every shell is sampled on a common local z lattice
# and the shell radius increases gradually out to a broad halo.  This produces
# a smooth sequence of small near-well cells -> intermediate collar cells ->
# coarse matrix cells without adding any internal PLC facets or holes.
#
# Entries are explicit shell radii in metres.  The exact physical tag boundary
# is included at r=R for each well.  The outer halos are deliberately large:
# 700 m for the injection well and 650 m for each strainmeter, so the local
# field blends into the 500 m background matrix rather than producing a star
# of long Delaunay edges.
INJECTION_LOCAL_Z_STEP_M = 2.0
STRAINMETER_LOCAL_Z_STEP_M = 2.0
INJECTION_SHELL_RADII_M: Tuple[float, ...] = (
    0.0, 0.40, 0.80, 1.20, 1.70, 2.30, 3.00, 3.80, 4.40, 5.00,
    6.00, 7.30, 9.00, 11.0, 14.0, 18.0, 23.0, 30.0, 40.0, 54.0,
    72.0, 96.0, 128.0, 170.0, 225.0, 300.0, 400.0, 525.0, 700.0,
)
STRAINMETER_SHELL_RADII_M: Tuple[float, ...] = (
    0.0, 0.18, 0.36, 0.55, 0.78, 1.05, 1.35, 1.65, 2.00,
    2.50, 3.20, 4.20, 5.50, 7.50, 10.0, 13.5, 18.0, 24.0,
    32.0, 43.0, 58.0, 78.0, 105.0, 140.0, 190.0, 255.0,
    340.0, 455.0, 650.0,
)

# The target tangential spacing grows continuously with distance from the
# well.  It governs the angular resolution of each cylindrical shell.  The
# lower cap guarantees clean circular cells near r=R; the upper cap prevents
# unnecessary far-field points while retaining a dense transition halo.
def shell_target_spacing(radius_m: float, kind: str) -> float:
    if kind == "injection":
        return min(70.0, max(0.75, 0.12 * radius_m + 0.75))
    return min(65.0, max(0.45, 0.16 * radius_m + 0.45))


def shell_segment_count(radius_m: float, kind: str) -> int:
    if radius_m <= 0.0:
        return 1
    target = shell_target_spacing(radius_m, kind)
    # Extra angular resolution at and immediately around the physical borehole
    # radius makes the material-6--10 cells visibly circular rather than a
    # coarse polygon.  Farther from the well, the segment count grows smoothly
    # from the requested tangential spacing.
    if kind == "injection" and radius_m <= 10.0:
        minimum_segments = 36
    elif kind == "strainmeter" and radius_m <= 4.0:
        minimum_segments = 24
    else:
        minimum_segments = 16
    return max(minimum_segments, int(math.ceil(2.0 * math.pi * radius_m / target)))

INJECTION_MATERIAL_ID = 6
STRAINMETER_MATERIAL_IDS = (7, 8, 9, 10)
# Markers are metadata only. This robust point-lattice implementation does not
# create borehole surface facets.
INJECTION_SURFACE_MARKER = 7
STRAINMETER_SURFACE_MARKERS = (8, 9, 10, 11)


@dataclass(frozen=True)
class Borehole:
    name: str
    kind: str
    color_hint: str
    material_id: int
    surface_marker: int
    center_xy: Tuple[float, float]
    radius_m: float
    z_bottom_m: float
    z_top_m: float
    lattice_bottom_z_m: float
    lattice_top_z_m: float
    vertical_step_m: float
    # Coordinates in the HEC-local frame: u along the 580 m long axis and v
    # along the 300 m short axis. They document whether a well lies inside or
    # outside the HEC footprint.
    hec_local_uv_m: Tuple[float, float]
    terminates_on_hec_top: bool
    is_outside_hec_footprint: bool


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
    "top": 1, "bottom": 2, "north": 3, "south": 4, "east": 5, "west": 6,
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


def local_hec_xy(u_m: float, v_m: float) -> Tuple[float, float]:
    length_axis, width_axis, _ = hec_axes()
    xy = HEC_CENTER[:2] + u_m * length_axis[:2] + v_m * width_axis[:2]
    return float(xy[0]), float(xy[1])


# Four shallow strainmeters far from the HEC. The coordinates are written in
# the HEC-local (u, v) frame only so their arrangement remains symmetric with
# respect to the 5-degree HEC orientation. Each is ~2.3 km from the HEC centre.
# The HEC footprint is |u| <= 290 m and |v| <= 150 m, so the wells are clearly
# separated in plan view and cannot be mistaken for HEC contacts.
STRAINMETER_HEC_LOCAL_UV_M: Tuple[Tuple[float, float], ...] = (
    (+1800.0, +1400.0),
    (+1800.0, -1400.0),
    (-1800.0, +1400.0),
    (-1800.0, -1400.0),
)


def build_boreholes() -> Tuple[Borehole, ...]:
    wells: List[Borehole] = [
        Borehole(
            name="injection_borehole", kind="injection", color_hint="grey",
            material_id=INJECTION_MATERIAL_ID, surface_marker=INJECTION_SURFACE_MARKER,
            center_xy=(float(HEC_CENTER[0]), float(HEC_CENTER[1])), radius_m=INJECTION_RADIUS_M,
            z_bottom_m=INJECTION_BOTTOM_Z_M, z_top_m=BOREHOLE_TAG_TOP_Z_M,
            lattice_bottom_z_m=INJECTION_LATTICE_BOTTOM_Z_M, lattice_top_z_m=BOREHOLE_LATTICE_TOP_Z_M,
            vertical_step_m=INJECTION_LOCAL_Z_STEP_M,
            hec_local_uv_m=(0.0, 0.0), terminates_on_hec_top=True, is_outside_hec_footprint=False,
        )
    ]
    # Four green strainmeters are far from the HEC and only occupy z=650--750 m,
    # i.e. the upper 100 m of the model. They do not contact material 5.
    for index, (u_m, v_m) in enumerate(STRAINMETER_HEC_LOCAL_UV_M, start=1):
        wells.append(
            Borehole(
                name=f"strainmeter_{index}", kind="strainmeter", color_hint="green",
                material_id=STRAINMETER_MATERIAL_IDS[index - 1],
                surface_marker=STRAINMETER_SURFACE_MARKERS[index - 1],
                center_xy=local_hec_xy(u_m, v_m), radius_m=STRAINMETER_RADIUS_M,
                z_bottom_m=STRAINMETER_BOTTOM_Z_M, z_top_m=BOREHOLE_TAG_TOP_Z_M,
                lattice_bottom_z_m=STRAINMETER_LATTICE_BOTTOM_Z_M, lattice_top_z_m=BOREHOLE_LATTICE_TOP_Z_M,
                vertical_step_m=STRAINMETER_LOCAL_Z_STEP_M,
                hec_local_uv_m=(u_m, v_m), terminates_on_hec_top=False, is_outside_hec_footprint=True,
            )
        )
    return tuple(wells)


BOREHOLES: Tuple[Borehole, ...] = build_boreholes()


def inclusive_values(start: float, stop: float, step: float) -> List[float]:
    count = int(round((stop - start) / step))
    if not math.isclose(start + count * step, stop, abs_tol=1.0e-9):
        raise ValueError(f"[{start}, {stop}] is not divisible by {step}.")
    return [start + i * step for i in range(count + 1)]


def endpoint_values(start: float, stop: float, step: float) -> List[float]:
    values = [float(start)]
    value = float(start)
    while value + step < stop - 1.0e-9:
        value += step
        values.append(value)
    if not math.isclose(values[-1], stop, abs_tol=1.0e-9):
        values.append(float(stop))
    return values


def unique_sorted(values: Iterable[float]) -> np.ndarray:
    return np.asarray(sorted({round(float(v), 9) for v in values}), dtype=float)


def make_x_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4600.0]
    values += inclusive_values(X_FINE_START, X_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5400.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    if axis[0] != 0.0 or axis[-1] != 10000.0 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("Invalid x-axis construction.")
    return axis


def make_y_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4500.0]
    values += inclusive_values(Y_FINE_START, Y_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5500.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    if axis[0] != 0.0 or axis[-1] != 10000.0 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("Invalid y-axis construction.")
    return axis


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


def add_oriented_triangle(registry: PointRegistry, facets: List[Facet], ids: Sequence[int], normal: Sequence[float], marker: int) -> None:
    value = list(ids)
    a, b, c = (registry.xyz(i) for i in value)
    if float(np.dot(np.cross(b - a, c - a), np.asarray(normal, dtype=float))) < 0.0:
        value[1], value[2] = value[2], value[1]
    facets.append(Facet(tuple(value), marker))


def validate_configuration() -> None:
    tolerance = 1.0e-8
    if not np.isclose(VERTICAL_BANDS[0].z_min, DOMAIN_MIN[2], atol=tolerance):
        raise ValueError("Vertical bands must start at z=0.")
    if not np.isclose(VERTICAL_BANDS[-1].z_max, DOMAIN_MAX[2], atol=tolerance):
        raise ValueError("Vertical bands must end at z=750.")
    previous = VERTICAL_BANDS[0].z_min
    for band in VERTICAL_BANDS:
        if not np.isclose(band.z_min, previous, atol=tolerance):
            raise ValueError(f"Gap or overlap before {band.label}.")
        layer = next(item for item in LAYERS if item.number == band.geological_layer)
        if band.z_min < layer.z_min - tolerance or band.z_max > layer.z_max + tolerance:
            raise ValueError(f"{band.label} lies outside layer {layer.number}.")
        previous = band.z_max
    for level in (525.0, 530.0, 535.0):
        if not any(math.isclose(level, z, abs_tol=tolerance) for z in Z_LEVELS):
            raise ValueError(f"Missing z={level:g} matrix level for the HEC tag.")
    if len(BOREHOLES) != 5:
        raise ValueError("Expected one injection well and four strainmeters.")
    hec_half_length = 0.5 * HEC_LENGTH_M
    hec_half_width = 0.5 * HEC_WIDTH_M
    for well in BOREHOLES:
        if not (250.0 < well.z_bottom_m <= well.lattice_bottom_z_m < well.lattice_top_z_m < well.z_top_m <= DOMAIN_MAX[2]):
            raise ValueError(f"Invalid vertical borehole lattice range: {well.name}")
        if well.radius_m <= 0.0:
            raise ValueError(f"Invalid borehole radius for {well.name}")
        local_u, local_v = well.hec_local_uv_m
        if well.terminates_on_hec_top:
            if not math.isclose(well.z_bottom_m, HEC_TOP_Z_M, abs_tol=tolerance):
                raise ValueError(f"{well.name} must terminate at the HEC top z={HEC_TOP_Z_M:g} m.")
            if abs(local_u) + well.radius_m > hec_half_length - tolerance:
                raise ValueError(f"{well.name} extends beyond the HEC length footprint.")
            if abs(local_v) + well.radius_m > hec_half_width - tolerance:
                raise ValueError(f"{well.name} extends beyond the HEC width footprint.")
        if well.is_outside_hec_footprint:
            fully_outside = (abs(local_u) - well.radius_m > hec_half_length + tolerance) or (abs(local_v) - well.radius_m > hec_half_width + tolerance)
            if not fully_outside:
                raise ValueError(f"{well.name} must lie completely outside the HEC footprint.")
            if math.hypot(local_u, local_v) < 2000.0:
                raise ValueError(f"{well.name} must be at least 2 km from the HEC centre.")
            if not math.isclose(well.z_bottom_m, 650.0, abs_tol=tolerance):
                raise ValueError(f"{well.name} must extend only from z=650 to the top face.")
    for i, first in enumerate(BOREHOLES):
        first_shells = INJECTION_SHELL_RADII_M if first.kind == "injection" else STRAINMETER_SHELL_RADII_M
        first_outer = max(first_shells)
        if not (DOMAIN_MIN[0] < first.center_xy[0] - first_outer and first.center_xy[0] + first_outer < DOMAIN_MAX[0]
                and DOMAIN_MIN[1] < first.center_xy[1] - first_outer and first.center_xy[1] + first_outer < DOMAIN_MAX[1]):
            raise ValueError(f"{first.name} outer refinement collar leaves the domain.")
        for second in BOREHOLES[i + 1:]:
            second_shells = INJECTION_SHELL_RADII_M if second.kind == "injection" else STRAINMETER_SHELL_RADII_M
            second_outer = max(second_shells)
            distance = math.hypot(first.center_xy[0] - second.center_xy[0], first.center_xy[1] - second.center_xy[1])
            if distance <= first_outer + second_outer:
                raise ValueError(f"Refinement collars overlap: {first.name} and {second.name}")


def build_matrix_surface_plc(registry: PointRegistry, facets: List[Facet]) -> Tuple[np.ndarray, np.ndarray]:
    """Build the known working matrix-only layered PLC."""
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
                raise RuntimeError(f"Warped plan-view grid folded at ({i}, {j}).")
    print(f"    minimum warped plan-view triangle area: {min_area:.6g} m^2")

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


def add_borehole_lattice_points(registry: PointRegistry) -> Dict[str, int]:
    """Create smooth, fully resolved cylindrical refinement halos around wells.

    Each borehole is represented only by ordinary Part-1 points.  It has no
    internal PLC wall, cap, region seed, or TetGen hole.  The point field is a
    continuous sequence of concentric cylindrical shells, sampled at a common
    2 m vertical increment.  A golden-ratio azimuthal phase shift between
    consecutive planes and neighbouring shells breaks artificial spoke chains.

    The tag boundary r=R is an explicit shell.  Shells inside r=R resolve the
    solid borehole material; shells outside r=R form a wide, smooth transition
    halo.  The large 650--700 m outer halo avoids a direct jump from the tiny
    well radius to the 500 m background matrix.
    """
    counts: Dict[str, int] = {}
    golden = 0.6180339887498949
    for well in BOREHOLES:
        before = len(registry.points)
        radii = INJECTION_SHELL_RADII_M if well.kind == "injection" else STRAINMETER_SHELL_RADII_M
        z_step = INJECTION_LOCAL_Z_STEP_M if well.kind == "injection" else STRAINMETER_LOCAL_Z_STEP_M
        z_values = endpoint_values(well.lattice_bottom_z_m, well.lattice_top_z_m, z_step)

        # Centreline.  It is included independently so every well has robust
        # material-tag support at the exact well axis.
        for z_value in z_values:
            registry.add((well.center_xy[0], well.center_xy[1], z_value))

        # Cylindrical shells.  The phase changes between adjacent z planes and
        # neighbouring rings.  This makes a locally body-centred / helical
        # point field instead of a stack of aligned radial spokes.
        for shell_index, radius in enumerate(radii[1:], start=1):
            segments = shell_segment_count(radius, well.kind)
            shell_phase = (0.21132486540518713 * shell_index) % 1.0
            for plane_index, z_value in enumerate(z_values):
                plane_phase = (golden * plane_index + shell_phase) % 1.0
                for segment in range(segments):
                    theta = 2.0 * math.pi * (segment + plane_phase) / segments
                    registry.add((
                        well.center_xy[0] + radius * math.cos(theta),
                        well.center_xy[1] + radius * math.sin(theta),
                        z_value,
                    ))
        counts[well.name] = len(registry.points) - before
    return counts


def write_poly(path: Path, registry: PointRegistry, facets: Sequence[Facet], regions: Sequence[Region]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# North Avant / Bartlesville matrix PLC with tag-only HEC and locally resolved borehole mesh zones\n")
        handle.write("# HEC and boreholes add Part-1 mesh points only; neither creates PLC facets, regions, nor TetGen holes.\n\n")
        handle.write("# Part 1 - node list\n")
        handle.write(f"{len(registry.points)} 3 0 0\n")
        for point_id, xyz in enumerate(registry.points, start=1):
            handle.write(f"{point_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
        handle.write("\n# Part 2 - facet list\n")
        handle.write(f"{len(facets)} 1\n")
        for facet in facets:
            handle.write(f"1 0 {facet.marker}\n")
            handle.write(f"3 {facet.point_ids[0]} {facet.point_ids[1]} {facet.point_ids[2]}\n")
        handle.write("\n# Part 3 - hole list\n0\n")
        handle.write("\n# Part 4 - region list\n")
        handle.write(f"{len(regions)}\n")
        for index, region in enumerate(regions, start=1):
            x_value, y_value, z_value = region.point
            handle.write(f"{index} {x_value:.10f} {y_value:.10f} {z_value:.10f} {region.attribute}\n")


def write_sidecars(mesh_prefix: str, point_count: int, facet_count: int, x_values: np.ndarray, y_values: np.ndarray, borehole_point_counts: Dict[str, int]) -> None:
    length_axis, width_axis, up_axis = hec_axes()
    tag_u = np.arange(-280.0, 280.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)
    tag_v = np.arange(-140.0, 140.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)
    geometry = {
        "domain": {"min": DOMAIN_MIN.tolist(), "max": DOMAIN_MAX.tolist(), "size": DOMAIN_SIZE.tolist()},
        "layers": [asdict(layer) for layer in LAYERS],
        "vertical_bands": [asdict(band) for band in VERTICAL_BANDS],
        "z_levels_m": list(Z_LEVELS),
        "meshing": {
            "strategy": "matrix-only layered PLC plus smooth radially graded Part-1 cylindrical shell lattices for the HEC tag and five solid borehole tag zones",
            "base_x_axis_m": x_values.tolist(), "base_y_axis_m": y_values.tolist(),
            "rotated_hec_tag_lattice_step_m": ROTATED_TAG_LATTICE_STEP_M,
            "tetgen_flags": DEFAULT_TETGEN_FLAGS, "no_tetgen_a": True,
            "borehole_local_refinement": {
                "method": "continuous_cylindrical_shell_lattice_with_golden_phase_stagger",
                "common_injection_vertical_step_m": INJECTION_LOCAL_Z_STEP_M,
                "common_strainmeter_vertical_step_m": STRAINMETER_LOCAL_Z_STEP_M,
                "injection_shell_radii_m": list(INJECTION_SHELL_RADII_M),
                "strainmeter_shell_radii_m": list(STRAINMETER_SHELL_RADII_M),
                "injection_outer_halo_radius_m": max(INJECTION_SHELL_RADII_M),
                "strainmeter_outer_halo_radius_m": max(STRAINMETER_SHELL_RADII_M),
            },
        },
        "hec": {
            "name": HEC_NAME, "material_id": HEC_MATERIAL_ID, "host_material_id": HEC_HOST_MATERIAL_ID,
            "representation": "tag_only_on_rotated_matrix_lattice", "center": HEC_CENTER.tolist(),
            "length_m": HEC_LENGTH_M, "width_m": HEC_WIDTH_M, "thickness_m": HEC_THICKNESS_M,
            "bottom_z_m": HEC_CENTER[2] - 0.5 * HEC_THICKNESS_M, "top_z_m": HEC_CENTER[2] + 0.5 * HEC_THICKNESS_M,
            "azimuth_deg_east_of_north": HEC_AZIMUTH_EAST_OF_NORTH_DEG, "dip_deg": 0.0,
            "axes": {"length": length_axis.tolist(), "width": width_axis.tolist(), "normal_up": up_axis.tolist()},
            "tagging": {
                "method": "strict_z530_vertex_centres_inside_exact_oriented_rectangle",
                "expected_vertical_dual_support_m": [527.5, 532.5],
                "expected_tagged_vertex_count": int(tag_u.size * tag_v.size),
            },
        },
        "boreholes": [
            {
                **asdict(well),
                "representation": "tagged_solid_cylinder_on_radially_graded_part1_polar_lattice",
                "tetgen_hole": False, "plc_facets": False, "plc_region": False,
                "local_lattice_point_count": borehole_point_counts[well.name],
                "tag_top_reaches_domain_top_m": True,
                "first_explicit_local_lattice_z_m": well.lattice_bottom_z_m,
                "last_explicit_local_lattice_z_m": well.lattice_top_z_m,
                "contacts_hec": bool(well.terminates_on_hec_top),
                "outside_hec_footprint": bool(well.is_outside_hec_footprint),
            }
            for well in BOREHOLES
        ],
        "plc": {
            "point_count": point_count, "facet_count": facet_count, "holes": 0, "regions": 4,
            "contains_hec_facets": False, "contains_hec_region": False,
            "contains_borehole_facets": False, "contains_borehole_regions": False,
            "contains_borehole_local_mesh_points": True,
        },
        "boundary_markers": BOUNDARY_MARKERS,
    }
    Path(f"{mesh_prefix}_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
    with Path(f"{mesh_prefix}_hec_tag_geometry.xyz").open("w", encoding="utf-8") as handle:
        handle.write("# Exact HEC prism corners; diagnostic only, not PLC facets.\n# id x_m y_m z_m\n")
        for index, corner in enumerate(hec_corners(), start=1):
            handle.write(f"{index} {corner[0]:.10f} {corner[1]:.10f} {corner[2]:.10f}\n")
    with Path(f"{mesh_prefix}_boreholes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "kind", "material_id", "color_hint", "center_x_m", "center_y_m", "radius_m", "bottom_z_m", "tag_top_z_m", "lattice_bottom_z_m", "lattice_top_z_m", "vertical_step_m", "hec_local_u_m", "hec_local_v_m", "terminates_on_hec_top", "is_outside_hec_footprint", "local_lattice_points"])
        for well in BOREHOLES:
            writer.writerow([well.name, well.kind, well.material_id, well.color_hint, f"{well.center_xy[0]:.10f}", f"{well.center_xy[1]:.10f}", f"{well.radius_m:.10f}", f"{well.z_bottom_m:.10f}", f"{well.z_top_m:.10f}", f"{well.lattice_bottom_z_m:.10f}", f"{well.lattice_top_z_m:.10f}", f"{well.vertical_step_m:.10f}", f"{well.hec_local_uv_m[0]:.10f}", f"{well.hec_local_uv_m[1]:.10f}", well.terminates_on_hec_top, well.is_outside_hec_footprint, borehole_point_counts[well.name]])
    with Path(f"{mesh_prefix}_borehole_refinement_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["well_kind", "radius_m", "shell_index", "shell_radius_m", "radius_factor", "azimuthal_points", "target_tangential_spacing_m", "common_vertical_step_m", "role"])
        for kind, radius, shells, z_step in (
            ("injection", INJECTION_RADIUS_M, INJECTION_SHELL_RADII_M, INJECTION_LOCAL_Z_STEP_M),
            ("strainmeter", STRAINMETER_RADIUS_M, STRAINMETER_SHELL_RADII_M, STRAINMETER_LOCAL_Z_STEP_M),
        ):
            for shell_index, shell_radius in enumerate(shells):
                if shell_index == 0:
                    segments = 1
                    role = "axis"
                else:
                    segments = shell_segment_count(shell_radius, kind)
                    if math.isclose(shell_radius, radius, abs_tol=1.0e-12):
                        role = "exact_material_tag_boundary"
                    elif shell_radius < radius:
                        role = "solid_well_interior_resolution"
                    else:
                        role = "smooth_transition_halo"
                writer.writerow([kind, f"{radius:.6f}", shell_index, f"{shell_radius:.6f}", f"{shell_radius / radius:.6f}", segments, f"{shell_target_spacing(shell_radius, kind):.6f}", f"{z_step:.6f}", role])
    with Path(f"{mesh_prefix}_vertical_grading.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "z_min_m", "z_max_m", "vertical_thickness_m", "geological_layer", "note"])
        writer.writeheader()
        for band in VERTICAL_BANDS:
            writer.writerow({"label": band.label, "z_min_m": f"{band.z_min:.6f}", "z_max_m": f"{band.z_max:.6f}", "vertical_thickness_m": f"{band.z_max - band.z_min:.6f}", "geological_layer": band.geological_layer, "note": band.note})


def build_geometry(mesh_prefix: str) -> Tuple[Path, Dict[str, int]]:
    validate_configuration()
    registry = PointRegistry()
    facets: List[Facet] = []
    x_values, y_values = build_matrix_surface_plc(registry, facets)
    borehole_point_counts = add_borehole_lattice_points(registry)
    regions = (
        Region(np.array([5000.0, 5000.0, 500.0]), 1, "overburden"),
        Region(np.array([5000.0, 5000.0, 210.0]), 2, "bartlesville_sand"),
        Region(np.array([5000.0, 5000.0, 235.0]), 3, "basal_layer"),
        Region(np.array([5000.0, 5000.0, 100.0]), 4, "underburden"),
    )
    poly_path = Path(f"{mesh_prefix}.poly")
    write_poly(poly_path, registry, facets, regions)
    write_sidecars(mesh_prefix, len(registry.points), len(facets), x_values, y_values, borehole_point_counts)
    return poly_path, {
        "points": len(registry.points), "facets": len(facets), "hec_plc_points": 0,
        "hec_plc_facets": 0, "hec_plc_regions": 0,
        "borehole_local_mesh_points": sum(borehole_point_counts.values()),
        "borehole_plc_facets": 0, "borehole_plc_regions": 0, "holes": 0, "regions": len(regions),
    }


def run_tetgen(tetgen_exe: str, poly_path: Path, diagnose: bool) -> None:
    flags = os.environ.get("BARTLESVILLE_TETGEN_FLAGS", DEFAULT_TETGEN_FLAGS).strip()
    if not flags:
        raise ValueError("BARTLESVILLE_TETGEN_FLAGS cannot be empty.")
    if "a" in flags:
        raise ValueError("Do not use TetGen -a in this low-cell workflow.")
    if diagnose and "d" not in flags:
        flags += "d"
    command = [tetgen_exe, *shlex.split(flags), str(poly_path)]
    print("\n--> Running TetGen")
    print("    HEC PLC entities       : 0 points, 0 facets, 0 regions (tag-only)")
    print("    borehole PLC entities : 0 facets, 0 regions, 0 holes")
    print("    borehole mesh zones   : continuous cylindrical-shell Part-1 refinement halos; materials 6--10 assigned after TetGen")
    print(f"    borehole refinement   : common {INJECTION_LOCAL_Z_STEP_M:g}-m local z spacing; {len(INJECTION_SHELL_RADII_M)} injection / {len(STRAINMETER_SHELL_RADII_M)} strainmeter shells")
    print(f"    refinement halos      : injection r<= {max(INJECTION_SHELL_RADII_M):g} m; strainmeter r<= {max(STRAINMETER_SHELL_RADII_M):g} m")
    print(f"    injection borehole    : R={INJECTION_RADIUS_M:g} m; z={INJECTION_BOTTOM_Z_M:g}--{BOREHOLE_TAG_TOP_Z_M:g} m; centred on HEC")
    print(f"    strainmeter boreholes : 4 x R={STRAINMETER_RADIUS_M:g} m; z={STRAINMETER_BOTTOM_Z_M:g}--{BOREHOLE_TAG_TOP_Z_M:g} m; ~2.3 km from HEC centre; upper 100 m only")
    print("    TetGen flags          :", flags)
    print("CMD:", " ".join(shlex.quote(token) for token in command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bartlesville PLC with tag-only HEC and smoothly graded borehole mesh zones.")
    parser.add_argument("mesh_prefix", help="Output mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("tetgen_exe", nargs="?", help="TetGen executable path unless --write-only is used")
    parser.add_argument("--write-only", action="store_true", help="Write .poly and sidecars without TetGen")
    parser.add_argument("--diagnose", action="store_true", help="Append TetGen -d diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix.removesuffix(".poly")
    poly_path, counts = build_geometry(prefix)
    print("\n--> Wrote Bartlesville PLC with tag-only HEC and five smoothly graded local borehole mesh zones")
    print(f"    poly file            : {poly_path}")
    print(f"    geometry JSON        : {prefix}_geometry.json")
    print(f"    HEC diagnostic       : {prefix}_hec_tag_geometry.xyz")
    print(f"    borehole definition  : {prefix}_boreholes.csv")
    print(f"    vertical profile     : {prefix}_vertical_grading.csv")
    print(f"    borehole profile    : {prefix}_borehole_refinement_profile.csv")
    for name, value in counts.items():
        print(f"    {name:27s}: {value}")
    if args.write_only:
        print("\n--> --write-only selected; TetGen was not run.\n")
        return
    if not args.tetgen_exe:
        raise SystemExit("ERROR: provide <tetgen_exe>, or use --write-only.")
    run_tetgen(args.tetgen_exe, poly_path, args.diagnose)
    print("\n--> TetGen completed successfully.\n")


if __name__ == "__main__":
    main()
