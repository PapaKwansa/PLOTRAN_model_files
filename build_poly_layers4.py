#!/usr/bin/env python3
"""Build a Bartlesville / North Avant TetGen PLC with graded tetrahedral refinement.

Geometry convention
-------------------
The vertical coordinate is elevation relative to the model top:

    top face    z =    0 m
    bottom face z = -750 m

The former 0--750 m depth-positive geometry has therefore been translated with
``z_new = z_old - 750``.  The HEC centre is at z=-220 m, its top is at
z=-217.5 m, the shallow strainmeters are at z=-720 m, and AVN31 is at
z=-230 m.

Local-mesh strategy
-------------------
The previous scripts placed independent random/BCC refinement bands around a
very small cylinder.  TetGen could connect those unrelated bands with long
radial edges, producing star-shaped and highly irregular dual cells.

This revision uses a body-fitted, smoothly graded point layout:

* the injection well is a closed 8-sided cylindrical PLC with an axial surface
  spacing comparable to its circumference edge length;
* surrounding refinement points lie on staggered coaxial tube shells whose
  radial, tangential, and axial spacings increase together;
* strainmeters are closed subdivided-icosahedron PLCs, avoiding polar triangles;
* surrounding strainmeter refinement uses rotated geodesic shells with a
  nearly constant radial-to-tangential size ratio;
* no centerline point chain and no radial center spokes are inserted;
* TetGen quality refinement is deliberately disabled by default.  On this
  multi-scale 10-km model, ``-q1.5`` inserts hundreds of thousands of Steiner
  points on the geological interfaces and can create several million
  tetrahedra.  The default constrained-Delaunay run preserves the graded input
  layout without that global refinement explosion.

The HEC remains tag-only and is described in the JSON sidecar.  Geological
material interfaces are explicit horizontal PLC facets.  Other vertical
refinement levels are Part-1 points only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# Domain and material geometry: top z=0, bottom z=-750.
# =============================================================================
DOMAIN_MIN = np.array([0.0, 0.0, -750.0], dtype=float)
DOMAIN_MAX = np.array([10000.0, 10000.0, 0.0], dtype=float)
DOMAIN_SIZE = DOMAIN_MAX - DOMAIN_MIN


@dataclass(frozen=True)
class GeologicalLayer:
    number: int
    material_id: int
    name: str
    z_min: float
    z_max: float


LAYERS: Tuple[GeologicalLayer, ...] = (
    GeologicalLayer(1, 4, "underburden", -750.0, -550.0),
    GeologicalLayer(2, 2, "bartlesville_sand", -550.0, -530.0),
    GeologicalLayer(3, 3, "basal_layer", -530.0, -500.0),
    GeologicalLayer(4, 1, "overburden", -500.0, 0.0),
)

# =============================================================================
# HEC: tag-only.
# =============================================================================
HEC_NAME = "bartlesville_hec"
HEC_MATERIAL_ID = 5
HEC_HOST_MATERIAL_ID = 1
HEC_CENTER = np.array([5000.0, 5000.0, -220.0], dtype=float)
HEC_LENGTH_M = 580.0
HEC_WIDTH_M = 300.0
HEC_THICKNESS_M = 5.0
HEC_AZIMUTH_EAST_OF_NORTH_DEG = 5.0
HEC_BOTTOM_Z_M = HEC_CENTER[2] - 0.5 * HEC_THICKNESS_M
HEC_TOP_Z_M = HEC_CENTER[2] + 0.5 * HEC_THICKNESS_M

# =============================================================================
# Matrix point lattice.
# =============================================================================
MATRIX_COARSE_STEP_M = 250.0
ROTATED_TAG_LATTICE_STEP_M = 20.0
X_FINE_START, X_FINE_END = 4800.0, 5200.0
Y_FINE_START, Y_FINE_END = 4680.0, 5320.0
ROTATION_INNER_HALF_X = 220.0
ROTATION_INNER_HALF_Y = 360.0
ROTATION_OUTER_HALF_X = 800.0
ROTATION_OUTER_HALF_Y = 1000.0


@dataclass(frozen=True)
class VerticalBand:
    label: str
    z_min: float
    z_max: float
    geological_layer: int
    note: str


VERTICAL_BANDS: Tuple[VerticalBand, ...] = (
    VerticalBand("L1_bottom_coarse", -750.0, -650.0, 1, "largest cells near bottom boundary"),
    VerticalBand("L1_lower_transition", -650.0, -590.0, 1, "transition toward refined beds"),
    VerticalBand("L1_mid_transition", -590.0, -565.0, 1, "transition toward refined beds"),
    VerticalBand("L1_upper_transition", -565.0, -550.0, 1, "matches fine spacing near layer 2"),
    VerticalBand("L2_lower_refined", -550.0, -540.0, 2, "refined Bartlesville interval"),
    VerticalBand("L2_upper_refined", -540.0, -530.0, 2, "refined Bartlesville interval"),
    VerticalBand("L3_lower_refined", -530.0, -515.0, 3, "basal layer refined band"),
    VerticalBand("L3_upper_refined", -515.0, -500.0, 3, "basal layer refined band"),
    VerticalBand("L4_lower_refined", -500.0, -400.0, 4, "coarsening upward"),
    VerticalBand("L4_transition_1", -400.0, -250.0, 4, "coarsening toward HEC"),
    VerticalBand("L4_hec_lower_matrix", -250.0, -225.0, 4, "matrix level below HEC"),
    VerticalBand("L4_hec_center_lower", -225.0, -220.0, 4, "HEC centre support"),
    VerticalBand("L4_hec_center_upper", -220.0, -215.0, 4, "HEC centre support"),
    VerticalBand("L4_transition_2a", -215.0, -170.0, 4, "coarsening upward"),
    VerticalBand("L4_transition_2b", -170.0, -135.0, 4, "coarsening upward"),
    VerticalBand("L4_transition_2c", -135.0, -100.0, 4, "coarsening upward"),
    VerticalBand("L4_top_transition", -100.0, 0.0, 4, "larger cells near top boundary"),
)
Z_LEVELS: Tuple[float, ...] = tuple(
    [VERTICAL_BANDS[0].z_min] + [band.z_max for band in VERTICAL_BANDS]
)

# Only true material interfaces and external faces are constrained horizontal
# PLC planes.  The other z levels remain free refinement-point planes.
HORIZONTAL_PLC_LEVELS_M = frozenset({-750.0, -550.0, -530.0, -500.0, 0.0})

BOUNDARY_MARKERS: Dict[str, int] = {
    "top": 1,
    "bottom": 2,
    "north": 3,
    "south": 4,
    "east": 5,
    "west": 6,
}

# =============================================================================
# Well and strainmeters.
# =============================================================================
INJECTION_MATERIAL_ID = 6
STRAINMETER_MATERIAL_IDS = {"AVN2": 7, "AVN87": 8, "AVN31": 9}

INJECTION_RADIUS_M = 0.75
# Keep the internal top cap slightly below z=0 so it does not intersect the
# external top PLC.  The model top itself remains exactly z=0.
INJECTION_TOP_CLEARANCE_M = 0.50
INJECTION_Z_MIN_M = HEC_TOP_Z_M
INJECTION_Z_MAX_M = -INJECTION_TOP_CLEARANCE_M
INJECTION_SURFACE_POINTS_PER_RING = 8
INJECTION_SURFACE_AXIAL_SPACING_M = 0.75

STRAINMETER_RADIUS_M = 0.25
STRAINMETER_SURFACE_SUBDIVISIONS = 1


@dataclass(frozen=True)
class StrainmeterInput:
    sensor_id: str
    purpose: str
    x_m: float
    y_m: float
    z_m: float


STRAINMETER_INPUTS: Tuple[StrainmeterInput, ...] = (
    StrainmeterInput("AVN2", "Shallow coupled strainmeter", 5160.0, 5185.0, -720.0),
    StrainmeterInput("AVN87", "Shallow coupled strainmeter", 5460.0, 5185.0, -720.0),
    StrainmeterInput("AVN31", "Deep strainmeter near HEC", 5350.0, 4720.0, -230.0),
)

# Low feature markers remain separate from boundary markers 1--6.
TARGET_FACET_MARKERS: Dict[str, int] = {
    "injection_borehole": 11,
    "AVN2": 12,
    "AVN87": 13,
    "AVN31": 14,
}
MAX_SAFE_FACET_MARKER = 20

# =============================================================================
# Graded local point layouts.
# =============================================================================


@dataclass(frozen=True)
class TubeShell:
    radius_m: float
    axial_spacing_m: float
    circumferential_points: int
    endpoint_padding_m: float


# Radial gap, tangential edge length, and axial spacing grow together.
# This is an O-grid-like point layout around the complete injection well.
INJECTION_TUBE_SHELLS: Tuple[TubeShell, ...] = (
    # radius, maximum axial spacing, points/ring, end padding
    # The innermost rings have edge lengths comparable to the 0.1-m well
    # diameter.  Spacing then grows by about 1.6--1.9 per shell until it
    # approaches the 20-m central matrix lattice.
    TubeShell(1.00, 0.80, 8, 1.00),
    TubeShell(1.60, 1.00, 8, 1.50),
    TubeShell(2.50, 1.40, 10, 2.00),
    TubeShell(4.00, 2.00, 10, 2.80),
    TubeShell(6.20, 3.20, 12, 4.00),
    TubeShell(9.50, 4.80, 12, 5.50),
    TubeShell(14.0, 7.20, 12, 8.00),
    TubeShell(20.0, 10.5, 12, 11.0),
    TubeShell(28.0, 15.0, 12, 15.0),
)

# Rotated geodesic shells around point-like strainmeters.  The geometric
# progression keeps radial spacing comparable to surface edge length.
STRAINMETER_SHELL_RADII_M: Tuple[float, ...] = (
    0.35,
    0.60,
    1.00,
    1.70,
    2.80,
    4.50,
    7.00,
    10.5,
    15.5,
    22.0,
)
STRAINMETER_REFINEMENT_SUBDIVISIONS = 1

INJECTION_YAW_DEG = 17.0
STRAINMETER_ROTATIONS_DEG: Dict[str, Tuple[float, float, float]] = {
    "AVN2": (31.0, 17.0, 9.0),
    "AVN87": (-23.0, 29.0, 13.0),
    "AVN31": (11.0, -19.0, 27.0),
}

# Near-coincident free points can create nearly zero-volume tetrahedra.
POINT_HASH_BIN_SIZE_M = 0.75
MIN_POINT_SEPARATION_FACTOR = 0.25
DOMAIN_FREE_POINT_CLEARANCE_M = 0.10

# IMPORTANT: do not enable TetGen -q by default for this geometry.  The
# quality-refinement pass acts on the entire 10-km domain and all geological
# interface facets, not only on the well.  In the reported run, -q1.5 expanded
# 85,266 input points to 779,329 nodes and 4,740,032 tetrahedra.
#
# The explicit, graded local point layout is therefore used with constrained
# Delaunay tetrahedralisation only.  This normally keeps the output well below
# the workflow's 900,000-tetrahedron guard.
DEFAULT_TETGEN_FLAGS = "-pnAef"
MAX_RECOMMENDED_INPUT_POINTS = 100_000
MAX_RECOMMENDED_INPUT_FACETS = 100_000

# =============================================================================
# Data structures.
# =============================================================================


@dataclass(frozen=True)
class Facet:
    point_ids: Tuple[int, int, int]
    marker: int


@dataclass(frozen=True)
class Region:
    point: np.ndarray
    attribute: int
    label: str


@dataclass(frozen=True)
class RefinementTarget:
    name: str
    kind: str
    purpose: str
    material_id: int
    tag_shape: str
    center_xyz: Tuple[float, float, float]
    radius_m: float
    z_min_m: Optional[float]
    z_max_m: Optional[float]
    source_sensor_id: Optional[str]

    @property
    def surface_marker(self) -> int:
        return TARGET_FACET_MARKERS[self.name]


class PointRegistry:
    """Unique Part-1 point store using TetGen one-based IDs."""

    def __init__(self) -> None:
        self.points: List[np.ndarray] = []
        self._ids: Dict[Tuple[float, float, float], int] = {}

    @staticmethod
    def _key(point: Sequence[float]) -> Tuple[float, float, float]:
        return tuple(round(float(value), 10) for value in point)

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


class PointSpacingFilter:
    """Reject free points that are too close to existing PLC/free points."""

    def __init__(self, registry: PointRegistry, bin_size_m: float = POINT_HASH_BIN_SIZE_M) -> None:
        if bin_size_m <= 0.0:
            raise ValueError("Point hash bin size must be positive.")
        self.registry = registry
        self.bin_size_m = float(bin_size_m)
        self._bins: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        for point in registry.points:
            self._insert(point)

    def _key(self, point: Sequence[float]) -> Tuple[int, int, int]:
        xyz = np.asarray(point, dtype=float)
        return tuple(int(math.floor(value / self.bin_size_m)) for value in xyz)

    def _insert(self, point: Sequence[float]) -> None:
        self._bins.setdefault(self._key(point), []).append(np.asarray(point, dtype=float).copy())

    def try_add(self, point: Sequence[float], minimum_distance_m: float) -> bool:
        if minimum_distance_m <= 0.0:
            raise ValueError("Minimum distance must be positive.")
        xyz = np.asarray(point, dtype=float)
        key = self._key(xyz)
        search = int(math.ceil(minimum_distance_m / self.bin_size_m))
        limit2 = minimum_distance_m * minimum_distance_m
        for i in range(key[0] - search, key[0] + search + 1):
            for j in range(key[1] - search, key[1] + search + 1):
                for k in range(key[2] - search, key[2] + search + 1):
                    for existing in self._bins.get((i, j, k), ()):
                        delta = xyz - existing
                        if float(np.dot(delta, delta)) < limit2:
                            return False
        before = len(self.registry.points)
        self.registry.add(xyz)
        if len(self.registry.points) == before:
            return False
        self._insert(xyz)
        return True

# =============================================================================
# Basic geometry helpers.
# =============================================================================


def unit(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(values))
    if magnitude <= 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return values / magnitude


def hec_axes() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(HEC_AZIMUTH_EAST_OF_NORTH_DEG)
    length_axis = unit([math.sin(angle), math.cos(angle), 0.0])
    width_axis = unit([math.cos(angle), -math.sin(angle), 0.0])
    return length_axis, width_axis, np.array([0.0, 0.0, 1.0], dtype=float)


def hec_corners() -> np.ndarray:
    length_axis, width_axis, up_axis = hec_axes()
    corners: List[np.ndarray] = []
    for dz in (-0.5 * HEC_THICKNESS_M, 0.5 * HEC_THICKNESS_M):
        for su, sv in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
            corners.append(
                HEC_CENTER
                + su * 0.5 * HEC_LENGTH_M * length_axis
                + sv * 0.5 * HEC_WIDTH_M * width_axis
                + dz * up_axis
            )
    return np.asarray(corners, dtype=float)


def hec_local_uv(x_m: float, y_m: float) -> Tuple[float, float]:
    length_axis, width_axis, _ = hec_axes()
    delta = np.array([x_m - HEC_CENTER[0], y_m - HEC_CENTER[1]], dtype=float)
    return float(np.dot(delta, length_axis[:2])), float(np.dot(delta, width_axis[:2]))


def build_refinement_targets() -> Tuple[RefinementTarget, ...]:
    targets: List[RefinementTarget] = [
        RefinementTarget(
            name="injection_borehole",
            kind="injection_borehole",
            purpose="Injection well from the HEC top toward the model top",
            material_id=INJECTION_MATERIAL_ID,
            tag_shape="vertical_cylinder",
            center_xyz=(5000.0, 5000.0, 0.5 * (INJECTION_Z_MIN_M + INJECTION_Z_MAX_M)),
            radius_m=INJECTION_RADIUS_M,
            z_min_m=INJECTION_Z_MIN_M,
            z_max_m=INJECTION_Z_MAX_M,
            source_sensor_id=None,
        )
    ]
    for item in STRAINMETER_INPUTS:
        targets.append(
            RefinementTarget(
                name=item.sensor_id,
                kind="strainmeter_sensor",
                purpose=item.purpose,
                material_id=STRAINMETER_MATERIAL_IDS[item.sensor_id],
                tag_shape="sphere",
                center_xyz=(item.x_m, item.y_m, item.z_m),
                radius_m=STRAINMETER_RADIUS_M,
                z_min_m=None,
                z_max_m=None,
                source_sensor_id=item.sensor_id,
            )
        )
    return tuple(targets)


REFINEMENT_TARGETS = build_refinement_targets()


def inclusive_values(start: float, stop: float, step: float) -> List[float]:
    count = int(round((stop - start) / step))
    if not math.isclose(start + count * step, stop, abs_tol=1.0e-9):
        raise ValueError(f"[{start}, {stop}] is not divisible by {step}.")
    return [start + index * step for index in range(count + 1)]


def unique_sorted(values: Iterable[float]) -> np.ndarray:
    return np.asarray(sorted({round(float(value), 9) for value in values}), dtype=float)


def make_x_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4600.0, 4700.0]
    values += inclusive_values(X_FINE_START, X_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5300.0, 5400.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    if axis[0] != 0.0 or axis[-1] != 10000.0 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("Invalid x axis.")
    return axis


def make_y_axis() -> np.ndarray:
    values: List[float] = []
    values += inclusive_values(0.0, 4000.0, MATRIX_COARSE_STEP_M)
    values += [4200.0, 4400.0, 4500.0, 4660.0]
    values += inclusive_values(Y_FINE_START, Y_FINE_END, ROTATED_TAG_LATTICE_STEP_M)
    values += [5340.0, 5500.0, 5600.0, 5800.0, 6000.0]
    values += inclusive_values(6500.0, 10000.0, MATRIX_COARSE_STEP_M)
    axis = unique_sorted(values)
    if axis[0] != 0.0 or axis[-1] != 10000.0 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("Invalid y axis.")
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
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        HEC_CENTER[0] + cosine * dx - sine * dy,
        HEC_CENTER[1] + sine * dx + cosine * dy,
    )


def add_oriented_triangle(
    registry: PointRegistry,
    facets: List[Facet],
    ids: Sequence[int],
    normal: Sequence[float],
    marker: int,
) -> None:
    values = list(ids)
    a, b, c = (registry.xyz(point_id) for point_id in values)
    if float(np.dot(np.cross(b - a, c - a), np.asarray(normal, dtype=float))) < 0.0:
        values[1], values[2] = values[2], values[1]
    facets.append(Facet(tuple(values), marker))


def rotation_matrix_xyz(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    ry = np.array(
        [[math.cos(pitch), 0.0, math.sin(pitch)],
         [0.0, 1.0, 0.0],
         [-math.sin(pitch), 0.0, math.cos(pitch)]],
        dtype=float,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, math.cos(roll), -math.sin(roll)],
         [0.0, math.sin(roll), math.cos(roll)]],
        dtype=float,
    )
    return rz @ ry @ rx

# =============================================================================
# Validation.
# =============================================================================


def validate_configuration() -> None:
    tolerance = 1.0e-9
    if not np.allclose(DOMAIN_MIN, [0.0, 0.0, -750.0]):
        raise ValueError("Domain bottom must be z=-750 m.")
    if not np.allclose(DOMAIN_MAX, [10000.0, 10000.0, 0.0]):
        raise ValueError("Domain top must be z=0 m.")

    previous = DOMAIN_MIN[2]
    for band in VERTICAL_BANDS:
        if not math.isclose(band.z_min, previous, abs_tol=tolerance):
            raise ValueError(f"Vertical-band gap before {band.label}.")
        layer = next(layer for layer in LAYERS if layer.number == band.geological_layer)
        if band.z_min < layer.z_min - tolerance or band.z_max > layer.z_max + tolerance:
            raise ValueError(f"{band.label} lies outside geological layer {layer.number}.")
        previous = band.z_max
    if not math.isclose(previous, DOMAIN_MAX[2], abs_tol=tolerance):
        raise ValueError("Vertical bands must terminate at z=0.")

    if not (DOMAIN_MIN[2] < INJECTION_Z_MIN_M < INJECTION_Z_MAX_M < DOMAIN_MAX[2]):
        raise ValueError("Injection cylinder must be strictly inside the z boundaries.")
    if INJECTION_RADIUS_M <= 0.0 or STRAINMETER_RADIUS_M <= 0.0:
        raise ValueError("Target radii must be positive.")

    prior_radius = INJECTION_RADIUS_M
    prior_spacing = 0.0
    for shell in INJECTION_TUBE_SHELLS:
        if shell.radius_m <= prior_radius:
            raise ValueError("Injection shell radii must increase.")
        if shell.axial_spacing_m <= prior_spacing:
            raise ValueError("Injection shell axial spacings must increase.")
        if shell.circumferential_points < 8 or shell.circumferential_points % 2:
            raise ValueError("Tube-shell circumferential point counts must be even and at least 8.")
        prior_radius = shell.radius_m
        prior_spacing = shell.axial_spacing_m

    prior_radius = STRAINMETER_RADIUS_M
    for radius in STRAINMETER_SHELL_RADII_M:
        if radius <= prior_radius:
            raise ValueError("Strainmeter shell radii must increase.")
        prior_radius = radius

    markers = [target.surface_marker for target in REFINEMENT_TARGETS]
    if len(markers) != len(set(markers)):
        raise ValueError("Target facet markers must be unique.")
    if min(markers) <= max(BOUNDARY_MARKERS.values()):
        raise ValueError("Target markers overlap external boundary markers.")
    if max(markers) > MAX_SAFE_FACET_MARKER:
        raise ValueError("Target marker exceeds the configured Voronoi-safe range.")

    for target in REFINEMENT_TARGETS:
        center = np.asarray(target.center_xyz, dtype=float)
        if np.any(center <= DOMAIN_MIN) or np.any(center >= DOMAIN_MAX):
            raise ValueError(f"Target center outside domain: {target.name}")
        if target.tag_shape == "sphere":
            if np.any(center - target.radius_m <= DOMAIN_MIN) or np.any(center + target.radius_m >= DOMAIN_MAX):
                raise ValueError(f"Strainmeter PLC crosses external boundary: {target.name}")

# =============================================================================
# Matrix PLC.
# =============================================================================


def matrix_point_is_excluded(x_m: float, y_m: float, z_m: float) -> bool:
    """Remove coarse/free matrix points from local refinement envelopes.

    Constrained external/material-interface planes are never altered.  At the
    remaining grading levels, coarse points inside a target's outer refinement
    envelope would compete with the body-fitted shells, create centerline
    points inside the well, and reconnect the tiny target directly to the
    20--500 m matrix.  Those points are therefore omitted.
    """
    if float(z_m) in HORIZONTAL_PLC_LEVELS_M:
        return False

    injection = REFINEMENT_TARGETS[0]
    assert injection.z_min_m is not None and injection.z_max_m is not None
    outer_tube = INJECTION_TUBE_SHELLS[-1]
    if (
        injection.z_min_m - outer_tube.endpoint_padding_m
        <= z_m
        <= injection.z_max_m + outer_tube.endpoint_padding_m
    ):
        radial = math.hypot(x_m - injection.center_xyz[0], y_m - injection.center_xyz[1])
        if radial < 1.08 * outer_tube.radius_m:
            return True

    outer_sensor_radius = STRAINMETER_SHELL_RADII_M[-1]
    point = np.array([x_m, y_m, z_m], dtype=float)
    for target in REFINEMENT_TARGETS[1:]:
        if float(np.linalg.norm(point - np.asarray(target.center_xyz, dtype=float))) < 1.08 * outer_sensor_radius:
            return True
    return False


def build_matrix_surface_plc(
    registry: PointRegistry,
    facets: List[Facet],
) -> Tuple[np.ndarray, np.ndarray]:
    x_values, y_values = make_x_axis(), make_y_axis()
    nx, ny = len(x_values) - 1, len(y_values) - 1
    nodes: Dict[Tuple[int, int, int], int] = {}

    for k, z_value in enumerate(Z_LEVELS):
        for i, base_x in enumerate(x_values):
            for j, base_y in enumerate(y_values):
                x_value, y_value = rotate_matrix_xy(float(base_x), float(base_y))
                if matrix_point_is_excluded(x_value, y_value, float(z_value)):
                    continue
                nodes[(i, j, k)] = registry.add((x_value, y_value, z_value))

    min_area = math.inf
    for i in range(nx):
        for j in range(ny):
            p00 = registry.xyz(nodes[(i, j, 0)])[:2]
            p10 = registry.xyz(nodes[(i + 1, j, 0)])[:2]
            p11 = registry.xyz(nodes[(i + 1, j + 1, 0)])[:2]
            p01 = registry.xyz(nodes[(i, j + 1, 0)])[:2]
            area_1 = float((p10[0]-p00[0])*(p11[1]-p00[1]) - (p10[1]-p00[1])*(p11[0]-p00[0]))
            area_2 = float((p11[0]-p00[0])*(p01[1]-p00[1]) - (p11[1]-p00[1])*(p01[0]-p00[0]))
            min_area = min(min_area, area_1, area_2)
            if area_1 <= 1.0e-8 or area_2 <= 1.0e-8:
                raise RuntimeError(f"Rotated plan-view grid folded at ({i}, {j}).")
    print(f"    minimum plan-view triangle area : {min_area:.6g} m^2")

    for k, z_value in enumerate(Z_LEVELS):
        if float(z_value) not in HORIZONTAL_PLC_LEVELS_M:
            continue
        if math.isclose(z_value, DOMAIN_MIN[2]):
            marker, normal = BOUNDARY_MARKERS["bottom"], (0.0, 0.0, -1.0)
        elif math.isclose(z_value, DOMAIN_MAX[2]):
            marker, normal = BOUNDARY_MARKERS["top"], (0.0, 0.0, 1.0)
        else:
            marker, normal = 0, (0.0, 0.0, 1.0)
        for i in range(nx):
            for j in range(ny):
                p00 = nodes[(i, j, k)]
                p10 = nodes[(i + 1, j, k)]
                p11 = nodes[(i + 1, j + 1, k)]
                p01 = nodes[(i, j + 1, k)]
                # Alternate diagonals to avoid a domain-wide directional bias.
                if (i + j + k) % 2 == 0:
                    add_oriented_triangle(registry, facets, (p00, p10, p11), normal, marker)
                    add_oriented_triangle(registry, facets, (p00, p11, p01), normal, marker)
                else:
                    add_oriented_triangle(registry, facets, (p00, p10, p01), normal, marker)
                    add_oriented_triangle(registry, facets, (p10, p11, p01), normal, marker)

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

# =============================================================================
# Injection-well PLC and tube-shell refinement.
# =============================================================================


def axis_values(start: float, stop: float, maximum_spacing: float) -> np.ndarray:
    if stop <= start:
        raise ValueError("Axis stop must exceed start.")
    if maximum_spacing <= 0.0:
        raise ValueError("Axis spacing must be positive.")
    count = max(1, int(math.ceil((stop - start) / maximum_spacing)))
    return np.linspace(start, stop, count + 1)


def rounded_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


def ring_ids(
    registry: PointRegistry,
    center_xy: Sequence[float],
    z_value: float,
    radius_m: float,
    point_count: int,
    phase_rad: float,
) -> List[int]:
    return [
        registry.add(
            (
                float(center_xy[0]) + radius_m * math.cos(phase_rad + 2.0 * math.pi * index / point_count),
                float(center_xy[1]) + radius_m * math.sin(phase_rad + 2.0 * math.pi * index / point_count),
                float(z_value),
            )
        )
        for index in range(point_count)
    ]


def add_vertical_cylinder_plc(
    registry: PointRegistry,
    facets: List[Facet],
    target: RefinementTarget,
) -> Dict[str, int]:
    if target.tag_shape != "vertical_cylinder":
        raise ValueError("Target is not a vertical cylinder.")
    assert target.z_min_m is not None and target.z_max_m is not None

    before_points, before_facets = len(registry.points), len(facets)
    z_values = axis_values(target.z_min_m, target.z_max_m, INJECTION_SURFACE_AXIAL_SPACING_M)
    rings: List[List[int]] = []
    phase0 = math.radians(INJECTION_YAW_DEG)
    half_step = math.pi / INJECTION_SURFACE_POINTS_PER_RING

    for ring_index, z_value in enumerate(z_values):
        # Alternating half-step offsets remove long vertical surface strips.
        phase = phase0 + (half_step if ring_index % 2 else 0.0)
        rings.append(
            ring_ids(
                registry,
                target.center_xyz[:2],
                float(z_value),
                target.radius_m,
                INJECTION_SURFACE_POINTS_PER_RING,
                phase,
            )
        )

    for ring_index, (lower, upper) in enumerate(zip(rings[:-1], rings[1:])):
        for index in range(INJECTION_SURFACE_POINTS_PER_RING):
            nxt = (index + 1) % INJECTION_SURFACE_POINTS_PER_RING
            p0, p1 = lower[index], lower[nxt]
            p2, p3 = upper[nxt], upper[index]
            midpoint = 0.25 * (registry.xyz(p0) + registry.xyz(p1) + registry.xyz(p2) + registry.xyz(p3))
            normal = (midpoint[0] - target.center_xyz[0], midpoint[1] - target.center_xyz[1], 0.0)
            # Alternate panel diagonal direction between ring intervals.
            if (ring_index + index) % 2 == 0:
                add_oriented_triangle(registry, facets, (p0, p1, p2), normal, target.surface_marker)
                add_oriented_triangle(registry, facets, (p0, p2, p3), normal, target.surface_marker)
            else:
                add_oriented_triangle(registry, facets, (p0, p1, p3), normal, target.surface_marker)
                add_oriented_triangle(registry, facets, (p1, p2, p3), normal, target.surface_marker)

    # Cap centers are boundary vertices only; no centerline chain is created.
    bottom_center = registry.add((target.center_xyz[0], target.center_xyz[1], target.z_min_m))
    top_center = registry.add((target.center_xyz[0], target.center_xyz[1], target.z_max_m))
    for index in range(INJECTION_SURFACE_POINTS_PER_RING):
        nxt = (index + 1) % INJECTION_SURFACE_POINTS_PER_RING
        add_oriented_triangle(
            registry,
            facets,
            (bottom_center, rings[0][index], rings[0][nxt]),
            (0.0, 0.0, -1.0),
            target.surface_marker,
        )
        add_oriented_triangle(
            registry,
            facets,
            (top_center, rings[-1][index], rings[-1][nxt]),
            (0.0, 0.0, 1.0),
            target.surface_marker,
        )

    return {
        "surface_points": len(registry.points) - before_points,
        "surface_facets": len(facets) - before_facets,
        "surface_rings": len(rings),
    }


def inside_domain_for_free_point(point: np.ndarray, local_spacing_m: float) -> bool:
    clearance = max(DOMAIN_FREE_POINT_CLEARANCE_M, 0.01 * local_spacing_m)
    return bool(np.all(point > DOMAIN_MIN + clearance) and np.all(point < DOMAIN_MAX - clearance))


def add_injection_tube_shell_points(
    registry: PointRegistry,
    spacing_filter: PointSpacingFilter,
    target: RefinementTarget,
) -> Dict[str, int]:
    assert target.z_min_m is not None and target.z_max_m is not None
    before = len(registry.points)
    attempted = accepted = rejected_domain = rejected_spacing = 0
    previous_radius = target.radius_m
    golden_angle = math.radians(137.50776405003785)

    for shell_index, shell in enumerate(INJECTION_TUBE_SHELLS):
        circumference_count = shell.circumferential_points
        z_start = max(DOMAIN_MIN[2] + DOMAIN_FREE_POINT_CLEARANCE_M, target.z_min_m - shell.endpoint_padding_m)
        z_stop = min(DOMAIN_MAX[2] - DOMAIN_FREE_POINT_CLEARANCE_M, target.z_max_m + shell.endpoint_padding_m)
        z_values = axis_values(z_start, z_stop, shell.axial_spacing_m)
        radial_gap = shell.radius_m - previous_radius
        tangential_edge = 2.0 * math.pi * shell.radius_m / circumference_count
        local_scale = min(radial_gap, shell.axial_spacing_m, tangential_edge)
        minimum_distance = MIN_POINT_SEPARATION_FACTOR * local_scale
        shell_phase = math.radians(INJECTION_YAW_DEG) + shell_index * golden_angle

        for ring_index, z_value in enumerate(z_values):
            # Half-cell helical stagger prevents radial spokes and vertical planes.
            phase = shell_phase + (ring_index % 2) * math.pi / circumference_count
            phase += ring_index * 0.17 * math.pi / circumference_count
            for angular_index in range(circumference_count):
                attempted += 1
                angle = phase + 2.0 * math.pi * angular_index / circumference_count
                point = np.array(
                    [
                        target.center_xyz[0] + shell.radius_m * math.cos(angle),
                        target.center_xyz[1] + shell.radius_m * math.sin(angle),
                        z_value,
                    ],
                    dtype=float,
                )
                if not inside_domain_for_free_point(point, local_scale):
                    rejected_domain += 1
                    continue
                if spacing_filter.try_add(point, minimum_distance):
                    accepted += 1
                else:
                    rejected_spacing += 1
        previous_radius = shell.radius_m

    return {
        "refinement_points": len(registry.points) - before,
        "attempted": attempted,
        "accepted": accepted,
        "rejected_domain": rejected_domain,
        "rejected_spacing": rejected_spacing,
    }

# =============================================================================
# Icosphere target surfaces and geodesic refinement shells.
# =============================================================================


def base_icosahedron() -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return vertices, faces


def icosphere_unit(subdivisions: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    if subdivisions < 0:
        raise ValueError("Icosphere subdivisions cannot be negative.")
    vertices_array, faces = base_icosahedron()
    vertices: List[np.ndarray] = [point.copy() for point in vertices_array]

    for _ in range(subdivisions):
        midpoint_cache: Dict[Tuple[int, int], int] = {}

        def midpoint_id(first: int, second: int) -> int:
            key = (first, second) if first < second else (second, first)
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            midpoint = unit(vertices[first] + vertices[second])
            vertices.append(midpoint)
            point_id = len(vertices) - 1
            midpoint_cache[key] = point_id
            return point_id

        refined: List[Tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = midpoint_id(a, b)
            bc = midpoint_id(b, c)
            ca = midpoint_id(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined

    return np.asarray(vertices, dtype=float), faces


def add_icosphere_plc(
    registry: PointRegistry,
    facets: List[Facet],
    target: RefinementTarget,
) -> Dict[str, int]:
    if target.tag_shape != "sphere":
        raise ValueError("Target is not a sphere.")
    before_points, before_facets = len(registry.points), len(facets)
    vertices, faces = icosphere_unit(STRAINMETER_SURFACE_SUBDIVISIONS)
    rotation = rotation_matrix_xyz(*STRAINMETER_ROTATIONS_DEG[target.name])
    center = np.asarray(target.center_xyz, dtype=float)
    point_ids = [
        registry.add(center + target.radius_m * (rotation @ vertex))
        for vertex in vertices
    ]

    for a, b, c in faces:
        ids = (point_ids[a], point_ids[b], point_ids[c])
        centroid = (registry.xyz(ids[0]) + registry.xyz(ids[1]) + registry.xyz(ids[2])) / 3.0
        add_oriented_triangle(registry, facets, ids, centroid - center, target.surface_marker)

    return {
        "surface_points": len(registry.points) - before_points,
        "surface_facets": len(facets) - before_facets,
        "surface_rings": 0,
    }


def add_strainmeter_geodesic_shell_points(
    registry: PointRegistry,
    spacing_filter: PointSpacingFilter,
    target: RefinementTarget,
) -> Dict[str, int]:
    before = len(registry.points)
    attempted = accepted = rejected_domain = rejected_spacing = 0
    vertices, _ = icosphere_unit(STRAINMETER_REFINEMENT_SUBDIVISIONS)
    base_rotation = STRAINMETER_ROTATIONS_DEG[target.name]
    center = np.asarray(target.center_xyz, dtype=float)
    previous_radius = target.radius_m

    for shell_index, radius in enumerate(STRAINMETER_SHELL_RADII_M):
        # Rotate every shell differently so vertices do not form radial chains.
        rotation = rotation_matrix_xyz(
            base_rotation[0] + 37.0 * shell_index,
            base_rotation[1] + 23.0 * shell_index,
            base_rotation[2] + 19.0 * shell_index,
        )
        radial_gap = radius - previous_radius
        # Subdivision-1 icosphere mean edge is about 0.58*radius.
        tangential_scale = 0.55 * radius
        local_scale = min(radial_gap, tangential_scale)
        minimum_distance = MIN_POINT_SEPARATION_FACTOR * local_scale

        for vertex in vertices:
            attempted += 1
            point = center + radius * (rotation @ vertex)
            if not inside_domain_for_free_point(point, local_scale):
                rejected_domain += 1
                continue
            if spacing_filter.try_add(point, minimum_distance):
                accepted += 1
            else:
                rejected_spacing += 1
        previous_radius = radius

    return {
        "refinement_points": len(registry.points) - before,
        "attempted": attempted,
        "accepted": accepted,
        "rejected_domain": rejected_domain,
        "rejected_spacing": rejected_spacing,
    }


def add_target_plcs_and_refinement(
    registry: PointRegistry,
    facets: List[Facet],
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}

    # First add every constrained material surface.
    for target in REFINEMENT_TARGETS:
        if target.tag_shape == "vertical_cylinder":
            stats[target.name] = add_vertical_cylinder_plc(registry, facets, target)
        elif target.tag_shape == "sphere":
            stats[target.name] = add_icosphere_plc(registry, facets, target)
        else:
            raise ValueError(f"Unsupported target shape: {target.tag_shape}")

    # Then add free refinement shells while respecting all constrained points.
    spacing_filter = PointSpacingFilter(registry)
    for target in REFINEMENT_TARGETS:
        if target.tag_shape == "vertical_cylinder":
            refinement = add_injection_tube_shell_points(registry, spacing_filter, target)
        else:
            refinement = add_strainmeter_geodesic_shell_points(registry, spacing_filter, target)
        stats[target.name].update(refinement)
        stats[target.name]["points_added"] = (
            stats[target.name]["surface_points"] + stats[target.name]["refinement_points"]
        )

    return stats

# =============================================================================
# PLC topology checks.
# =============================================================================


def validate_target_surface_topology(facets: Sequence[Facet]) -> Dict[int, Dict[str, int]]:
    results: Dict[int, Dict[str, int]] = {}
    for marker in TARGET_FACET_MARKERS.values():
        selected = [facet for facet in facets if facet.marker == marker]
        edge_counts: Counter[Tuple[int, int]] = Counter()
        for facet in selected:
            a, b, c = facet.point_ids
            for first, second in ((a, b), (b, c), (c, a)):
                edge_counts[(min(first, second), max(first, second))] += 1
        bad_edges = sum(1 for count in edge_counts.values() if count != 2)
        if bad_edges:
            raise RuntimeError(
                f"Target surface marker {marker} is not watertight: {bad_edges} edges do not have multiplicity 2."
            )
        results[marker] = {
            "facets": len(selected),
            "edges": len(edge_counts),
            "bad_edges": bad_edges,
        }
    return results

# =============================================================================
# Regions and output.
# =============================================================================


def target_region_seed(target: RefinementTarget) -> Region:
    center = np.asarray(target.center_xyz, dtype=float)
    if target.tag_shape == "sphere":
        point = center + np.array([0.15 * target.radius_m, 0.0, 0.0], dtype=float)
    else:
        point = center.copy()
    return Region(point=point, attribute=target.material_id, label=target.name)


def matrix_regions() -> Tuple[Region, ...]:
    # Use a location far from all local targets.
    x, y = 1000.0, 1000.0
    return (
        Region(np.array([x, y, -650.0]), 4, "underburden"),
        Region(np.array([x, y, -540.0]), 2, "bartlesville_sand"),
        Region(np.array([x, y, -515.0]), 3, "basal_layer"),
        Region(np.array([x, y, -250.0]), 1, "overburden"),
    )


def write_poly(
    path: Path,
    registry: PointRegistry,
    facets: Sequence[Facet],
    regions: Sequence[Region],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Bartlesville PLC: z=0 top, z=-750 bottom; graded tube/geodesic refinement\n\n")
        handle.write("# Part 1 - node list\n")
        handle.write(f"{len(registry.points)} 3 0 0\n")
        for point_id, xyz in enumerate(registry.points, start=1):
            handle.write(f"{point_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")

        handle.write("\n# Part 2 - facet list\n")
        handle.write(f"{len(facets)} 1\n")
        for facet in facets:
            handle.write(f"1 0 {facet.marker}\n")
            handle.write(f"3 {facet.point_ids[0]} {facet.point_ids[1]} {facet.point_ids[2]}\n")

        handle.write("\n# Part 3 - holes\n0\n")
        handle.write("\n# Part 4 - regions\n")
        handle.write(f"{len(regions)}\n")
        for index, region in enumerate(regions, start=1):
            x_value, y_value, z_value = region.point
            handle.write(
                f"{index} {x_value:.10f} {y_value:.10f} {z_value:.10f} "
                f"{region.attribute}\n"
            )


def validate_geometry_sidecar_schema(geometry: Dict[str, object]) -> None:
    """Fail early when the JSON schema is incompatible with downstream scripts."""
    hec = geometry.get("hec")
    if not isinstance(hec, dict):
        raise RuntimeError("Geometry sidecar is missing the 'hec' object.")

    required_hec_keys = (
        "center",
        "center_xyz_m",
        "length_m",
        "width_m",
        "thickness_m",
        "bottom_z_m",
        "top_z_m",
        "axes",
    )
    missing = [key for key in required_hec_keys if key not in hec]
    if missing:
        raise RuntimeError(
            "Geometry sidecar HEC schema is incomplete; missing: "
            + ", ".join(missing)
        )

    center = np.asarray(hec["center"], dtype=float)
    center_alias = np.asarray(hec["center_xyz_m"], dtype=float)
    if center.shape != (3,) or center_alias.shape != (3,):
        raise RuntimeError("HEC center fields must each contain three coordinates.")
    if not np.allclose(center, center_alias, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("HEC 'center' and 'center_xyz_m' fields disagree.")

    axes = hec["axes"]
    if not isinstance(axes, dict):
        raise RuntimeError("Geometry sidecar HEC axes must be an object.")
    missing_axes = [
        key for key in ("length", "width", "normal_up", "up")
        if key not in axes
    ]
    if missing_axes:
        raise RuntimeError(
            "Geometry sidecar HEC axes are incomplete; missing: "
            + ", ".join(missing_axes)
        )


def write_sidecars(
    mesh_prefix: str,
    registry: PointRegistry,
    facets: Sequence[Facet],
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_stats: Dict[str, Dict[str, int]],
    topology: Dict[int, Dict[str, int]],
) -> None:
    length_axis, width_axis, up_axis = hec_axes()

    target_records = []
    for target in REFINEMENT_TARGETS:
        record: Dict[str, object] = {
            "name": target.name,
            "kind": target.kind,
            "purpose": target.purpose,
            "material_id": target.material_id,
            "surface_marker": target.surface_marker,
            "tag_shape": target.tag_shape,
            "center_xyz_m": list(target.center_xyz),
            "tag_radius_m": target.radius_m,
            "source_sensor_id": target.source_sensor_id,
            "local_point_stats": target_stats[target.name],
            "surface_topology": topology[target.surface_marker],
        }
        if target.tag_shape == "vertical_cylinder":
            record["tag_z_min_m"] = target.z_min_m
            record["tag_z_max_m"] = target.z_max_m
            record["tube_shells"] = [asdict(shell) for shell in INJECTION_TUBE_SHELLS]
        else:
            record["geodesic_shell_radii_m"] = list(STRAINMETER_SHELL_RADII_M)
        target_records.append(record)

    geometry = {
        "coordinate_convention": {
            "description": "elevation relative to model top",
            "top_z_m": 0.0,
            "bottom_z_m": -750.0,
            "conversion_from_previous_geometry": "z_new = z_old - 750",
        },
        "domain": {
            "min": DOMAIN_MIN.tolist(),
            "max": DOMAIN_MAX.tolist(),
            "size": DOMAIN_SIZE.tolist(),
        },
        "layers": [asdict(layer) for layer in LAYERS],
        "vertical_bands": [asdict(band) for band in VERTICAL_BANDS],
        "z_levels_m": list(Z_LEVELS),
        "matrix": {
            "x_axis_m": x_values.tolist(),
            "y_axis_m": y_values.tolist(),
            "rotated_hec_lattice_step_m": ROTATED_TAG_LATTICE_STEP_M,
        },
        "meshing": {
            "strategy": "closed target PLCs plus staggered tube shells and rotated geodesic shells",
            "tetgen_default_flags": DEFAULT_TETGEN_FLAGS,
            "point_count": len(registry.points),
            "facet_count": len(facets),
            "holes": 0,
            "regions": 4 + len(REFINEMENT_TARGETS),
        },
        "hec": {
            "name": HEC_NAME,
            "material_id": HEC_MATERIAL_ID,
            "host_material_id": HEC_HOST_MATERIAL_ID,
            "representation": "tag_only",

            # Backward-compatible field used by
            # layers4_get_material_boundary_tags.py.
            "center": HEC_CENTER.tolist(),

            # Explicit-unit alias retained for newer readers.
            "center_xyz_m": HEC_CENTER.tolist(),
            "length_m": HEC_LENGTH_M,
            "width_m": HEC_WIDTH_M,
            "thickness_m": HEC_THICKNESS_M,
            "bottom_z_m": HEC_BOTTOM_Z_M,
            "top_z_m": HEC_TOP_Z_M,
            "azimuth_deg_east_of_north": HEC_AZIMUTH_EAST_OF_NORTH_DEG,
            "dip_deg": 0.0,
            "axes": {
                "length": length_axis.tolist(),
                "width": width_axis.tolist(),

                # Both names are written because the existing tagging script
                # uses normal_up, whereas newer code may use up.
                "normal_up": up_axis.tolist(),
                "up": up_axis.tolist(),
            },
            "tagging": {
                "method": "oriented_rectangular_prism_in_local_coordinates",
                "center_field": "center",
                "z_range_m": [HEC_BOTTOM_Z_M, HEC_TOP_Z_M],
            },
        },
        "refinement_targets": target_records,
        "boundary_markers": BOUNDARY_MARKERS,
    }
    validate_geometry_sidecar_schema(geometry)
    Path(f"{mesh_prefix}_geometry.json").write_text(
        json.dumps(geometry, indent=2) + "\n", encoding="utf-8"
    )

    with Path(f"{mesh_prefix}_hec_tag_geometry.xyz").open("w", encoding="utf-8") as handle:
        handle.write("# id x_m y_m z_m\n")
        for index, corner in enumerate(hec_corners(), start=1):
            handle.write(f"{index} {corner[0]:.10f} {corner[1]:.10f} {corner[2]:.10f}\n")

    with Path(f"{mesh_prefix}_refinement_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "kind", "material_id", "surface_marker", "tag_shape",
            "center_x_m", "center_y_m", "center_z_m", "radius_m",
            "z_min_m", "z_max_m", "surface_points", "surface_facets",
            "refinement_points", "points_added",
        ])
        for target in REFINEMENT_TARGETS:
            stats = target_stats[target.name]
            writer.writerow([
                target.name, target.kind, target.material_id, target.surface_marker,
                target.tag_shape, f"{target.center_xyz[0]:.10f}",
                f"{target.center_xyz[1]:.10f}", f"{target.center_xyz[2]:.10f}",
                f"{target.radius_m:.10f}",
                "" if target.z_min_m is None else f"{target.z_min_m:.10f}",
                "" if target.z_max_m is None else f"{target.z_max_m:.10f}",
                stats["surface_points"], stats["surface_facets"],
                stats["refinement_points"], stats["points_added"],
            ])

    with Path(f"{mesh_prefix}_tube_refinement_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "shell_index", "radius_m", "axial_spacing_m",
            "circumferential_points", "endpoint_padding_m",
        ])
        for index, shell in enumerate(INJECTION_TUBE_SHELLS):
            writer.writerow([
                index, f"{shell.radius_m:.8f}", f"{shell.axial_spacing_m:.8f}",
                shell.circumferential_points, f"{shell.endpoint_padding_m:.8f}",
            ])

    with Path(f"{mesh_prefix}_strainmeters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sensor_id", "purpose", "x_m", "y_m", "z_m", "material_id",
            "radius_m", "hec_local_u_m", "hec_local_v_m",
        ])
        for item in STRAINMETER_INPUTS:
            u_value, v_value = hec_local_uv(item.x_m, item.y_m)
            writer.writerow([
                item.sensor_id, item.purpose, f"{item.x_m:.10f}",
                f"{item.y_m:.10f}", f"{item.z_m:.10f}",
                STRAINMETER_MATERIAL_IDS[item.sensor_id], f"{STRAINMETER_RADIUS_M:.10f}",
                f"{u_value:.10f}", f"{v_value:.10f}",
            ])

    with Path(f"{mesh_prefix}_vertical_grading.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "z_min_m", "z_max_m", "thickness_m", "geological_layer", "note"],
        )
        writer.writeheader()
        for band in VERTICAL_BANDS:
            writer.writerow({
                "label": band.label,
                "z_min_m": f"{band.z_min:.6f}",
                "z_max_m": f"{band.z_max:.6f}",
                "thickness_m": f"{band.z_max - band.z_min:.6f}",
                "geological_layer": band.geological_layer,
                "note": band.note,
            })

# =============================================================================
# Build, TetGen run, and TetGen-output validation.
# =============================================================================


def build_geometry(mesh_prefix: str) -> Tuple[Path, Dict[str, int]]:
    validate_configuration()
    registry = PointRegistry()
    facets: List[Facet] = []

    x_values, y_values = build_matrix_surface_plc(registry, facets)
    target_stats = add_target_plcs_and_refinement(registry, facets)
    topology = validate_target_surface_topology(facets)

    if len(registry.points) > MAX_RECOMMENDED_INPUT_POINTS:
        raise RuntimeError(
            f"PLC contains {len(registry.points):,} input points, above the configured "
            f"preflight limit of {MAX_RECOMMENDED_INPUT_POINTS:,}. Coarsen the local shell profile."
        )
    if len(facets) > MAX_RECOMMENDED_INPUT_FACETS:
        raise RuntimeError(
            f"PLC contains {len(facets):,} facets, above the configured preflight limit "
            f"of {MAX_RECOMMENDED_INPUT_FACETS:,}."
        )

    regions = matrix_regions() + tuple(target_region_seed(target) for target in REFINEMENT_TARGETS)
    poly_path = Path(f"{mesh_prefix}.poly")
    write_poly(poly_path, registry, facets, regions)
    write_sidecars(mesh_prefix, registry, facets, x_values, y_values, target_stats, topology)

    counts: Dict[str, int] = {
        "points": len(registry.points),
        "facets": len(facets),
        "holes": 0,
        "regions": len(regions),
        "local_refinement_points": sum(stats["refinement_points"] for stats in target_stats.values()),
    }
    for name, stats in target_stats.items():
        counts[f"{name}_surface_points"] = stats["surface_points"]
        counts[f"{name}_surface_facets"] = stats["surface_facets"]
        counts[f"{name}_refinement_points"] = stats["refinement_points"]
    return poly_path, counts


def parse_tetgen_node(path: Path) -> Dict[int, np.ndarray]:
    nodes: Dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        header_read = False
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if not header_read:
                header_read = True
                continue
            node_id = int(fields[0])
            nodes[node_id] = np.array([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)
    return nodes


def parse_tetgen_ele(path: Path) -> List[Tuple[int, int, int, int]]:
    elements: List[Tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header_read = False
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if not header_read:
                header_read = True
                continue
            elements.append(tuple(int(value) for value in fields[1:5]))
    return elements


def tetra_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return abs(float(np.dot(b - a, np.cross(c - a, d - a)))) / 6.0


def tetra_edge_ratio(points: Sequence[np.ndarray]) -> float:
    edges = []
    for first in range(4):
        for second in range(first + 1, 4):
            edges.append(float(np.linalg.norm(points[first] - points[second])))
    minimum = min(edges)
    return math.inf if minimum <= 0.0 else max(edges) / minimum


def validate_tetgen_output(poly_path: Path) -> Dict[str, float]:
    node_path = poly_path.with_suffix(".1.node")
    ele_path = poly_path.with_suffix(".1.ele")
    if not node_path.exists() or not ele_path.exists():
        raise FileNotFoundError("TetGen did not create the expected .1.node and .1.ele files.")

    nodes = parse_tetgen_node(node_path)
    elements = parse_tetgen_ele(ele_path)
    repeated = zero_or_tiny = extreme_ratio = 0
    min_volume = math.inf
    max_ratio = 0.0

    for element in elements:
        if len(set(element)) != 4:
            repeated += 1
            continue
        points = [nodes[node_id] for node_id in element]
        volume = tetra_volume(*points)
        ratio = tetra_edge_ratio(points)
        min_volume = min(min_volume, volume)
        max_ratio = max(max_ratio, ratio)
        if volume <= 1.0e-16:
            zero_or_tiny += 1
        if ratio > 25.0:
            extreme_ratio += 1

    if repeated or zero_or_tiny:
        raise RuntimeError(
            "TetGen output contains invalid tetrahedra: "
            f"repeated-node={repeated}, near-zero-volume={zero_or_tiny}. "
            "Do not run the Voronoi converter on this mesh."
        )

    return {
        "nodes": float(len(nodes)),
        "tetrahedra": float(len(elements)),
        "minimum_volume_m3": float(min_volume),
        "maximum_edge_ratio": float(max_ratio),
        "edge_ratio_above_25": float(extreme_ratio),
    }


def tetgen_switch_present(flags: str, switch: str) -> bool:
    """Return True when a TetGen switch token contains the requested letter."""
    for token in shlex.split(flags):
        if token.startswith("-") and switch in token[1:]:
            return True
    return False


def run_tetgen(tetgen_exe: str, poly_path: Path, diagnose: bool) -> None:
    flags = os.environ.get("BARTLESVILLE_TETGEN_FLAGS", DEFAULT_TETGEN_FLAGS).strip()
    if not flags:
        raise ValueError("BARTLESVILLE_TETGEN_FLAGS cannot be empty.")
    if tetgen_switch_present(flags, "q") and os.environ.get("BARTLESVILLE_ALLOW_Q", "0") != "1":
        raise ValueError(
            "TetGen -q is disabled for this large multi-scale PLC because it previously "
            "generated 4.74 million tetrahedra. Remove -q from "
            "BARTLESVILLE_TETGEN_FLAGS. Set BARTLESVILLE_ALLOW_Q=1 only for a deliberate test."
        )
    if tetgen_switch_present(flags, "a"):
        raise ValueError(
            "Do not use TetGen -a with this point-controlled profile; global/region volume "
            "refinement can exceed the workflow mesh-size guard."
        )
    if diagnose and not tetgen_switch_present(flags, "d"):
        flags += "d"
    command = [tetgen_exe, *shlex.split(flags), str(poly_path)]

    print("\n--> Running TetGen")
    print(f"    domain z range          : {DOMAIN_MIN[2]:g} to {DOMAIN_MAX[2]:g} m")
    print(f"    injection radius        : {INJECTION_RADIUS_M:g} m")
    print(f"    strainmeter radius      : {STRAINMETER_RADIUS_M:g} m")
    print("    local layout            : lean staggered tube shells + rotated geodesic shells")
    print("    quality refinement      : disabled by default to prevent global Steiner-point growth")
    print("    TetGen flags            :", flags)
    print("CMD:", " ".join(shlex.quote(token) for token in command))
    subprocess.run(command, check=True)

    diagnostics = validate_tetgen_output(poly_path)
    print("\n--> TetGen mesh validation")
    for name, value in diagnostics.items():
        print(f"    {name:27s}: {value:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Bartlesville PLC with z=0 top, z=-750 bottom, and graded tetrahedral local refinement."
    )
    parser.add_argument("mesh_prefix", help="Output mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("tetgen_exe", nargs="?", help="TetGen executable unless --write-only is used")
    parser.add_argument("--write-only", action="store_true", help="Write .poly and sidecars without running TetGen")
    parser.add_argument("--diagnose", action="store_true", help="Append TetGen -d diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix.removesuffix(".poly")
    poly_path, counts = build_geometry(prefix)

    print("\n--> Wrote graded Bartlesville PLC")
    print(f"    poly file               : {poly_path}")
    print(f"    top face z              : {DOMAIN_MAX[2]:g} m")
    print(f"    bottom face z           : {DOMAIN_MIN[2]:g} m")
    print(f"    geometry JSON           : {prefix}_geometry.json")
    print(f"    refinement targets      : {prefix}_refinement_targets.csv")
    print(f"    tube refinement profile : {prefix}_tube_refinement_profile.csv")
    print(f"    strainmeter locations   : {prefix}_strainmeters.csv")
    print(f"    vertical grading        : {prefix}_vertical_grading.csv")
    for name, value in counts.items():
        print(f"    {name:27s}: {value}")
    print(f"    rough no-q tet estimate : {5 * counts['points']:,}--{8 * counts['points']:,}")
    print("    workflow tet guard      : 900,000")

    if args.write_only:
        print("\n--> --write-only selected; TetGen was not run.\n")
        return
    if not args.tetgen_exe:
        raise SystemExit("ERROR: provide <tetgen_exe>, or use --write-only.")
    run_tetgen(args.tetgen_exe, poly_path, args.diagnose)
    print("\n--> TetGen completed successfully.\n")


if __name__ == "__main__":
    main()
