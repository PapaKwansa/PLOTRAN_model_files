#!/usr/bin/env python3
"""Build the North Avant / Bartlesville TetGen PLC with a fast local-refinement profile.

This version uses the supplied physical strainmeter locations:
    AVN2  : (5160, 5185,  30) m   shallow coupled strainmeter
    AVN87 : (5460, 5185,  30) m   shallow coupled strainmeter
    AVN31 : (5350, 4720, 520) m   deep strainmeter near the HEC

Key meshing choice
------------------
The HEC remains tag-only and the injection borehole remains a solid,
material-tagged cylindrical mesh zone.  The strainmeters are represented as
small, material-tagged *sensor pods* centred at their actual 3-D locations;
the supplied data identify sensor locations, not full borehole trajectories.

The earlier polar-shell / concentric-ring point clouds are intentionally not
used.  They can form radial spoke chains and large fan-shaped tetrahedra when
TetGen connects a very fine ring to the surrounding matrix.  Local refinement is generated with compact deterministic, lightly jittered
body-centred-cubic (BCC) point clouds in a small number of graded bands. This
gives a volumetric point field rather than aligned radial rings while keeping
the Part-1 vertex count low enough that the default TetGen run does not enter
an expensive iterative quality-refinement loop.

No borehole or strainmeter is a TetGen hole or an internal closed PLC shell.
All are ordinary Part-1 points.  Materials are assigned after TetGen using the
explicit tag geometry written to the JSON/CSV sidecars.
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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
# Tag-only HEC. x=east and y=north.  The HEC is horizontal and its 580 m axis
# is 5 degrees east (+x) of north (+y).
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
# Matrix grid.  This is the existing working rotated central lattice used for
# the tag-only HEC footprint.
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
# Tag materials.
# -----------------------------------------------------------------------------
INJECTION_MATERIAL_ID = 6
STRAINMETER_MATERIAL_IDS = {
    "AVN2": 7,
    "AVN87": 8,
    "AVN31": 9,
}

# -----------------------------------------------------------------------------
# Injection borehole tag and its local point cloud.
# -----------------------------------------------------------------------------
INJECTION_RADIUS_M = 5.0
HEC_BOTTOM_Z_M = HEC_CENTER[2] - 0.5 * HEC_THICKNESS_M
HEC_TOP_Z_M    = HEC_CENTER[2] + 0.5 * HEC_THICKNESS_M

INJECTION_TAG_Z_MIN_M = HEC_BOTTOM_Z_M
INJECTION_TAG_Z_MAX_M = DOMAIN_MAX[2]

# put the local refinement inside the source interval
INJECTION_LATTICE_Z_MIN_M = 527.75  # or 528.0
INJECTION_LATTICE_Z_MAX_M = 535.25

# -----------------------------------------------------------------------------
# Strainmeter data supplied by the user.  These are sensor locations, so the
# mesh uses compact 3-D sensor pods rather than invented long vertical wells.
# -----------------------------------------------------------------------------
STRAINMETER_TAG_RADIUS_M = 5.0


@dataclass(frozen=True)
class StrainmeterInput:
    sensor_id: str
    purpose: str
    x_m: float
    y_m: float
    z_m: float


STRAINMETER_INPUTS: Tuple[StrainmeterInput, ...] = (
    StrainmeterInput("AVN2", "Shallow coupled strainmeter", 5160.0, 5185.0, 30.0),
    StrainmeterInput("AVN87", "Shallow coupled strainmeter", 5460.0, 5185.0, 30.0),
    StrainmeterInput("AVN31", "Deep strainmeter near HEC", 5350.0, 4720.0, 520.0),
)

# -----------------------------------------------------------------------------
# Graded BCC refinement profiles.  Each pair is (outer_radius_m, bcc_spacing_m)
# and consecutive bands overlap slightly.  The growth ratios are intentionally
# modest; no ring jumps directly from sub-metre / metre cells to the 20--500 m
# matrix lattice.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementBand:
    outer_radius_m: float
    spacing_m: float


# Fast, graded BCC refinement profiles.
#
# The previous profile generated about 98,500 local Part-1 points.  Most of
# those points came from the 2-m injection lattice through a 100-m radius and
# from sub-metre strainmeter clouds through a 100-m radius.  That is far more
# detailed than is needed to preserve a material-6 injection cylinder of
# radius 5 m and material-7--9 sensor pods of radius 2 m.
#
# These compact bands preserve at least two local point spacings across each
# physical tag diameter, then expand with an approximately 1.6--1.8 spacing
# ratio.  The outer bands are deliberately large enough to connect to the
# existing 20-m central matrix lattice without a direct tiny-to-coarse jump,
# but not so large that TetGen is flooded with vertices.
INJECTION_REFINEMENT_BANDS: Tuple[RefinementBand, ...] = (
    RefinementBand(10.0, 3.00),
    RefinementBand(18.0, 5.00),
    RefinementBand(32.0, 8.00),
    RefinementBand(55.0, 13.00),
    RefinementBand(80.0, 21.00),
)

SENSOR_REFINEMENT_BANDS: Tuple[RefinementBand, ...] = (
    RefinementBand(4.0, 1.50),
    RefinementBand(8.0, 3.00),
    RefinementBand(16.0, 6.00),
    RefinementBand(30.0, 10.00),
    RefinementBand(50.0, 17.00),
)

# BCC generation controls.  A moderately larger point-separation threshold
# avoids redundant points in neighbouring overlap bands while preserving the
# non-polar volumetric point cloud that prevents spoke-like tetrahedra.
BAND_OVERLAP_FRACTION = 0.10
BCC_JITTER_FRACTION = 0.025
MIN_POINT_SEPARATION_FRACTION = 0.45
POINT_HASH_BIN_SIZE_M = 8.0
DOMAIN_POINT_CLEARANCE_M = 0.25

# Distinct yaw angles prevent local BCC planes from lining up with the global
# x-y matrix lattice or with one another.
INJECTION_YAW_DEG = 17.0
STRAINMETER_YAW_DEG = {
    "AVN2": 31.0,
    "AVN87": -23.0,
    "AVN31": 11.0,
}

# Keep the default as a constrained Delaunay tetrahedralisation only.
#
# Adding -q makes TetGen enter its iterative "Refining mesh..." phase and
# insert Steiner points until every requested quality limit is met.  With a
# 10-km domain and local point clouds this can become extremely expensive.
# The BCC field, smooth band grading, and point-spacing filter are therefore
# the default anti-sliver mechanism.  Do not enable -q unless a later mesh
# inspection shows a specific quality defect that cannot be fixed geometrically.
DEFAULT_TETGEN_FLAGS = "-pnAef"


@dataclass(frozen=True)
class RefinementTarget:
    """One post-mesh material tag plus the point-cloud envelope supporting it."""

    name: str
    kind: str
    purpose: str
    color_hint: str
    material_id: int
    tag_shape: str  # "vertical_cylinder" or "sphere"
    center_xyz: Tuple[float, float, float]
    tag_radius_m: float
    tag_z_min_m: Optional[float]
    tag_z_max_m: Optional[float]
    lattice_z_min_m: Optional[float]
    lattice_z_max_m: Optional[float]
    refinement_bands: Tuple[RefinementBand, ...]
    yaw_deg: float
    source_sensor_id: Optional[str] = None

    @property
    def outer_radius_m(self) -> float:
        return self.refinement_bands[-1].outer_radius_m


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
    """Reject near-coincident Part-1 points before they create sliver tetrahedra.

    The filter contains all pre-existing PLC vertices as well as previously
    accepted local points.  It is deliberately lightweight and does not impose
    a global spacing; each candidate carries its own local minimum separation.
    """

    def __init__(self, registry: PointRegistry, bin_size_m: float = POINT_HASH_BIN_SIZE_M) -> None:
        if bin_size_m <= 0.0:
            raise ValueError("Point-hash bin size must be positive.")
        self.registry = registry
        self.bin_size_m = float(bin_size_m)
        self._bins: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        for point in registry.points:
            self._insert(point)

    def _key(self, point: Sequence[float]) -> Tuple[int, int, int]:
        xyz = np.asarray(point, dtype=float)
        return tuple(int(math.floor(value / self.bin_size_m)) for value in xyz)

    def _insert(self, point: Sequence[float]) -> None:
        key = self._key(point)
        self._bins.setdefault(key, []).append(np.asarray(point, dtype=float).copy())

    def add_forced(self, point: Sequence[float]) -> int:
        """Add an explicitly required target point, then index it."""
        before = len(self.registry.points)
        point_id = self.registry.add(point)
        if len(self.registry.points) > before:
            self._insert(point)
        return point_id

    def try_add(self, point: Sequence[float], minimum_distance_m: float) -> bool:
        if minimum_distance_m <= 0.0:
            raise ValueError("Minimum point separation must be positive.")
        xyz = np.asarray(point, dtype=float)
        key = self._key(xyz)
        search = int(math.ceil(minimum_distance_m / self.bin_size_m))
        squared_limit = minimum_distance_m * minimum_distance_m
        for i in range(key[0] - search, key[0] + search + 1):
            for j in range(key[1] - search, key[1] + search + 1):
                for k in range(key[2] - search, key[2] + search + 1):
                    for existing in self._bins.get((i, j, k), ()):
                        if float(np.dot(xyz - existing, xyz - existing)) < squared_limit:
                            return False
        before = len(self.registry.points)
        self.registry.add(xyz)
        if len(self.registry.points) > before:
            self._insert(xyz)
        return True


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


def hec_local_uv(x_m: float, y_m: float) -> Tuple[float, float]:
    length_axis, width_axis, _ = hec_axes()
    delta = np.array([x_m - HEC_CENTER[0], y_m - HEC_CENTER[1]], dtype=float)
    return float(np.dot(delta, length_axis[:2])), float(np.dot(delta, width_axis[:2]))


def build_refinement_targets() -> Tuple[RefinementTarget, ...]:
    targets: List[RefinementTarget] = [
        RefinementTarget(
            name="injection_borehole",
            kind="injection_borehole",
            purpose="Injection borehole terminating on the HEC top",
            color_hint="grey",
            material_id=INJECTION_MATERIAL_ID,
            tag_shape="vertical_cylinder",
            center_xyz=(float(HEC_CENTER[0]), float(HEC_CENTER[1]), 0.5 * (INJECTION_TAG_Z_MIN_M + INJECTION_TAG_Z_MAX_M)),
            tag_radius_m=INJECTION_RADIUS_M,
            tag_z_min_m=INJECTION_TAG_Z_MIN_M,
            tag_z_max_m=INJECTION_TAG_Z_MAX_M,
            lattice_z_min_m=INJECTION_LATTICE_Z_MIN_M,
            lattice_z_max_m=INJECTION_LATTICE_Z_MAX_M,
            refinement_bands=INJECTION_REFINEMENT_BANDS,
            yaw_deg=INJECTION_YAW_DEG,
            source_sensor_id=None,
        )
    ]
    for item in STRAINMETER_INPUTS:
        targets.append(
            RefinementTarget(
                name=item.sensor_id,
                kind="strainmeter_sensor",
                purpose=item.purpose,
                color_hint="green",
                material_id=STRAINMETER_MATERIAL_IDS[item.sensor_id],
                tag_shape="sphere",
                center_xyz=(item.x_m, item.y_m, item.z_m),
                tag_radius_m=STRAINMETER_TAG_RADIUS_M,
                tag_z_min_m=None,
                tag_z_max_m=None,
                lattice_z_min_m=None,
                lattice_z_max_m=None,
                refinement_bands=SENSOR_REFINEMENT_BANDS,
                yaw_deg=STRAINMETER_YAW_DEG[item.sensor_id],
                source_sensor_id=item.sensor_id,
            )
        )
    return tuple(targets)


REFINEMENT_TARGETS: Tuple[RefinementTarget, ...] = build_refinement_targets()


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
    cosine, sine = math.cos(angle), math.sin(angle)
    return HEC_CENTER[0] + cosine * dx - sine * dy, HEC_CENTER[1] + sine * dx + cosine * dy


def add_oriented_triangle(
    registry: PointRegistry,
    facets: List[Facet],
    ids: Sequence[int],
    normal: Sequence[float],
    marker: int,
) -> None:
    value = list(ids)
    a, b, c = (registry.xyz(point_id) for point_id in value)
    if float(np.dot(np.cross(b - a, c - a), np.asarray(normal, dtype=float))) < 0.0:
        value[1], value[2] = value[2], value[1]
    facets.append(Facet(tuple(value), marker))


def _distance_cylinder_to_point(
    center_xy: Sequence[float],
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
    point_xyz: Sequence[float],
) -> float:
    """Minimum 3-D distance from a finite vertical cylinder volume to a point."""
    point = np.asarray(point_xyz, dtype=float)
    horizontal = max(0.0, math.hypot(point[0] - center_xy[0], point[1] - center_xy[1]) - radius_m)
    if point[2] < z_min_m:
        vertical = z_min_m - point[2]
    elif point[2] > z_max_m:
        vertical = point[2] - z_max_m
    else:
        vertical = 0.0
    return math.hypot(horizontal, vertical)


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
        if not any(math.isclose(level, z_value, abs_tol=tolerance) for z_value in Z_LEVELS):
            raise ValueError(f"Missing z={level:g} matrix level for the HEC tag.")

    if len(STRAINMETER_INPUTS) != 3:
        raise ValueError("Expected the three supplied strainmeter locations: AVN2, AVN87, AVN31.")
    if len(REFINEMENT_TARGETS) != 4:
        raise ValueError("Expected one injection target plus three strainmeter targets.")

    injection = REFINEMENT_TARGETS[0]
    if injection.tag_shape != "vertical_cylinder":
        raise ValueError("The first refinement target must be the injection cylinder.")
    if not math.isclose(injection.tag_z_min_m or math.nan, HEC_BOTTOM_Z_M, abs_tol=tolerance):
        raise ValueError("Injection borehole must terminate at the HEC base.")
    if not (injection.lattice_z_min_m and injection.lattice_z_max_m):
        raise ValueError("Injection lattice z limits are missing.")
    if not (injection.tag_z_min_m <= injection.lattice_z_min_m < injection.lattice_z_max_m < injection.tag_z_max_m):
        raise ValueError("Injection local point interval must lie strictly inside the tag interval.")

    for target in REFINEMENT_TARGETS:
        center = np.asarray(target.center_xyz, dtype=float)
        if not np.all(center >= DOMAIN_MIN) or not np.all(center <= DOMAIN_MAX):
            raise ValueError(f"Target centre outside domain: {target.name}")
        if target.tag_radius_m <= 0.0:
            raise ValueError(f"Invalid tag radius for {target.name}")
        if not target.refinement_bands:
            raise ValueError(f"No refinement bands for {target.name}")
        previous_radius = 0.0
        previous_spacing = 0.0
        for band in target.refinement_bands:
            if band.outer_radius_m <= previous_radius:
                raise ValueError(f"Non-increasing refinement radius for {target.name}")
            if band.spacing_m <= previous_spacing:
                raise ValueError(f"Non-increasing BCC spacing for {target.name}")
            previous_radius = band.outer_radius_m
            previous_spacing = band.spacing_m
        if target.tag_shape == "sphere":
            if np.any(center - target.tag_radius_m <= DOMAIN_MIN) or np.any(center + target.tag_radius_m >= DOMAIN_MAX):
                raise ValueError(f"Physical sensor tag crosses the external domain boundary: {target.name}")
        elif target.tag_shape == "vertical_cylinder":
            if target.tag_z_min_m is None or target.tag_z_max_m is None:
                raise ValueError(f"Cylinder z limits missing: {target.name}")
            if target.tag_z_min_m < DOMAIN_MIN[2] or target.tag_z_max_m > DOMAIN_MAX[2]:
                raise ValueError(f"Cylinder tag leaves domain: {target.name}")
        else:
            raise ValueError(f"Unsupported tag shape for {target.name}: {target.tag_shape}")

    # Validate that the finite 3-D refinement envelopes do not overlap.  The
    # shallow sensors can have horizontally broad halos because they are far
    # below the injection interval in z; use true 3-D checks rather than an
    # unnecessarily restrictive plan-view-only test.
    for first_index, first in enumerate(REFINEMENT_TARGETS):
        for second in REFINEMENT_TARGETS[first_index + 1:]:
            if first.tag_shape == "sphere" and second.tag_shape == "sphere":
                centre_distance = float(np.linalg.norm(np.asarray(first.center_xyz) - np.asarray(second.center_xyz)))
                gap = centre_distance - first.outer_radius_m - second.outer_radius_m
            elif first.tag_shape == "vertical_cylinder" and second.tag_shape == "sphere":
                assert first.lattice_z_min_m is not None and first.lattice_z_max_m is not None
                gap = _distance_cylinder_to_point(
                    first.center_xyz[:2], first.outer_radius_m,
                    first.lattice_z_min_m, first.lattice_z_max_m,
                    second.center_xyz,
                ) - second.outer_radius_m
            elif first.tag_shape == "sphere" and second.tag_shape == "vertical_cylinder":
                assert second.lattice_z_min_m is not None and second.lattice_z_max_m is not None
                gap = _distance_cylinder_to_point(
                    second.center_xyz[:2], second.outer_radius_m,
                    second.lattice_z_min_m, second.lattice_z_max_m,
                    first.center_xyz,
                ) - first.outer_radius_m
            else:
                raise RuntimeError("Only one cylinder target is expected.")
            guard = 0.5 * min(first.refinement_bands[-1].spacing_m, second.refinement_bands[-1].spacing_m)
            if gap < guard:
                raise ValueError(
                    f"Refinement halos are too close: {first.name} and {second.name}; "
                    f"gap={gap:.3f} m, required >= {guard:.3f} m."
                )


def build_matrix_surface_plc(registry: PointRegistry, facets: List[Facet]) -> Tuple[np.ndarray, np.ndarray]:
    """Build the working matrix-only layered PLC."""
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
            area_1 = float((p10[0] - p00[0]) * (p11[1] - p00[1]) - (p10[1] - p00[1]) * (p11[0] - p00[0]))
            area_2 = float((p11[0] - p00[0]) * (p01[1] - p00[1]) - (p11[1] - p00[1]) * (p01[0] - p00[0]))
            min_area = min(min_area, area_1, area_2)
            if area_1 <= 1.0e-8 or area_2 <= 1.0e-8:
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


def _mix64(value: int) -> int:
    """Stable integer hash used for deterministic BCC jitter."""
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value & ((1 << 64) - 1)


def _uniform_hash(*values: int) -> float:
    state = 0x9E3779B97F4A7C15
    for value in values:
        state = _mix64(state ^ (int(value) + 0x9E3779B97F4A7C15))
    return state / float(1 << 64)


def _target_seed(target: RefinementTarget) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(target.name))


def _rotated_xy(local_x: float, local_y: float, yaw_deg: float) -> Tuple[float, float]:
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * local_x - sine * local_y, sine * local_x + cosine * local_y


def _local_candidate_bounds(target: RefinementTarget, outer_radius_m: float) -> Tuple[float, float]:
    """Return local z bounds for BCC candidates before global clipping."""
    if target.tag_shape == "vertical_cylinder":
        assert target.lattice_z_min_m is not None and target.lattice_z_max_m is not None
        return target.lattice_z_min_m - target.center_xyz[2], target.lattice_z_max_m - target.center_xyz[2]
    if target.tag_shape == "sphere":
        return -outer_radius_m, outer_radius_m
    raise ValueError(f"Unsupported target shape: {target.tag_shape}")


def _integer_index_bounds(lower: float, upper: float, spacing_m: float, phase: float, subshift: float) -> range:
    """Inclusive integer indices that safely cover a coordinate interval."""
    start = int(math.floor(lower / spacing_m - phase - subshift)) - 1
    stop = int(math.ceil(upper / spacing_m - phase - subshift)) + 1
    return range(start, stop + 1)


def _inside_domain_with_clearance(point: np.ndarray, spacing_m: float) -> bool:
    # Shallow AVN2/AVN87 halos are intentionally clipped by the external bottom
    # boundary.  Do not place free Part-1 points on the boundary itself.
    clearance = max(DOMAIN_POINT_CLEARANCE_M, 0.12 * spacing_m)
    return bool(np.all(point >= DOMAIN_MIN + clearance) and np.all(point <= DOMAIN_MAX - clearance))


def _point_in_band(
    target: RefinementTarget,
    point: np.ndarray,
    inner_radius_m: float,
    outer_radius_m: float,
    band_spacing_m: float,
) -> bool:
    centre = np.asarray(target.center_xyz, dtype=float)
    if target.tag_shape == "vertical_cylinder":
        assert target.lattice_z_min_m is not None and target.lattice_z_max_m is not None
        radial_distance = math.hypot(point[0] - centre[0], point[1] - centre[1])
        z_buffer = 0.12 * band_spacing_m
        if not (target.lattice_z_min_m + z_buffer <= point[2] <= target.lattice_z_max_m - z_buffer):
            return False
    elif target.tag_shape == "sphere":
        radial_distance = float(np.linalg.norm(point - centre))
    else:
        raise ValueError(f"Unsupported target shape: {target.tag_shape}")

    overlap = 0.55 * band_spacing_m
    lower = max(0.0, inner_radius_m - overlap)
    upper = outer_radius_m + overlap
    return lower <= radial_distance <= upper


def iter_bcc_candidates(
    target: RefinementTarget,
    outer_radius_m: float,
    spacing_m: float,
    band_index: int,
) -> Iterator[np.ndarray]:
    """Yield a lightly jittered, yaw-rotated BCC point cloud around one target."""
    centre = np.asarray(target.center_xyz, dtype=float)
    z_lower, z_upper = _local_candidate_bounds(target, outer_radius_m)
    extent = outer_radius_m + 0.75 * spacing_m
    phase_x = _uniform_hash(_target_seed(target), band_index, 101)
    phase_y = _uniform_hash(_target_seed(target), band_index, 211)
    phase_z = _uniform_hash(_target_seed(target), band_index, 307)

    # A BCC lattice is the union of a cubic lattice and a half-cell-shifted
    # lattice.  It is a good volumetric seed distribution for Delaunay meshes.
    for sublattice, shift in enumerate(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))):
        x_indices = _integer_index_bounds(-extent, extent, spacing_m, phase_x, shift[0])
        y_indices = _integer_index_bounds(-extent, extent, spacing_m, phase_y, shift[1])
        z_indices = _integer_index_bounds(z_lower - 0.75 * spacing_m, z_upper + 0.75 * spacing_m, spacing_m, phase_z, shift[2])
        for i in x_indices:
            base_x = (i + phase_x + shift[0]) * spacing_m
            if abs(base_x) > extent:
                continue
            for j in y_indices:
                base_y = (j + phase_y + shift[1]) * spacing_m
                if abs(base_y) > extent:
                    continue
                for k in z_indices:
                    base_z = (k + phase_z + shift[2]) * spacing_m
                    if base_z < z_lower - 0.75 * spacing_m or base_z > z_upper + 0.75 * spacing_m:
                        continue
                    jitter_amplitude = BCC_JITTER_FRACTION * spacing_m
                    jitter_x = jitter_amplitude * (2.0 * _uniform_hash(i, j, k, sublattice, band_index, 1, _target_seed(target)) - 1.0)
                    jitter_y = jitter_amplitude * (2.0 * _uniform_hash(i, j, k, sublattice, band_index, 2, _target_seed(target)) - 1.0)
                    jitter_z = jitter_amplitude * (2.0 * _uniform_hash(i, j, k, sublattice, band_index, 3, _target_seed(target)) - 1.0)
                    rotated_x, rotated_y = _rotated_xy(base_x + jitter_x, base_y + jitter_y, target.yaw_deg)
                    yield centre + np.array([rotated_x, rotated_y, base_z + jitter_z], dtype=float)


def add_target_bcc_points(
    registry: PointRegistry,
    spacing_filter: PointSpacingFilter,
    target: RefinementTarget,
) -> Dict[str, int]:
    """Add a graded, non-polar BCC point cloud for one target."""
    before = len(registry.points)
    attempted = 0
    accepted = 0
    rejected_outside = 0
    rejected_separation = 0

    # Preserve the exact strainmeter coordinate as a mesh vertex.  It is the
    # observation location used later for extraction / material tagging.
    if target.tag_shape == "sphere":
        spacing_filter.add_forced(target.center_xyz)

    previous_outer = 0.0
    for band_index, band in enumerate(target.refinement_bands):
        inner = 0.0 if band_index == 0 else previous_outer * (1.0 - BAND_OVERLAP_FRACTION)
        for candidate in iter_bcc_candidates(target, band.outer_radius_m, band.spacing_m, band_index):
            attempted += 1
            if not _inside_domain_with_clearance(candidate, band.spacing_m):
                rejected_outside += 1
                continue
            if not _point_in_band(target, candidate, inner, band.outer_radius_m, band.spacing_m):
                continue
            minimum_distance = MIN_POINT_SEPARATION_FRACTION * band.spacing_m
            if spacing_filter.try_add(candidate, minimum_distance):
                accepted += 1
            else:
                rejected_separation += 1
        previous_outer = band.outer_radius_m

    return {
        "points_added": len(registry.points) - before,
        "attempted": attempted,
        "accepted": accepted,
        "rejected_outside_domain": rejected_outside,
        "rejected_minimum_separation": rejected_separation,
    }


def add_refinement_points(registry: PointRegistry) -> Dict[str, Dict[str, int]]:
    """Build all local point clouds after the matrix PLC points are present."""
    spacing_filter = PointSpacingFilter(registry)
    stats: Dict[str, Dict[str, int]] = {}
    for target in REFINEMENT_TARGETS:
        stats[target.name] = add_target_bcc_points(registry, spacing_filter, target)
    return stats


def target_tag_definition(target: RefinementTarget) -> Dict[str, object]:
    definition: Dict[str, object] = {
        "name": target.name,
        "kind": target.kind,
        "purpose": target.purpose,
        "material_id": target.material_id,
        "color_hint": target.color_hint,
        "tag_shape": target.tag_shape,
        "center_xyz_m": list(target.center_xyz),
        "tag_radius_m": target.tag_radius_m,
        "source_sensor_id": target.source_sensor_id,
    }
    if target.tag_shape == "vertical_cylinder":
        definition["tag_z_min_m"] = target.tag_z_min_m
        definition["tag_z_max_m"] = target.tag_z_max_m
    return definition


def write_poly(path: Path, registry: PointRegistry, facets: Sequence[Facet], regions: Sequence[Region]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# North Avant / Bartlesville matrix PLC with tag-only HEC and BCC-refined injection/sensor mesh zones\n")
        handle.write("# HEC, injection borehole, and strainmeters add Part-1 mesh points only; none creates a PLC hole.\n\n")
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


def write_sidecars(
    mesh_prefix: str,
    point_count: int,
    facet_count: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_stats: Dict[str, Dict[str, int]],
) -> None:
    length_axis, width_axis, up_axis = hec_axes()
    tag_u = np.arange(-280.0, 280.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)
    tag_v = np.arange(-140.0, 140.0 + 1.0e-9, ROTATED_TAG_LATTICE_STEP_M)

    target_records = []
    for target in REFINEMENT_TARGETS:
        record = {
            **target_tag_definition(target),
            "yaw_deg": target.yaw_deg,
            "outer_refinement_radius_m": target.outer_radius_m,
            "refinement_bands": [asdict(band) for band in target.refinement_bands],
            "local_point_stats": target_stats[target.name],
        }
        if target.tag_shape == "vertical_cylinder":
            record["lattice_z_min_m"] = target.lattice_z_min_m
            record["lattice_z_max_m"] = target.lattice_z_max_m
        local_u, local_v = hec_local_uv(target.center_xyz[0], target.center_xyz[1])
        record["hec_local_uv_m"] = [local_u, local_v]
        target_records.append(record)

    geometry = {
        "domain": {"min": DOMAIN_MIN.tolist(), "max": DOMAIN_MAX.tolist(), "size": DOMAIN_SIZE.tolist()},
        "layers": [asdict(layer) for layer in LAYERS],
        "vertical_bands": [asdict(band) for band in VERTICAL_BANDS],
        "z_levels_m": list(Z_LEVELS),
        "meshing": {
            "strategy": "matrix-only layered PLC plus compact graded BCC Part-1 refinement clouds; no polar shells or radial spokes",
            "base_x_axis_m": x_values.tolist(),
            "base_y_axis_m": y_values.tolist(),
            "rotated_hec_tag_lattice_step_m": ROTATED_TAG_LATTICE_STEP_M,
            "tetgen_flags": DEFAULT_TETGEN_FLAGS,
            "no_tetgen_a": True,
            "local_refinement": {
                "method": "compact_overlapping_yaw_rotated_jittered_body_centred_cubic_point_clouds",
                "band_overlap_fraction": BAND_OVERLAP_FRACTION,
                "bcc_jitter_fraction": BCC_JITTER_FRACTION,
                "minimum_point_separation_fraction": MIN_POINT_SEPARATION_FRACTION,
                "point_hash_bin_size_m": POINT_HASH_BIN_SIZE_M,
                "shallow_sensor_boundary_handling": "spherical halos are clipped inside the external domain with a positive point clearance; no free point is placed on z=0",
            },
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
            "axes": {"length": length_axis.tolist(), "width": width_axis.tolist(), "normal_up": up_axis.tolist()},
            "tagging": {
                "method": "strict_z530_vertex_centres_inside_exact_oriented_rectangle",
                "expected_vertical_dual_support_m": [527.5, 532.5],
                "expected_tagged_vertex_count": int(tag_u.size * tag_v.size),
            },
        },
        "refinement_targets": target_records,
        "strainmeters": [
            {
                "sensor_id": item.sensor_id,
                "purpose": item.purpose,
                "location_xyz_m": [item.x_m, item.y_m, item.z_m],
                "material_id": STRAINMETER_MATERIAL_IDS[item.sensor_id],
                "representation": "spherical_sensor_pod",
                "tag_radius_m": STRAINMETER_TAG_RADIUS_M,
            }
            for item in STRAINMETER_INPUTS
        ],
        "plc": {
            "point_count": point_count,
            "facet_count": facet_count,
            "holes": 0,
            "regions": 4,
            "contains_hec_facets": False,
            "contains_hec_region": False,
            "contains_target_facets": False,
            "contains_target_regions": False,
            "contains_target_local_mesh_points": True,
        },
        "boundary_markers": BOUNDARY_MARKERS,
    }
    Path(f"{mesh_prefix}_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")

    with Path(f"{mesh_prefix}_hec_tag_geometry.xyz").open("w", encoding="utf-8") as handle:
        handle.write("# Exact HEC prism corners; diagnostic only, not PLC facets.\n# id x_m y_m z_m\n")
        for index, corner in enumerate(hec_corners(), start=1):
            handle.write(f"{index} {corner[0]:.10f} {corner[1]:.10f} {corner[2]:.10f}\n")

    with Path(f"{mesh_prefix}_refinement_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "name", "kind", "purpose", "source_sensor_id", "material_id", "color_hint", "tag_shape",
            "center_x_m", "center_y_m", "center_z_m", "tag_radius_m", "tag_z_min_m", "tag_z_max_m",
            "lattice_z_min_m", "lattice_z_max_m", "yaw_deg", "outer_refinement_radius_m", "points_added",
        ])
        for target in REFINEMENT_TARGETS:
            stats = target_stats[target.name]
            writer.writerow([
                target.name, target.kind, target.purpose, target.source_sensor_id or "", target.material_id,
                target.color_hint, target.tag_shape, f"{target.center_xyz[0]:.10f}", f"{target.center_xyz[1]:.10f}",
                f"{target.center_xyz[2]:.10f}", f"{target.tag_radius_m:.10f}",
                "" if target.tag_z_min_m is None else f"{target.tag_z_min_m:.10f}",
                "" if target.tag_z_max_m is None else f"{target.tag_z_max_m:.10f}",
                "" if target.lattice_z_min_m is None else f"{target.lattice_z_min_m:.10f}",
                "" if target.lattice_z_max_m is None else f"{target.lattice_z_max_m:.10f}",
                f"{target.yaw_deg:.10f}", f"{target.outer_radius_m:.10f}", stats["points_added"],
            ])

    with Path(f"{mesh_prefix}_strainmeters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "purpose", "x_m", "y_m", "z_m", "material_id", "tag_radius_m", "hec_local_u_m", "hec_local_v_m"])
        for item in STRAINMETER_INPUTS:
            local_u, local_v = hec_local_uv(item.x_m, item.y_m)
            writer.writerow([
                item.sensor_id, item.purpose, f"{item.x_m:.10f}", f"{item.y_m:.10f}", f"{item.z_m:.10f}",
                STRAINMETER_MATERIAL_IDS[item.sensor_id], f"{STRAINMETER_TAG_RADIUS_M:.10f}",
                f"{local_u:.10f}", f"{local_v:.10f}",
            ])

    with Path(f"{mesh_prefix}_bcc_refinement_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "target_name", "target_kind", "band_index", "outer_radius_m", "bcc_spacing_m",
            "nominal_inner_radius_m", "band_overlap_fraction", "minimum_separation_m",
        ])
        for target in REFINEMENT_TARGETS:
            previous_outer = 0.0
            for index, band in enumerate(target.refinement_bands):
                inner = 0.0 if index == 0 else previous_outer * (1.0 - BAND_OVERLAP_FRACTION)
                writer.writerow([
                    target.name, target.kind, index, f"{band.outer_radius_m:.6f}", f"{band.spacing_m:.6f}",
                    f"{inner:.6f}", f"{BAND_OVERLAP_FRACTION:.6f}",
                    f"{MIN_POINT_SEPARATION_FRACTION * band.spacing_m:.6f}",
                ])
                previous_outer = band.outer_radius_m

    with Path(f"{mesh_prefix}_vertical_grading.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "z_min_m", "z_max_m", "vertical_thickness_m", "geological_layer", "note"])
        writer.writeheader()
        for band in VERTICAL_BANDS:
            writer.writerow({
                "label": band.label,
                "z_min_m": f"{band.z_min:.6f}",
                "z_max_m": f"{band.z_max:.6f}",
                "vertical_thickness_m": f"{band.z_max - band.z_min:.6f}",
                "geological_layer": band.geological_layer,
                "note": band.note,
            })


def build_geometry(mesh_prefix: str) -> Tuple[Path, Dict[str, int]]:
    validate_configuration()
    registry = PointRegistry()
    facets: List[Facet] = []
    x_values, y_values = build_matrix_surface_plc(registry, facets)
    target_stats = add_refinement_points(registry)
    regions = (
        Region(np.array([5000.0, 5000.0, 500.0]), 1, "overburden"),
        Region(np.array([5000.0, 5000.0, 210.0]), 2, "bartlesville_sand"),
        Region(np.array([5000.0, 5000.0, 235.0]), 3, "basal_layer"),
        Region(np.array([5000.0, 5000.0, 100.0]), 4, "underburden"),
    )
    poly_path = Path(f"{mesh_prefix}.poly")
    write_poly(poly_path, registry, facets, regions)
    write_sidecars(mesh_prefix, len(registry.points), len(facets), x_values, y_values, target_stats)
    return poly_path, {
        "points": len(registry.points),
        "facets": len(facets),
        "hec_plc_points": 0,
        "hec_plc_facets": 0,
        "hec_plc_regions": 0,
        "local_refinement_points": sum(stat["points_added"] for stat in target_stats.values()),
        "target_plc_facets": 0,
        "target_plc_regions": 0,
        "holes": 0,
        "regions": len(regions),
        **{f"{name}_points": stat["points_added"] for name, stat in target_stats.items()},
    }


def _contains_tetgen_volume_flag(flags: str) -> bool:
    """Return True only when the -a option appears in a TetGen switch token."""
    for token in shlex.split(flags):
        if token.startswith("-") and "a" in token[1:]:
            return True
    return False


def run_tetgen(tetgen_exe: str, poly_path: Path, diagnose: bool) -> None:
    flags = os.environ.get("BARTLESVILLE_TETGEN_FLAGS", DEFAULT_TETGEN_FLAGS).strip()
    if not flags:
        raise ValueError("BARTLESVILLE_TETGEN_FLAGS cannot be empty.")
    if _contains_tetgen_volume_flag(flags):
        raise ValueError("Do not use TetGen -a in this low-cell workflow.")
    if diagnose and "d" not in flags:
        flags += "d"
    command = [tetgen_exe, *shlex.split(flags), str(poly_path)]
    print("\n--> Running TetGen")
    print("    HEC PLC entities       : 0 points, 0 facets, 0 regions (tag-only)")
    print("    target PLC entities    : 0 facets, 0 regions, 0 holes")
    print("    injection refinement   : compact graded yaw-rotated BCC cylinder, R=5 m tag, z=532.5--750 m")
    print("    strainmeter refinement : compact sensor pods at AVN2, AVN87, AVN31; R=2 m tags")
    print("    anti-sliver strategy   : BCC bands + yaw + deterministic jitter + point-spacing rejection")
    if "q" in flags:
        print("    WARNING                : -q enables iterative Steiner refinement and can be slow.")
    print("    TetGen flags           :", flags)
    print("CMD:", " ".join(shlex.quote(token) for token in command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bartlesville PLC with tag-only HEC and BCC-refined injection/strainmeter mesh zones.")
    parser.add_argument("mesh_prefix", help="Output mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("tetgen_exe", nargs="?", help="TetGen executable path unless --write-only is used")
    parser.add_argument("--write-only", action="store_true", help="Write .poly and sidecars without TetGen")
    parser.add_argument("--diagnose", action="store_true", help="Append TetGen -d diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix.removesuffix(".poly")
    poly_path, counts = build_geometry(prefix)
    print("\n--> Wrote Bartlesville PLC with tag-only HEC and compact BCC-refined local mesh zones")
    print(f"    poly file              : {poly_path}")
    print(f"    geometry JSON          : {prefix}_geometry.json")
    print(f"    HEC diagnostic         : {prefix}_hec_tag_geometry.xyz")
    print(f"    target definitions     : {prefix}_refinement_targets.csv")
    print(f"    strainmeter locations  : {prefix}_strainmeters.csv")
    print(f"    BCC profile             : {prefix}_bcc_refinement_profile.csv")
    print(f"    vertical profile        : {prefix}_vertical_grading.csv")
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
