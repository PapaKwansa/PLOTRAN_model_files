#!/usr/bin/env python3
"""
Build the deterministic TetGen PLC for the V1 homogeneous 3-D
COMSOL-PFLOTRAN verification benchmark.

V1-M2 changes relative to the diagnostic M1 mesh
--------------------------------------------------
1. The six external domain faces are explicitly tessellated using the
   regional mesh spacing. Boundary vertices are therefore part of the PLC.
2. The regional lattice reuses those same boundary vertices instead of
   placing free vertices on constrained PLC facets.
3. The first four injector refinement shells use a common axial grid based
   on the finest inner-shell axial spacing. This reduces axial-grid mismatch
   between the highest-gradient shells.
4. Normal output uses the "_m2" stem so the previous M1 files are preserved.

V1 contains only:
    * homogeneous matrix
    * finite cylindrical injector
    * outer rectangular domain
    * graded local refinement around injector

No geological layers, HEC, or strainmeters are included.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TOL = 1.0e-9


# ============================================================================
# Basic helpers
# ============================================================================

def add_point(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    xyz: tuple[float, float, float],
) -> int:
    """Add a unique point and return its 1-based TetGen point ID."""

    key = tuple(round(float(v), 10) for v in xyz)

    if key in registry:
        return registry[key]

    points.append(
        (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
    )

    point_id = len(points)
    registry[key] = point_id

    return point_id


def make_axis(
    start: float,
    stop: float,
    spacing: float,
) -> list[float]:
    """
    Create a deterministic axis with nominal spacing and exact endpoints.
    """

    if spacing <= 0.0:
        raise ValueError("Axis spacing must be positive.")

    values = [float(start)]
    current = float(start)

    while current + spacing < stop - TOL:
        current += spacing
        values.append(float(current))

    if not math.isclose(
        values[-1],
        stop,
        abs_tol=TOL,
    ):
        values.append(float(stop))

    return values


def make_common_centered_levels(
    center: float,
    half_extent: float,
    spacing: float,
) -> list[float]:
    """
    Create symmetric axial levels centered on the injector.

    Levels are generated at center + n*spacing and clipped to the requested
    half extent. Injector end-cap planes are excluded explicitly.
    """

    if spacing <= 0.0:
        raise ValueError(
            "Common inner-shell axial spacing must be positive."
        )

    nmax = int(
        math.floor(
            half_extent / spacing + TOL
        )
    )

    values = []

    for n in range(
        -nmax,
        nmax + 1,
    ):
        z = center + n * spacing

        if (
            z < center - half_extent - TOL
            or z > center + half_extent + TOL
        ):
            continue

        values.append(float(z))

    return values


def inside_injector(
    x: float,
    y: float,
    z: float,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    height: float,
) -> bool:
    """Return True when a point is strictly inside the injector cylinder."""

    radial2 = (
        (x - cx) ** 2
        + (y - cy) ** 2
    )

    zmin = cz - 0.5 * height
    zmax = cz + 0.5 * height

    return (
        radial2 < (radius - TOL) ** 2
        and zmin + TOL < z < zmax - TOL
    )


def on_or_near_injector_envelope(
    x: float,
    y: float,
    z: float,
    cx: float,
    cy: float,
    cz: float,
    outer_radius: float,
    z_half_extent: float,
) -> bool:
    """Identify coarse lattice points that should yield to local refinement."""

    radial = math.hypot(
        x - cx,
        y - cy,
    )

    return (
        radial < outer_radius
        and abs(z - cz) < z_half_extent
    )


# ============================================================================
# Outer-domain structured PLC
# ============================================================================

def add_grid_face(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    facets: list[tuple[tuple[int, int, int], int]],
    u_values: list[float],
    v_values: list[float],
    point_function,
    marker: int,
) -> None:
    """
    Add a structured quadrilateral surface as two triangles per cell.

    point_function(u, v) -> (x, y, z)

    The structured surface points are added to the shared point registry,
    so the regional interior lattice will reuse exactly the same boundary
    coordinates.
    """

    ids: list[list[int]] = []

    for v in v_values:

        row = []

        for u in u_values:

            xyz = point_function(
                float(u),
                float(v),
            )

            row.append(
                add_point(
                    points,
                    registry,
                    xyz,
                )
            )

        ids.append(row)

    for j in range(
        len(v_values) - 1
    ):

        for i in range(
            len(u_values) - 1
        ):

            p00 = ids[j][i]
            p10 = ids[j][i + 1]
            p11 = ids[j + 1][i + 1]
            p01 = ids[j + 1][i]

            facets.append(
                (
                    (
                        p00,
                        p10,
                        p11,
                    ),
                    marker,
                )
            )

            facets.append(
                (
                    (
                        p00,
                        p11,
                        p01,
                    ),
                    marker,
                )
            )


def add_box_geometry(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    facets: list[tuple[tuple[int, int, int], int]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    regional_spacing: float,
) -> dict[str, Any]:
    """
    Create a structured, regionally resolved outer-box PLC.

    Boundary markers:
        top    = 1
        bottom = 2
        north  = 3
        south  = 4
        east   = 5
        west   = 6
    """

    x_values = make_axis(
        xmin,
        xmax,
        regional_spacing,
    )

    y_values = make_axis(
        ymin,
        ymax,
        regional_spacing,
    )

    z_values = make_axis(
        zmin,
        zmax,
        regional_spacing,
    )

    facet_start = len(facets)

    # Bottom z = zmin, marker 2.
    # Parametric orientation gives outward normal approximately -z.
    add_grid_face(
        points,
        registry,
        facets,
        x_values,
        y_values,
        lambda x, y: (
            x,
            y,
            zmin,
        ),
        2,
    )

    # Top z = zmax, marker 1.
    add_grid_face(
        points,
        registry,
        facets,
        x_values,
        y_values,
        lambda x, y: (
            x,
            y,
            zmax,
        ),
        1,
    )

    # South y = ymin, marker 4.
    add_grid_face(
        points,
        registry,
        facets,
        x_values,
        z_values,
        lambda x, z: (
            x,
            ymin,
            z,
        ),
        4,
    )

    # North y = ymax, marker 3.
    add_grid_face(
        points,
        registry,
        facets,
        x_values,
        z_values,
        lambda x, z: (
            x,
            ymax,
            z,
        ),
        3,
    )

    # West x = xmin, marker 6.
    add_grid_face(
        points,
        registry,
        facets,
        y_values,
        z_values,
        lambda y, z: (
            xmin,
            y,
            z,
        ),
        6,
    )

    # East x = xmax, marker 5.
    add_grid_face(
        points,
        registry,
        facets,
        y_values,
        z_values,
        lambda y, z: (
            xmax,
            y,
            z,
        ),
        5,
    )

    facet_count = len(facets) - facet_start

    return {
        "x_points": len(x_values),
        "y_points": len(y_values),
        "z_points": len(z_values),
        "facet_count": facet_count,
        "x_spacing_nominal_m": regional_spacing,
        "y_spacing_nominal_m": regional_spacing,
        "z_spacing_nominal_m": regional_spacing,
    }


# ============================================================================
# Injector PLC
# ============================================================================

def make_cylinder_z_levels(
    zmin: float,
    zmax: float,
    spacing: float,
) -> list[float]:
    """Create exact axial levels for the injector cylinder."""

    if spacing <= 0.0:
        raise ValueError(
            "Cylinder axial spacing must be positive."
        )

    levels = [float(zmin)]
    current = float(zmin)

    while current + spacing < zmax - TOL:
        current += spacing
        levels.append(float(current))

    if not math.isclose(
        levels[-1],
        zmax,
        abs_tol=TOL,
    ):
        levels.append(float(zmax))

    return levels


def add_injector_geometry(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    facets: list[tuple[tuple[int, int, int], int]],
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    height: float,
    circumferential_points: int,
    axial_spacing: float,
) -> dict[str, Any]:
    """Create the closed internal injector-cylinder PLC."""

    if circumferential_points < 8:
        raise ValueError(
            "Injector requires at least 8 circumferential points."
        )

    if circumferential_points % 2 != 0:
        raise ValueError(
            "Injector circumferential point count must be even."
        )

    zmin = cz - 0.5 * height
    zmax = cz + 0.5 * height

    z_levels = make_cylinder_z_levels(
        zmin,
        zmax,
        axial_spacing,
    )

    theta = np.linspace(
        0.0,
        2.0 * math.pi,
        circumferential_points,
        endpoint=False,
    )

    rings: list[list[int]] = []

    for z in z_levels:

        ring = []

        for angle in theta:

            x = (
                cx
                + radius
                * math.cos(float(angle))
            )

            y = (
                cy
                + radius
                * math.sin(float(angle))
            )

            ring.append(
                add_point(
                    points,
                    registry,
                    (x, y, z),
                )
            )

        rings.append(ring)

    injector_marker = 7

    # Cylinder side wall.
    for k in range(
        len(rings) - 1
    ):

        lower = rings[k]
        upper = rings[k + 1]

        for j in range(
            circumferential_points
        ):

            j2 = (
                j + 1
            ) % circumferential_points

            facets.append(
                (
                    (
                        lower[j],
                        lower[j2],
                        upper[j],
                    ),
                    injector_marker,
                )
            )

            facets.append(
                (
                    (
                        lower[j2],
                        upper[j2],
                        upper[j],
                    ),
                    injector_marker,
                )
            )

    bottom_center = add_point(
        points,
        registry,
        (cx, cy, zmin),
    )

    top_center = add_point(
        points,
        registry,
        (cx, cy, zmax),
    )

    bottom_ring = rings[0]
    top_ring = rings[-1]

    # Bottom cap.
    for j in range(
        circumferential_points
    ):

        j2 = (
            j + 1
        ) % circumferential_points

        facets.append(
            (
                (
                    bottom_center,
                    bottom_ring[j2],
                    bottom_ring[j],
                ),
                injector_marker,
            )
        )

    # Top cap.
    for j in range(
        circumferential_points
    ):

        j2 = (
            j + 1
        ) % circumferential_points

        facets.append(
            (
                (
                    top_center,
                    top_ring[j],
                    top_ring[j2],
                ),
                injector_marker,
            )
        )

    return {
        "center": [
            cx,
            cy,
            cz,
        ],
        "radius_m": radius,
        "height_m": height,
        "zmin_m": zmin,
        "zmax_m": zmax,
        "volume_m3": (
            math.pi
            * radius ** 2
            * height
        ),
        "circumferential_points": (
            circumferential_points
        ),
        "axial_levels": len(z_levels),
        "surface_marker": injector_marker,
    }


# ============================================================================
# Regional lattice
# ============================================================================

def add_regional_lattice(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    domain: dict[str, float],
    injector: dict[str, Any],
    regional_spacing: float,
    local_exclusion_radius: float,
    local_exclusion_z_half_extent: float,
) -> int:
    """
    Add the deterministic regional lattice.

    Boundary points are intentionally included because the outer boundary is
    now explicitly tessellated using the same coordinates. The shared registry
    prevents duplicate nodes.
    """

    x_values = make_axis(
        domain["xmin_m"],
        domain["xmax_m"],
        regional_spacing,
    )

    y_values = make_axis(
        domain["ymin_m"],
        domain["ymax_m"],
        regional_spacing,
    )

    z_values = make_axis(
        domain["zmin_m"],
        domain["zmax_m"],
        regional_spacing,
    )

    cx = float(
        injector["center_m"]["x"]
    )

    cy = float(
        injector["center_m"]["y"]
    )

    cz = float(
        injector["center_m"]["z"]
    )

    radius = float(
        injector["radius_m"]
    )

    height = float(
        injector["height_m"]
    )

    added = 0

    for z in z_values:

        for y in y_values:

            for x in x_values:

                # Do not add matrix points strictly inside the injector.
                if inside_injector(
                    x,
                    y,
                    z,
                    cx,
                    cy,
                    cz,
                    radius,
                    height,
                ):
                    continue

                # Yield the local near-field to the dedicated injector
                # refinement shells.
                if on_or_near_injector_envelope(
                    x,
                    y,
                    z,
                    cx,
                    cy,
                    cz,
                    local_exclusion_radius,
                    local_exclusion_z_half_extent,
                ):
                    continue

                before = len(points)

                add_point(
                    points,
                    registry,
                    (x, y, z),
                )

                if len(points) > before:
                    added += 1

    return added


# ============================================================================
# Injector shell refinement
# ============================================================================

def add_injector_shell_points(
    points: list[tuple[float, float, float]],
    registry: dict[tuple[float, float, float], int],
    injector: dict[str, Any],
    shells: list[dict[str, Any]],
    z_half_extent_m: float = 110.0,
    inner_aligned_shell_count: int = 4,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Add deterministic coaxial injector shells.

    V1-M2:
      * the first four shells share a common axial grid;
      * the common spacing is the finest spacing among those inner shells;
      * outer shells retain their configured axial spacing.
    """

    cx = float(
        injector["center_m"]["x"]
    )

    cy = float(
        injector["center_m"]["y"]
    )

    cz = float(
        injector["center_m"]["z"]
    )

    injector_radius = float(
        injector["radius_m"]
    )

    injector_height = float(
        injector["height_m"]
    )

    zmin = (
        cz
        - 0.5
        * injector_height
    )

    zmax = (
        cz
        + 0.5
        * injector_height
    )

    if inner_aligned_shell_count < 1:
        raise ValueError(
            "inner_aligned_shell_count must be positive."
        )

    inner_count = min(
        inner_aligned_shell_count,
        len(shells),
    )

    inner_spacing = min(
        float(
            shell["axial_spacing_m"]
        )
        for shell in shells[:inner_count]
    )

    common_z_values = (
        make_common_centered_levels(
            cz,
            z_half_extent_m,
            inner_spacing,
        )
    )

    # Do not allow an aligned shell to contain a point exactly on the
    # constrained injector cap planes.
    common_z_values = [
        z
        for z in common_z_values
        if (
            not math.isclose(
                z,
                zmin,
                abs_tol=1.0e-6,
            )
            and not math.isclose(
                z,
                zmax,
                abs_tol=1.0e-6,
            )
        )
    ]

    accepted_by_shell: dict[str, int] = {}

    for shell_index, shell in enumerate(
        shells
    ):

        radius = float(
            shell["radius_m"]
        )

        configured_spacing = float(
            shell["axial_spacing_m"]
        )

        ntheta = int(
            shell["circumferential_points"]
        )

        if radius <= injector_radius:
            raise ValueError(
                f"Refinement shell radius {radius} m "
                f"is not outside injector radius "
                f"{injector_radius} m."
            )

        if ntheta < 8 or ntheta % 2 != 0:
            raise ValueError(
                "Shell circumferential point count "
                "must be even and >= 8."
            )

        if shell_index < inner_count:
            z_values = common_z_values
            axial_mode = (
                f"aligned-{inner_spacing:g}m"
            )
        else:
            z_values = (
                make_common_centered_levels(
                    cz,
                    z_half_extent_m,
                    configured_spacing,
                )
            )

            z_values = [
                z
                for z in z_values
                if (
                    not math.isclose(
                        z,
                        zmin,
                        abs_tol=1.0e-6,
                    )
                    and not math.isclose(
                        z,
                        zmax,
                        abs_tol=1.0e-6,
                    )
                )
            ]

            axial_mode = (
                f"configured-{configured_spacing:g}m"
            )

        # Fixed deterministic phase by shell.
        shell_phase = math.radians(
            11.0 * shell_index
        )

        count = 0

        for iz, z in enumerate(
            z_values
        ):

            # Alternate half-step circumferential staggering by axial level.
            local_phase = (
                shell_phase
                + (
                    math.pi / ntheta
                    if iz % 2
                    else 0.0
                )
            )

            for j in range(
                ntheta
            ):

                angle = (
                    2.0
                    * math.pi
                    * j
                    / ntheta
                    + local_phase
                )

                x = (
                    cx
                    + radius
                    * math.cos(angle)
                )

                y = (
                    cy
                    + radius
                    * math.sin(angle)
                )

                before = len(points)

                add_point(
                    points,
                    registry,
                    (x, y, z),
                )

                if len(points) > before:
                    count += 1

        accepted_by_shell[
            f"{radius:g}m"
        ] = count

        # Store the axial mode separately for metadata.
        accepted_by_shell[
            f"{radius:g}m_axial_mode"
        ] = axial_mode

    return accepted_by_shell, {
        "inner_aligned_shell_count": inner_count,
        "inner_common_axial_spacing_m": inner_spacing,
        "z_half_extent_m": z_half_extent_m,
    }


# ============================================================================
# TetGen writer
# ============================================================================

def write_poly(
    path: Path,
    points: list[tuple[float, float, float]],
    facets: list[tuple[tuple[int, int, int], int]],
    regions: list[dict[str, Any]],
) -> None:
    """Write a TetGen .poly file."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            "# V1 homogeneous 3-D "
            "COMSOL-PFLOTRAN verification PLC\n"
        )

        handle.write(
            "# V1-M2: structured outer PLC "
            "and aligned inner shell axial grid.\n\n"
        )

        handle.write(
            "# Part 1 - node list\n"
        )

        handle.write(
            f"{len(points)} 3 0 0\n"
        )

        for point_id, xyz in enumerate(
            points,
            start=1,
        ):

            handle.write(
                f"{point_id} "
                f"{xyz[0]:.10f} "
                f"{xyz[1]:.10f} "
                f"{xyz[2]:.10f}\n"
            )

        handle.write("\n")

        handle.write(
            "# Part 2 - facet list\n"
        )

        handle.write(
            f"{len(facets)} 1\n"
        )

        for triangle, marker in facets:

            handle.write(
                f"1 0 {marker}\n"
            )

            handle.write(
                f"3 "
                f"{triangle[0]} "
                f"{triangle[1]} "
                f"{triangle[2]}\n"
            )

        handle.write("\n")

        handle.write(
            "# Part 3 - holes\n"
        )

        handle.write(
            "0\n\n"
        )

        handle.write(
            "# Part 4 - regions\n"
        )

        handle.write(
            f"{len(regions)}\n"
        )

        for index, region in enumerate(
            regions,
            start=1,
        ):

            x, y, z = region["point"]

            attribute = int(
                region["attribute"]
            )

            handle.write(
                f"{index} "
                f"{x:.10f} "
                f"{y:.10f} "
                f"{z:.10f} "
                f"{attribute}\n"
            )


# ============================================================================
# Geometry metadata
# ============================================================================

def write_geometry_json(
    path: Path,
    domain: dict[str, float],
    injector: dict[str, Any],
    shell_stats: dict[str, int],
    shell_metadata: dict[str, Any],
    boundary_metadata: dict[str, Any],
    point_count: int,
    facet_count: int,
    diagnostic_mode: str | None,
) -> None:
    """Write V1 geometry metadata."""

    clean_shell_stats = {}

    for key, value in shell_stats.items():

        if key.endswith(
            "_axial_mode"
        ):
            continue

        clean_shell_stats[key] = value

    shell_axial_modes = {}

    for key, value in shell_stats.items():

        if key.endswith(
            "_axial_mode"
        ):
            shell_name = key[
                :-len("_axial_mode")
            ]

            shell_axial_modes[
                shell_name
            ] = value

    data = {
        "model": {
            "name": "V1_homogeneous_3D",
            "mesh_version": "M2",
            "description": (
                "Homogeneous 3-D "
                "COMSOL-PFLOTRAN "
                "verification benchmark"
            ),
            "diagnostic_mode": diagnostic_mode,
        },

        "domain": {
            "min": [
                domain["xmin_m"],
                domain["ymin_m"],
                domain["zmin_m"],
            ],
            "max": [
                domain["xmax_m"],
                domain["ymax_m"],
                domain["zmax_m"],
            ],
        },

        "boundary_markers": {
            "top": 1,
            "bottom": 2,
            "north": 3,
            "south": 4,
            "east": 5,
            "west": 6,
        },

        "layers": [
            {
                "number": 1,
                "name": (
                    "homogeneous_matrix"
                ),
                "material_id": 1,
                "zmin_m": (
                    domain["zmin_m"]
                ),
                "zmax_m": (
                    domain["zmax_m"]
                ),
            }
        ],

        "hec": None,

        "injection": injector,

        "refinement_targets": [
            {
                "name": "injection",
                "material_id": 2,
                "kind": (
                    "finite_cylinder"
                ),
                "center_xyz_m": (
                    injector["center"]
                ),
                "radius_m": (
                    injector["radius_m"]
                ),
                "height_m": (
                    injector["height_m"]
                ),
            }
        ],

        "boundary_tessellation": (
            boundary_metadata
        ),

        "mesh": {
            "point_count": (
                point_count
            ),
            "facet_count": (
                facet_count
            ),
            "injector_shell_points": (
                clean_shell_stats
            ),
            "injector_shell_axial_modes": (
                shell_axial_modes
            ),
            "shell_alignment": (
                shell_metadata
            ),
        },
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================================
# Configuration
# ============================================================================

def validate_configuration(
    cfg: dict[str, Any],
) -> None:

    domain = cfg["domain"]
    injector = cfg["injector"]
    mesh = cfg["mesh"]

    if not (
        domain["xmin_m"]
        < domain["xmax_m"]
    ):
        raise ValueError(
            "Invalid X bounds."
        )

    if not (
        domain["ymin_m"]
        < domain["ymax_m"]
    ):
        raise ValueError(
            "Invalid Y bounds."
        )

    if not (
        domain["zmin_m"]
        < domain["zmax_m"]
    ):
        raise ValueError(
            "Invalid Z bounds."
        )

    if injector["radius_m"] <= 0:
        raise ValueError(
            "Injector radius must be positive."
        )

    if injector["height_m"] <= 0:
        raise ValueError(
            "Injector height must be positive."
        )

    cx = float(
        injector["center_m"]["x"]
    )

    cy = float(
        injector["center_m"]["y"]
    )

    cz = float(
        injector["center_m"]["z"]
    )

    radius = float(
        injector["radius_m"]
    )

    half_height = (
        0.5
        * float(
            injector["height_m"]
        )
    )

    if not (
        domain["xmin_m"] < cx < domain["xmax_m"]
        and domain["ymin_m"] < cy < domain["ymax_m"]
        and domain["zmin_m"] < cz - half_height
        and cz + half_height < domain["zmax_m"]
    ):
        raise ValueError(
            "Injector must lie strictly inside domain."
        )

    if mesh["regional_spacing_m"] <= 0:
        raise ValueError(
            "Regional spacing must be positive."
        )

    shells = mesh["radial_shells"]

    if not shells:
        raise ValueError(
            "At least one radial shell is required."
        )

    previous_radius = radius

    for shell in shells:

        current_radius = float(
            shell["radius_m"]
        )

        if current_radius <= previous_radius:
            raise ValueError(
                "Shell radii must increase strictly."
            )

        if (
            int(shell["circumferential_points"])
            < 8
            or int(shell["circumferential_points"]) % 2
        ):
            raise ValueError(
                "Shell circumferential counts must be "
                "even and >= 8."
            )

        previous_radius = current_radius


def load_config(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        cfg = yaml.safe_load(
            handle
        )

    if not isinstance(
        cfg,
        dict,
    ):
        raise ValueError(
            "YAML root must be a mapping."
        )

    return cfg


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build the V1 homogeneous "
            "3-D TetGen PLC."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "verification/"
            "V1_homogeneous_3D/"
            "config/"
            "v1_parameters.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/"
            "V1_homogeneous_3D/"
            "mesh/"
            "generated"
        ),
    )

    parser.add_argument(
        "--plc-only",
        action="store_true",
        help=(
            "Build only the outer box and "
            "injector PLC."
        ),
    )

    parser.add_argument(
        "--regional-only",
        action="store_true",
        help=(
            "Build PLC plus regional lattice, "
            "without injector shells."
        ),
    )

    parser.add_argument(
        "--shells-only",
        action="store_true",
        help=(
            "Build PLC plus injector shells, "
            "without regional lattice."
        ),
    )

    parser.add_argument(
        "--output-stem",
        default="v1_homogeneous_3d_m2",
        help=(
            "Base name for generated files."
        ),
    )

    args = parser.parse_args()

    diagnostic_modes = [
        args.plc_only,
        args.regional_only,
        args.shells_only,
    ]

    if sum(
        bool(mode)
        for mode in diagnostic_modes
    ) > 1:

        parser.error(
            "Choose only one of "
            "--plc-only, --regional-only, "
            "or --shells-only."
        )

    if args.plc_only:
        diagnostic_mode = (
            "PLC-only"
        )
        output_stem = (
            f"{args.output_stem}_plc_only"
        )

    elif args.regional_only:
        diagnostic_mode = (
            "PLC + regional lattice"
        )
        output_stem = (
            f"{args.output_stem}_regional_only"
        )

    elif args.shells_only:
        diagnostic_mode = (
            "PLC + injector shells"
        )
        output_stem = (
            f"{args.output_stem}_shells_only"
        )

    else:
        diagnostic_mode = None
        output_stem = args.output_stem

    cfg = load_config(
        args.config
    )

    validate_configuration(
        cfg
    )

    domain = cfg["domain"]
    injector = cfg["injector"]
    mesh_cfg = cfg["mesh"]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    points: list[
        tuple[float, float, float]
    ] = []

    registry: dict[
        tuple[float, float, float],
        int,
    ] = {}

    facets: list[
        tuple[
            tuple[int, int, int],
            int,
        ]
    ] = []

    # ------------------------------------------------------------------------
    # 1. Structured outer PLC
    # ------------------------------------------------------------------------

    boundary_metadata = add_box_geometry(
        points,
        registry,
        facets,
        float(
            domain["xmin_m"]
        ),
        float(
            domain["xmax_m"]
        ),
        float(
            domain["ymin_m"]
        ),
        float(
            domain["ymax_m"]
        ),
        float(
            domain["zmin_m"]
        ),
        float(
            domain["zmax_m"]
        ),
        float(
            mesh_cfg["regional_spacing_m"]
        ),
    )

    # ------------------------------------------------------------------------
    # 2. Injector PLC
    # ------------------------------------------------------------------------

    injector_meta = add_injector_geometry(
        points,
        registry,
        facets,
        float(
            injector["center_m"]["x"]
        ),
        float(
            injector["center_m"]["y"]
        ),
        float(
            injector["center_m"]["z"]
        ),
        float(
            injector["radius_m"]
        ),
        float(
            injector["height_m"]
        ),
        int(
            mesh_cfg[
                "injector_surface"
            ][
                "circumferential_points"
            ]
        ),
        float(
            mesh_cfg[
                "injector_surface"
            ][
                "maximum_axial_spacing_m"
            ]
        ),
    )

    # ------------------------------------------------------------------------
    # 3. Free point populations
    # ------------------------------------------------------------------------

    regional_spacing = float(
        mesh_cfg[
            "regional_spacing_m"
        ]
    )

    outer_shell_radius = float(
        mesh_cfg[
            "radial_shells"
        ][-1][
            "radius_m"
        ]
    )

    coarse_added = 0
    shell_stats: dict[str, int] = {}
    shell_metadata: dict[str, Any] = {
        "inner_aligned_shell_count": 0,
        "inner_common_axial_spacing_m": None,
        "z_half_extent_m": 110.0,
    }

    if args.plc_only:

        print("")
        print(
            "DIAGNOSTIC MODE: PLC only"
        )

    elif args.regional_only:

        print("")
        print(
            "DIAGNOSTIC MODE: "
            "PLC + regional lattice"
        )

        coarse_added = (
            add_regional_lattice(
                points,
                registry,
                domain,
                injector,
                regional_spacing,
                local_exclusion_radius=max(
                    outer_shell_radius * 1.15,
                    160.0,
                ),
                local_exclusion_z_half_extent=140.0,
            )
        )

    elif args.shells_only:

        print("")
        print(
            "DIAGNOSTIC MODE: "
            "PLC + injector shells"
        )

        (
            shell_stats,
            shell_metadata,
        ) = add_injector_shell_points(
            points,
            registry,
            injector,
            mesh_cfg[
                "radial_shells"
            ],
            z_half_extent_m=110.0,
            inner_aligned_shell_count=4,
        )

    else:

        print("")
        print(
            "FULL V1-M2 MODE"
        )

        coarse_added = (
            add_regional_lattice(
                points,
                registry,
                domain,
                injector,
                regional_spacing,
                local_exclusion_radius=max(
                    outer_shell_radius * 1.15,
                    160.0,
                ),
                local_exclusion_z_half_extent=140.0,
            )
        )

        (
            shell_stats,
            shell_metadata,
        ) = add_injector_shell_points(
            points,
            registry,
            injector,
            mesh_cfg[
                "radial_shells"
            ],
            z_half_extent_m=110.0,
            inner_aligned_shell_count=4,
        )

    # ------------------------------------------------------------------------
    # 4. TetGen region seeds
    # ------------------------------------------------------------------------

    cx = float(
        injector["center_m"]["x"]
    )

    cy = float(
        injector["center_m"]["y"]
    )

    cz = float(
        injector["center_m"]["z"]
    )

    radius = float(
        injector["radius_m"]
    )

    height = float(
        injector["height_m"]
    )

    matrix_seed = {
        "point": (
            1000.0,
            1000.0,
            -1200.0,
        ),
        "attribute": 1,
    }

    injector_seed = {
        "point": (
            cx,
            cy,
            cz,
        ),
        "attribute": 2,
    }

    if inside_injector(
        *matrix_seed["point"],
        cx,
        cy,
        cz,
        radius,
        height,
    ):
        raise RuntimeError(
            "Matrix seed is inside injector."
        )

    if not inside_injector(
        *injector_seed["point"],
        cx,
        cy,
        cz,
        radius,
        height,
    ):
        raise RuntimeError(
            "Injector seed is not inside injector."
        )

    regions = [
        matrix_seed,
        injector_seed,
    ]

    # ------------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------------

    poly_path = (
        args.output_dir
        / f"{output_stem}.poly"
    )

    geometry_path = (
        args.output_dir
        / f"{output_stem}_geometry.json"
    )

    write_poly(
        poly_path,
        points,
        facets,
        regions,
    )

    write_geometry_json(
        geometry_path,
        domain,
        injector_meta,
        shell_stats,
        shell_metadata,
        boundary_metadata,
        len(points),
        len(facets),
        diagnostic_mode,
    )

    injector_volume = (
        math.pi
        * radius ** 2
        * height
    )

    print("")
    print(
        "V1 homogeneous mesh geometry generated"
    )
    print(
        "=" * 64
    )

    print(
        f"Configuration : {args.config}"
    )

    print(
        f"Output dir    : {args.output_dir}"
    )

    if args.plc_only:
        mode_name = (
            "PLC-only diagnostic"
        )
    elif args.regional_only:
        mode_name = (
            "PLC + regional lattice diagnostic"
        )
    elif args.shells_only:
        mode_name = (
            "PLC + injector shells diagnostic"
        )
    else:
        mode_name = "FULL V1-M2"

    print(
        f"Mode          : {mode_name}"
    )

    print("")

    print("Domain")

    print(
        f"  X : "
        f"{domain['xmin_m']} "
        f"-> "
        f"{domain['xmax_m']} m"
    )

    print(
        f"  Y : "
        f"{domain['ymin_m']} "
        f"-> "
        f"{domain['ymax_m']} m"
    )

    print(
        f"  Z : "
        f"{domain['zmin_m']} "
        f"-> "
        f"{domain['zmax_m']} m"
    )

    print("")

    print(
        "Outer PLC tessellation"
    )

    print(
        f"  x surface points : "
        f"{boundary_metadata['x_points']}"
    )

    print(
        f"  y surface points : "
        f"{boundary_metadata['y_points']}"
    )

    print(
        f"  z surface points : "
        f"{boundary_metadata['z_points']}"
    )

    print(
        f"  boundary facets  : "
        f"{boundary_metadata['facet_count']:,}"
    )

    print("")

    print("Injector")

    print(
        f"  center : "
        f"({cx}, {cy}, {cz}) m"
    )

    print(
        f"  radius : "
        f"{radius} m"
    )

    print(
        f"  height : "
        f"{height} m"
    )

    print(
        f"  volume : "
        f"{injector_volume:.12e} m^3"
    )

    print("")

    print("Mesh input points")

    print(
        f"  total          : "
        f"{len(points):,}"
    )

    print(
        f"  coarse lattice : "
        f"{coarse_added:,}"
    )

    print(
        f"  local shells   : "
        f"{sum(v for k, v in shell_stats.items() if not k.endswith('_axial_mode')):,}"
    )

    print("")

    print("PLC facets")

    print(
        f"  total : "
        f"{len(facets):,}"
    )

    if shell_stats:

        print("")

        print(
            "Injector shell points"
        )

        for key, count in shell_stats.items():

            if key.endswith(
                "_axial_mode"
            ):
                continue

            print(
                f"  {key:>8s} : "
                f"{count:,}"
            )

        print("")

        print(
            "Inner-shell alignment"
        )

        print(
            f"  shells aligned : "
            f"{shell_metadata['inner_aligned_shell_count']}"
        )

        print(
            f"  common axial spacing : "
            f"{shell_metadata['inner_common_axial_spacing_m']} m"
        )

    print("")

    print(
        f"POLY      : {poly_path}"
    )

    print(
        f"Geometry  : {geometry_path}"
    )

    print("")

    print(
        "The previous M1 output files are preserved."
    )

    print("")


if __name__ == "__main__":
    main()
