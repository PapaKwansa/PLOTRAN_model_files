#!/usr/bin/env python3
"""
Export PFLOTRAN geomechanics time series for strainmeters and other vset regions.

This is a generalized replacement for ``pflotran_strainmeter_timeseries.py``.
Every supplied ``--region NAME=FILE.vset`` is treated identically, so the same
run can process the three strainmeters, the injection interval, and the HEC.
The historical ``--sensor`` spelling remains available as an alias.

For every region and PFLOTRAN output time, the script computes spatial
statistics over all unique vset nodes and writes:

* one combined wide CSV and one CSV per region;
* one long-form strain-component CSV;
* one compact VTU per time and one PVD time-series index for ParaView;
* publication-quality strain figures per region in the requested display units;
* optional displacement plots per region;
* high-quality all-region strain comparison in nanostrain;
* cell-centered injector pressure and pressure-change figures when a flow
  HDF5 and validated flow-to-mechanics mapping are supplied;
* comparison plots showing one strain component across all regions;
* a JSON manifest describing inputs, aggregation, plots, and optional axes.

The six tensor components plotted together are:

    strain_xx, strain_yy, strain_zz,
    strain_xy, strain_yz, strain_zx

The HEC and injection deformation outputs are *spatial summaries of regions*, not point
measurements. Injector pressure is handled separately as a cell-centered flow quantity. Their mean can hide sign changes inside the region, so the script
also records standard deviation, minimum, and maximum and can draw a spread
band around each mean curve.

Optional axial strain
---------------------
A physical strainmeter generally measures strain along its installed unit axis
``n``. Supply a direction using repeated ``--axis`` arguments, for example:

    --axis AVN2=1,0,0
    --axis AVN87=0,1,0
    --axis AVN31=0,0,1

At every node, the script evaluates

    eps_axial = nx^2 eps_xx + ny^2 eps_yy + nz^2 eps_zz
                + 2 nx ny eps_xy + 2 ny nz eps_yz + 2 nz nx eps_zx

and then averages that axial strain over the region. Do not call a global
component such as ``strain_xx`` the gauge reading unless the instrument axis is
actually aligned with global x.

Requirements
------------

    numpy, h5py, meshio, matplotlib

Example: North Avant V5
-----------------------

    python3 pflotran_region_timeseries_plots.py \
      north_avant_v5_oneway_injection_smoke-geomech.h5 \
      --output-dir region_timeseries_v5 \
      --region AVN2=AVN2.vset \
      --region AVN87=AVN87.vset \
      --region AVN31=AVN31.vset \
      --region Injection=injection_borehole.vset \
      --region HEC=hec.vset \
      --plot-spread std \
      --strain-unit dimensionless

Open ``region_timeseries_v5/region_timeseries.pvd`` in ParaView for the compact
five-point time series. The automatically generated PNG/PDF figures are under
``region_timeseries_v5/plots``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

# Use a non-interactive backend so plotting is reliable on WSL and clusters.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

try:
    import meshio
except ImportError as exc:
    raise SystemExit(
        "meshio is required. Install it with:\n"
        "  python3 -m pip install --user meshio\n"
        "or:\n"
        "  conda install -c conda-forge meshio"
    ) from exc


TIME_PATTERN = re.compile(
    r"Time\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"\s*([A-Za-z]*)",
    re.IGNORECASE,
)

STRAIN_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("strain_xx", r"$\varepsilon_{xx}$"),
    ("strain_yy", r"$\varepsilon_{yy}$"),
    ("strain_zz", r"$\varepsilon_{zz}$"),
    ("strain_xy", r"$\varepsilon_{xy}$"),
    ("strain_yz", r"$\varepsilon_{yz}$"),
    ("strain_zx", r"$\varepsilon_{zx}$"),
)

DISPLACEMENT_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("displacement_x", r"$u_x$"),
    ("displacement_y", r"$u_y$"),
    ("displacement_z", r"$u_z$"),
)

STRAIN_UNIT_OPTIONS: Mapping[str, tuple[float, str]] = {
    "dimensionless": (1.0, "Strain [dimensionless]"),
    "microstrain": (1.0e6, r"Strain [$\mu\varepsilon$]"),
    "nanostrain": (1.0e9, r"Strain [n$\varepsilon$]"),
}

PLOT_RC = {
    "font.size": 11.0,
    "axes.titlesize": 14.0,
    "axes.labelsize": 12.0,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

LINE_STYLES: tuple[object, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1.5)),
    (0, (3, 1, 1, 1)),
)

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")


@dataclass(frozen=True)
class RegionDefinition:
    name: str
    vset_path: Path
    indices: np.ndarray
    center_xyz: np.ndarray


@dataclass(frozen=True)
class TimeGroup:
    value: float
    unit: str
    hdf5_path: str


def safe_name(name: str) -> str:
    """Convert a field or region name to a portable output name."""
    value = name.strip().replace("[", "").replace("]", "")
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def normalized_name(name: str) -> str:
    return safe_name(name).lower()


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Expected NAME=PATH, for example AVN2=AVN2.vset"
        )
    name, path = text.split("=", 1)
    name = safe_name(name)
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("NAME and PATH must both be nonempty")
    return name, Path(path)


def parse_axis(text: str) -> tuple[str, np.ndarray]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Expected NAME=nx,ny,nz, for example AVN2=1,0,0"
        )

    name, values_text = text.split("=", 1)
    values = values_text.split(",")
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "Axis must have three comma-separated components"
        )

    try:
        vector = np.asarray([float(value) for value in values], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid axis: {text}") from exc

    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise argparse.ArgumentTypeError("Axis must have nonzero finite length")

    return safe_name(name), vector / norm


def read_vset(path: Path, node_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    ids: set[int] = set()
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        node_id = int(value)
        if node_id < 1 or node_id > node_count:
            raise RuntimeError(
                f"{path}: node ID {node_id} is outside 1..{node_count}"
            )
        ids.add(node_id)

    if not ids:
        raise RuntimeError(f"{path}: no node IDs found")

    return np.asarray(sorted(value - 1 for value in ids), dtype=np.int64)


def parse_time(group_path: str) -> tuple[float, str]:
    match = TIME_PATTERN.search(group_path)
    if not match:
        raise ValueError(group_path)
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    unit = match.group(2) or ""
    return value, unit


def compatible_nodal_array(
    dataset: h5py.Dataset,
    node_count: int,
) -> np.ndarray | None:
    shape = tuple(int(value) for value in dataset.shape)
    if shape == (node_count,):
        values = np.asarray(dataset[...])
    elif shape == (node_count, 1):
        values = np.asarray(dataset[:, 0])
    else:
        return None

    if not (
        np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        return None

    if np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Dataset {dataset.name!r} contains NaN or infinity"
            )

    return values


def read_geomechanics_coordinates(
    h5: h5py.File,
) -> np.ndarray:
    """
    Read authoritative PFLOTRAN node coordinates.

    Prefer /Domain/X, /Domain/Y, and /Domain/Z because the current
    geomechanics HDF5 writer can leave /Domain/Vertices zero-filled.
    """
    required = ("/Domain/X", "/Domain/Y", "/Domain/Z")

    if all(path in h5 for path in required):
        x = np.asarray(
            h5["/Domain/X"][...],
            dtype=np.float64,
        ).reshape(-1)

        y = np.asarray(
            h5["/Domain/Y"][...],
            dtype=np.float64,
        ).reshape(-1)

        z = np.asarray(
            h5["/Domain/Z"][...],
            dtype=np.float64,
        ).reshape(-1)

        if not (x.size == y.size == z.size):
            raise RuntimeError(
                "PFLOTRAN coordinate arrays have inconsistent lengths: "
                f"X={x.size:,}, Y={y.size:,}, Z={z.size:,}"
            )

        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(z))
        ):
            raise RuntimeError(
                "PFLOTRAN /Domain/X/Y/Z contains NaN or infinity"
            )

        points = np.column_stack((x, y, z))

        if not np.any(np.abs(points) > 0.0):
            raise RuntimeError(
                "PFLOTRAN /Domain/X/Y/Z is zero-filled"
            )

        if "/Domain/Vertices" in h5:
            vertices = np.asarray(
                h5["/Domain/Vertices"][...],
                dtype=np.float64,
            )

            if (
                vertices.shape == points.shape
                and np.allclose(
                    vertices,
                    0.0,
                    atol=0.0,
                    rtol=0.0,
                )
            ):
                print(
                    "PFLOTRAN geometry: /Domain/Vertices is zero-filled; "
                    "using /Domain/X/Y/Z."
                )
            else:
                print(
                    "PFLOTRAN geometry: using /Domain/X/Y/Z."
                )
        else:
            print(
                "PFLOTRAN geometry: using /Domain/X/Y/Z."
            )

        return points

    if "/Domain/Vertices" not in h5:
        raise RuntimeError(
            "Geomechanics HDF5 contains neither /Domain/X,/Y,/Z "
            "nor /Domain/Vertices"
        )

    vertices = np.asarray(
        h5["/Domain/Vertices"][...],
        dtype=np.float64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(
            f"Unexpected /Domain/Vertices shape: {vertices.shape}"
        )

    if not np.all(np.isfinite(vertices)):
        raise RuntimeError(
            "PFLOTRAN /Domain/Vertices contains NaN or infinity"
        )

    if not np.any(np.abs(vertices) > 0.0):
        raise RuntimeError(
            "PFLOTRAN /Domain/Vertices is zero-filled and "
            "/Domain/X/Y/Z are unavailable"
        )

    print(
        "PFLOTRAN geometry: using /Domain/Vertices "
        "(validated fallback)."
    )

    return vertices


def discover_time_groups(
    h5: h5py.File,
    node_count: int,
) -> list[TimeGroup]:
    groups: list[TimeGroup] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return
        if "time" not in name.lower():
            return

        has_nodal_data = any(
            isinstance(item, h5py.Dataset)
            and tuple(int(value) for value in item.shape)
            in {(node_count,), (node_count, 1)}
            for item in obj.values()
        )
        if not has_nodal_data:
            return

        try:
            value, unit = parse_time(name)
        except ValueError:
            return

        groups.append(TimeGroup(value=value, unit=unit, hdf5_path="/" + name))

    h5.visititems(visitor)

    # Preserve unique HDF5 groups while sorting chronologically.
    unique = {
        (item.value, item.hdf5_path): item
        for item in groups
    }
    return sorted(unique.values(), key=lambda item: (item.value, item.hdf5_path))


def load_group_arrays(
    group: h5py.Group,
    node_count: int,
    prefix: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load numeric nodal/cell arrays from an HDF5 time group."""
    arrays: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}
    for dataset_name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        values = compatible_nodal_array(dataset, node_count)
        if values is None:
            continue
        output_name = prefix + safe_name(dataset_name)
        base_name = output_name
        suffix = 2
        while output_name in arrays:
            output_name = f"{base_name}_{suffix}"
            suffix += 1
        arrays[output_name] = values
        names[dataset_name] = output_name
    return arrays, names


def infer_dataset_count(h5: h5py.File) -> int:
    """Infer the dominant leading dimension of numeric HDF5 datasets."""
    counts: list[int] = []

    def visitor(_name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim not in {1, 2} or not obj.shape:
            return
        leading = int(obj.shape[0])
        if leading > 1:
            counts.append(leading)

    h5.visititems(visitor)
    if not counts:
        raise RuntimeError("Could not infer a numeric dataset size from HDF5")
    unique, frequencies = np.unique(np.asarray(counts, dtype=np.int64), return_counts=True)
    return int(unique[int(np.argmax(frequencies))])


def read_mapping(
    path: Path,
    flow_count: int,
    mechanics_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(f"{path}: expected at least two mapping columns")
    flow_ids = data[:, 0]
    mechanics_ids = data[:, 1]
    if data.shape[0] != mechanics_count:
        raise RuntimeError(
            f"{path}: expected {mechanics_count:,} mapping rows, "
            f"found {data.shape[0]:,}"
        )
    if len(np.unique(flow_ids)) != len(flow_ids):
        raise RuntimeError(f"{path}: duplicate flow IDs")
    if len(np.unique(mechanics_ids)) != len(mechanics_ids):
        raise RuntimeError(f"{path}: duplicate mechanics IDs")
    if flow_ids.min() < 1 or flow_ids.max() > flow_count:
        raise RuntimeError(f"{path}: flow IDs outside 1..{flow_count}")
    if mechanics_ids.min() < 1 or mechanics_ids.max() > mechanics_count:
        raise RuntimeError(f"{path}: mechanics IDs outside 1..{mechanics_count}")
    return flow_ids - 1, mechanics_ids - 1


def map_flow_array(
    values: np.ndarray,
    flow_ids: np.ndarray,
    mechanics_ids: np.ndarray,
    mechanics_count: int,
) -> np.ndarray:
    mapped = np.empty(mechanics_count, dtype=values.dtype)
    assigned = np.zeros(mechanics_count, dtype=bool)
    mapped[mechanics_ids] = values[flow_ids]
    assigned[mechanics_ids] = True
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned)[:10] + 1
        raise RuntimeError(
            f"Mapping does not assign mechanics vertices {missing.tolist()}"
        )
    return mapped


def read_flow_scalar_array(
    group: h5py.Group,
    flow_cell_count: int,
    requested: str,
) -> tuple[np.ndarray, str] | tuple[None, None]:
    """Read a numeric PFLOTRAN flow field containing one value per cell."""
    target = normalized_name(requested)
    matches: list[tuple[str, np.ndarray]] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        shape = tuple(int(value) for value in obj.shape)
        if shape == (flow_cell_count,):
            values = np.asarray(obj[...])
        elif shape == (flow_cell_count, 1):
            values = np.asarray(obj[:, 0])
        else:
            return
        if not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            return
        if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
            raise RuntimeError(f"Dataset {obj.name!r} contains NaN or infinity")
        leaf = normalized_name(Path(name).name)
        if leaf == target or leaf.startswith(target + "_") or target in leaf:
            matches.append(
                (obj.name, np.asarray(values, dtype=np.float64).reshape(-1))
            )

    group.visititems(visitor)
    if not matches:
        return None, None
    exact = [item for item in matches if normalized_name(Path(item[0]).name) == target]
    name, values = exact[0] if exact else matches[0]
    return values, name





def find_flow_field(arrays: Mapping[str, np.ndarray], requested: str) -> np.ndarray | None:
    """Find a flow field by base name despite unit suffixes and Flow_ prefix."""
    target = normalized_name(requested)
    for name, values in arrays.items():
        normalized = normalized_name(name)
        candidates = {normalized, normalized.removeprefix("flow_")}
        for candidate in candidates:
            if candidate == target or candidate.startswith(target + "_"):
                return np.asarray(values)
    return None


def resolve_pressure_field(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray | None, str | None]:
    """Find the liquid-pressure field used for the injector pressure plot.

    Keep this intentionally strict: the injector diagnostic should use the
    PFLOTRAN liquid-pressure field rather than a generic dataset whose name
    merely contains ``Pressure``.
    """
    candidates = (
        "Flow_Liquid_Pressure",
        "Liquid_Pressure",
    )
    for requested in candidates:
        values = find_flow_field(arrays, requested)
        if values is not None:
            for name in arrays:
                if np.shares_memory(np.asarray(arrays[name]), np.asarray(values)):
                    return np.asarray(values), name
            return np.asarray(values), requested
    return None, None


def tensor_strain_norm_values(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return Frobenius norm of the symmetric small-strain tensor."""
    components = {
        name: find_array(arrays, name)
        for name, _ in STRAIN_COMPONENTS
    }
    missing = [name for name, value in components.items() if value is None]
    if missing:
        raise RuntimeError(
            "Cannot calculate strain-tensor norm; missing components: "
            + ", ".join(missing)
        )
    exx = np.asarray(components["strain_xx"], dtype=float)
    eyy = np.asarray(components["strain_yy"], dtype=float)
    ezz = np.asarray(components["strain_zz"], dtype=float)
    exy = np.asarray(components["strain_xy"], dtype=float)
    eyz = np.asarray(components["strain_yz"], dtype=float)
    ezx = np.asarray(components["strain_zx"], dtype=float)
    return np.sqrt(
        exx * exx
        + eyy * eyy
        + ezz * ezz
        + 2.0 * (exy * exy + eyz * eyz + ezx * ezx)
    )


def plot_region_pressure(
    region: RegionDefinition,
    times: np.ndarray,
    pressure: Mapping[str, Sequence[float]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    cell_count: int,
    center_cell_id: int,
    baseline_time: float,
) -> list[str]:
    """Plot cell-centered injector liquid-pressure change and absolute pressure."""
    pressure_mean = np.asarray(pressure["mean"], dtype=float)
    pressure_median = np.asarray(pressure["median"], dtype=float)
    pressure_p05 = np.asarray(pressure["p05"], dtype=float)
    pressure_p95 = np.asarray(pressure["p95"], dtype=float)
    pressure_center = np.asarray(pressure["center"], dtype=float)
    pressure_max = np.asarray(pressure["max"], dtype=float)

    mean_change = np.asarray(pressure["change_mean"], dtype=float) / 1.0e6
    median_change = np.asarray(pressure["change_median"], dtype=float) / 1.0e6
    p05_change = np.asarray(pressure["change_p05"], dtype=float) / 1.0e6
    p95_change = np.asarray(pressure["change_p95"], dtype=float) / 1.0e6
    center_change = np.asarray(pressure["change_center"], dtype=float) / 1.0e6
    min_change = np.asarray(pressure["change_min"], dtype=float) / 1.0e6
    max_change = np.asarray(pressure["change_max"], dtype=float) / 1.0e6

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
        ax.fill_between(
            times, p05_change, p95_change, alpha=0.16,
            label="5–95% injector-cell range",
        )
        markevery = max(1, len(times) // 10)
        ax.plot(times, mean_change, marker="o", markevery=markevery,
                linewidth=2.6, label="mean Δp")
        ax.plot(times, median_change, linestyle="--", marker="s",
                markevery=markevery, linewidth=1.8, label="median Δp")
        ax.plot(times, center_change, linestyle=":", linewidth=2.2,
                label=f"representative-cell Δp (cell {center_cell_id})")
        ax.plot(times, min_change, linestyle=(0, (2, 2)), linewidth=1.0,
                alpha=0.7, label="minimum Δp")
        ax.plot(times, max_change, linestyle="-.", linewidth=1.2, alpha=0.8,
                label="maximum Δp")
        ax.axhline(0.0, linewidth=0.9, alpha=0.65)
        ax.axvline(19.0, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(19.0, 0.98, "Injection stops", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=9)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Pore-pressure change, Δp [MPa]")
        ax.set_title(
            f"{region.name}: cell-centered pore-pressure response\n"
            f"{cell_count:,} injector cells; baseline = first flow snapshot at {baseline_time:g} h"
        )
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.legend(loc="best", frameon=True)
        outputs = save_figure(fig, output_base, formats, dpi)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
        markevery = max(1, len(times) // 10)
        ax.plot(times, pressure_mean / 1.0e6, marker="o", markevery=markevery,
                linewidth=2.3, label="mean liquid pressure")
        ax.plot(times, pressure_center / 1.0e6, linestyle=":", linewidth=2.0,
                label=f"representative-cell pressure (cell {center_cell_id})")
        ax.plot(times, pressure_max / 1.0e6, linestyle="-.", linewidth=1.0,
                alpha=0.7, label="maximum liquid pressure")
        ax.axvline(19.0, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(19.0, 0.98, "Injection stops", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=9)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Liquid pressure [MPa]")
        ax.set_title(
            f"{region.name}: cell-centered liquid pressure\n"
            f"{cell_count:,} injector cells; representative cell {center_cell_id} "
            "selected by median baseline-pressure ordering"
        )
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.legend(loc="best", frameon=True)
        outputs.extend(
            save_figure(
                fig, output_base.with_name(output_base.name + "_absolute"),
                formats, dpi,
            )
        )

    return outputs


def plot_region_strain_norm_across_regions(
    regions: Sequence[RegionDefinition],
    times: np.ndarray,
    all_series: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[str]:
    """Compare scalar strain-tensor magnitude across all requested regions."""
    scale = STRAIN_UNIT_OPTIONS["nanostrain"][0]
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
        markevery = max(1, len(times) // 10)
        plotted = 0
        for index, region in enumerate(regions):
            field = all_series[region.name].get("strain_tensor_norm")
            if field is None:
                continue
            mean = np.asarray(field["mean"], dtype=float) * scale
            ax.plot(
                times,
                mean,
                label=region.name,
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                marker=MARKERS[index % len(MARKERS)],
                markevery=markevery,
                linewidth=2.2,
            )
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            return []
        ax.axhline(0.0, linewidth=0.8, alpha=0.6)
        ax.axvline(19.0, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(19.0, 0.98, "Injection stops", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=9)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(r"Strain tensor norm [n$\varepsilon$]")
        ax.set_title("Regional strain-response comparison\nFrobenius norm of the small-strain tensor")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        return save_figure(fig, output_base, formats, dpi)


def find_array(
    arrays: Mapping[str, np.ndarray],
    requested: str,
) -> np.ndarray | None:
    target = normalized_name(requested)
    for name, values in arrays.items():
        normalized = normalized_name(name)
        if normalized == target or normalized.startswith(target + "_"):
            return np.asarray(values)
    return None


def find_series_field(
    series: Mapping[str, Mapping[str, Sequence[float]]],
    requested: str,
) -> Mapping[str, Sequence[float]] | None:
    """Find a stored time-series field while allowing unit suffixes."""
    target = normalized_name(requested)
    for name, values in series.items():
        normalized = normalized_name(name)
        if normalized == target or normalized.startswith(target + "_"):
            return values
    return None


def mode_value(values: np.ndarray) -> float:
    flattened = [int(value) for value in np.asarray(values).ravel()]
    return float(Counter(flattened).most_common(1)[0][0])


def scalar_statistics(
    values: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(values)[indices]

    if (
        np.issubdtype(selected.dtype, np.integer)
        or np.issubdtype(selected.dtype, np.bool_)
    ):
        mode = mode_value(selected)
        return {
            "mean": mode,
            "std": 0.0,
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
        }

    selected_float = np.asarray(selected, dtype=np.float64)
    return {
        "mean": float(np.mean(selected_float, dtype=np.float64)),
        "std": float(np.std(selected_float, dtype=np.float64)),
        "min": float(np.min(selected_float)),
        "max": float(np.max(selected_float)),
    }


def axial_strain_values(
    arrays: Mapping[str, np.ndarray],
    axis: np.ndarray,
) -> np.ndarray:
    components = {
        name: find_array(arrays, name)
        for name, _ in STRAIN_COMPONENTS
    }
    missing = [name for name, value in components.items() if value is None]
    if missing:
        raise RuntimeError(
            "Cannot calculate axial strain; missing components: "
            + ", ".join(missing)
        )

    exx = np.asarray(components["strain_xx"], dtype=float)
    eyy = np.asarray(components["strain_yy"], dtype=float)
    ezz = np.asarray(components["strain_zz"], dtype=float)
    exy = np.asarray(components["strain_xy"], dtype=float)
    eyz = np.asarray(components["strain_yz"], dtype=float)
    ezx = np.asarray(components["strain_zx"], dtype=float)

    nx, ny, nz = axis
    return (
        nx * nx * exx
        + ny * ny * eyy
        + nz * nz * ezz
        + 2.0 * nx * ny * exy
        + 2.0 * ny * nz * eyz
        + 2.0 * nz * nx * ezx
    )


def write_pvd(path: Path, records: Sequence[tuple[float, Path]]) -> None:
    vtkfile = ET.Element(
        "VTKFile",
        {
            "type": "Collection",
            "version": "0.1",
            "byte_order": "LittleEndian",
        },
    )
    collection = ET.SubElement(vtkfile, "Collection")

    for time_value, vtu_path in records:
        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": f"{time_value:.16g}",
                "group": "",
                "part": "0",
                "file": vtu_path.name,
            },
        )

    tree = ET.ElementTree(vtkfile)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


def scientific_axis(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    for extension in formats:
        output = base_path.with_suffix("." + extension)
        if extension.lower() == "png":
            fig.savefig(output, dpi=dpi)
        else:
            fig.savefig(output)
        outputs.append(str(output))

    plt.close(fig)
    return outputs


def plot_region_strains(
    region: RegionDefinition,
    times: np.ndarray,
    time_unit: str,
    series: Mapping[str, Mapping[str, Sequence[float]]],
    axis_series: Mapping[str, Sequence[float]] | None,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    strain_unit: str,
    spread_mode: str,
    include_volumetric: bool,
    symmetric_y: bool,
) -> list[str]:
    scale, y_label = STRAIN_UNIT_OPTIONS[strain_unit]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)

        y_extent_values: list[np.ndarray] = []

        for index, (component, label) in enumerate(STRAIN_COMPONENTS):
            values = find_series_field(series, component)
            if values is None:
                continue

            mean = np.asarray(values["mean"], dtype=float) * scale
            std = np.asarray(values["std"], dtype=float) * scale
            minimum = np.asarray(values["min"], dtype=float) * scale
            maximum = np.asarray(values["max"], dtype=float) * scale

            line, = ax.plot(
                times,
                mean,
                label=label,
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(times) // 10),
            )

            if spread_mode == "std":
                lower = mean - std
                upper = mean + std
                ax.fill_between(
                    times,
                    lower,
                    upper,
                    color=line.get_color(),
                    alpha=0.10,
                    linewidth=0.0,
                )
                y_extent_values.extend([lower, upper])
            elif spread_mode == "minmax":
                ax.fill_between(
                    times,
                    minimum,
                    maximum,
                    color=line.get_color(),
                    alpha=0.08,
                    linewidth=0.0,
                )
                y_extent_values.extend([minimum, maximum])

            y_extent_values.append(mean)

        volumetric_series = find_series_field(series, "volumetric_strain")
        if include_volumetric and volumetric_series is not None:
            values = volumetric_series
            mean = np.asarray(values["mean"], dtype=float) * scale
            ax.plot(
                times,
                mean,
                label=r"$\varepsilon_v$",
                linewidth=2.4,
                marker="X",
            )
            y_extent_values.append(mean)

        if axis_series is not None:
            mean = np.asarray(axis_series["mean"], dtype=float) * scale
            ax.plot(
                times,
                mean,
                label=r"$\varepsilon_{\mathrm{axial}}$",
                linewidth=2.6,
                marker="*",
            )
            y_extent_values.append(mean)

        ax.axhline(0.0, linewidth=0.8, alpha=0.65)
        ax.set_xlabel(f"Time [{time_unit}]" if time_unit else "Time")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{region.name}: strain tensor time series\n"
            f"mean over {region.indices.size:,} vset nodes; "
            f"center = ({region.center_xyz[0]:.2f}, "
            f"{region.center_xyz[1]:.2f}, {region.center_xyz[2]:.2f}) m"
        )
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        scientific_axis(ax)

        if symmetric_y and y_extent_values:
            concatenated = np.concatenate(
                [np.ravel(value) for value in y_extent_values]
            )
            finite = concatenated[np.isfinite(concatenated)]
            if finite.size:
                limit = float(np.max(np.abs(finite)))
                if limit > 0.0:
                    ax.set_ylim(-1.08 * limit, 1.08 * limit)

        spread_text = {
            "none": "mean curves",
            "std": "mean curves; shading = mean ± 1 standard deviation",
            "minmax": "mean curves; shading = nodal minimum–maximum",
        }[spread_mode]
        ax.text(
            0.01,
            0.015,
            spread_text,
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

        return save_figure(fig, output_base, formats, dpi)


def plot_region_displacement(
    region: RegionDefinition,
    times: np.ndarray,
    time_unit: str,
    series: Mapping[str, Mapping[str, Sequence[float]]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[str]:
    component_series = {
        name: find_series_field(series, name)
        for name, _ in DISPLACEMENT_COMPONENTS
    }
    if any(value is None for value in component_series.values()):
        return []

    ux = np.asarray(component_series["displacement_x"]["mean"], dtype=float)
    uy = np.asarray(component_series["displacement_y"]["mean"], dtype=float)
    uz = np.asarray(component_series["displacement_z"]["mean"], dtype=float)
    magnitude = np.sqrt(ux * ux + uy * uy + uz * uz)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)

        for index, (values, label) in enumerate(
            (
                (ux, r"$u_x$"),
                (uy, r"$u_y$"),
                (uz, r"$u_z$"),
                (magnitude, r"$|\mathbf{u}|$"),
            )
        ):
            ax.plot(
                times,
                values,
                label=label,
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(times) // 10),
            )

        ax.axhline(0.0, linewidth=0.8, alpha=0.65)
        ax.set_xlabel(f"Time [{time_unit}]" if time_unit else "Time")
        ax.set_ylabel("Displacement [m]")
        ax.set_title(
            f"{region.name}: displacement time series\n"
            f"mean over {region.indices.size:,} vset nodes"
        )
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        scientific_axis(ax)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

        return save_figure(fig, output_base, formats, dpi)


def plot_component_across_regions(
    component: str,
    label: str,
    regions: Sequence[RegionDefinition],
    times: np.ndarray,
    time_unit: str,
    all_series: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
    strain_unit: str,
) -> list[str]:
    scale, y_label = STRAIN_UNIT_OPTIONS[strain_unit]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)

        for index, region in enumerate(regions):
            region_series = all_series[region.name]
            component_series = find_series_field(region_series, component)
            if component_series is None:
                continue
            mean = np.asarray(component_series["mean"], dtype=float)
            ax.plot(
                times,
                mean * scale,
                label=region.name,
                linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(times) // 10),
            )

        ax.axhline(0.0, linewidth=0.8, alpha=0.65)
        ax.set_xlabel(f"Time [{time_unit}]" if time_unit else "Time")
        ax.set_ylabel(y_label)
        ax.set_title(f"{label} across requested regions")
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        scientific_axis(ax)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

        return save_figure(fig, output_base, formats, dpi)


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    preferred_columns: Sequence[str],
) -> None:
    columns: list[str] = []
    seen: set[str] = set()

    for column in preferred_columns:
        if column not in seen:
            seen.add(column)
            columns.append(column)

    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export and plot PFLOTRAN strain/displacement time series for "
            "strainmeters, the injector, the HEC, or any other vset region."
        )
    )
    parser.add_argument("geomech_h5", type=Path)
    parser.add_argument(
        "--region",
        "--sensor",
        dest="regions",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=VSET",
        help=(
            "Region name and vset. Repeat for AVN2, AVN87, AVN31, Injection, "
            "HEC, etc. --sensor is retained as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        type=parse_axis,
        metavar="NAME=NX,NY,NZ",
        help="Optional instrument axis for axial strain; repeat as needed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("region_timeseries"),
    )
    parser.add_argument(
        "--include-spread",
        action="store_true",
        help=(
            "Include standard deviation, minimum, and maximum for all nodal "
            "fields in the wide CSV. Strain spread is always written."
        ),
    )
    parser.add_argument(
        "--plot-spread",
        choices=("none", "std", "minmax"),
        default="std",
        help="Spread band on per-region strain plots (default: std)",
    )
    parser.add_argument(
        "--strain-unit",
        choices=tuple(STRAIN_UNIT_OPTIONS),
        default="dimensionless",
        help="Display unit for plots only; CSV strain remains dimensionless",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Figure formats (default: png pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution (default: 600 dpi)",
    )
    parser.add_argument(
        "--flow-h5",
        type=Path,
        default=None,
        help=(
            "Optional PFLOTRAN flow HDF5 used to generate regional liquid-pressure "
            "and pressure-change plots."
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Validated flow-to-mechanics mapping. For injector pressure, the "
            "injection vset is interpreted in mechanics-vertex numbering and "
            "mapped to the corresponding flow-cell IDs."
        ),
    )
    parser.add_argument(
        "--pressure-region",
        default="Injection",
        help="Region used for the pressure plots (default: Injection)",
    )
    parser.add_argument(
        "--no-pressure-plots",
        action="store_true",
        help="Disable pressure plots even when --flow-h5 is supplied.",
    )
    parser.add_argument(
        "--include-volumetric-in-strain-plot",
        action="store_true",
        help="Add volumetric strain to each six-component plot",
    )
    parser.add_argument(
        "--symmetric-strain-axis",
        action="store_true",
        help="Force each per-region strain y axis to be symmetric about zero",
    )
    parser.add_argument(
        "--no-displacement-plots",
        action="store_true",
        help="Do not create one displacement-component plot per region",
    )
    parser.add_argument(
        "--no-component-comparison-plots",
        action="store_true",
        help="Do not create one all-region figure for each strain component",
    )
    parser.add_argument(
        "--no-regional-strain-norm-plot",
        action="store_true",
        help="Do not create the all-region strain-tensor-norm nanostrain figure.",
    )
    parser.add_argument(
        "--allow-missing-strain-components",
        action="store_true",
        help="Plot available components instead of failing when one is absent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    h5_path = args.geomech_h5.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    region_specs = [
        (name, path.expanduser().resolve())
        for name, path in args.regions
    ]
    region_names = [name for name, _ in region_specs]
    if len(set(region_names)) != len(region_names):
        raise RuntimeError("Region names must be unique")

    axes = dict(args.axis)
    unknown_axes = sorted(set(axes) - set(region_names))
    if unknown_axes:
        raise RuntimeError(
            "Axes supplied for unknown regions: " + ", ".join(unknown_axes)
        )

    if args.flow_h5 is None and args.mapping is not None:
        raise RuntimeError(
            "--mapping is only accepted when --flow-h5 is also supplied"
        )

    if args.flow_h5 is not None and args.mapping is None and not args.no_pressure_plots:
        raise RuntimeError(
            "Injector pressure diagnostics require both --flow-h5 and --mapping"
        )

    if args.flow_h5 is not None and not args.flow_h5.expanduser().is_file():
        raise FileNotFoundError(args.flow_h5)
    if args.mapping is not None and not args.mapping.expanduser().is_file():
        raise FileNotFoundError(args.mapping)

    all_rows: list[dict[str, object]] = []
    strain_long_rows: list[dict[str, object]] = []
    pvd_records: list[tuple[float, Path]] = []
    field_name_map: dict[str, str] = {}
    plot_outputs: list[str] = []

    # region -> field -> statistic -> values ordered by time
    all_series: dict[
        str,
        dict[str, dict[str, list[float]]],
    ] = {}
    axial_series: dict[str, dict[str, list[float]]] = {}
    pressure_series: dict[str, dict[str, list[float]]] = {}
    pressure_long_rows: list[dict[str, object]] = []

    with h5py.File(h5_path, "r") as h5:
        vertices = read_geomechanics_coordinates(h5)
        node_count = int(vertices.shape[0])
        regions: list[RegionDefinition] = []

        for name, path in region_specs:
            indices = read_vset(path, node_count)
            center = np.mean(vertices[indices], axis=0)
            regions.append(
                RegionDefinition(
                    name=name,
                    vset_path=path,
                    indices=indices,
                    center_xyz=center,
                )
            )
            all_series[name] = {}
            axial_series[name] = {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
            }
            pressure_series[name] = {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "change_mean": [],
                "change_std": [],
                "change_min": [],
                "change_max": [],
                "times": [],
            }

        time_groups = discover_time_groups(h5, node_count)
        if not time_groups:
            raise RuntimeError("No compatible time groups found")

        units = {item.unit for item in time_groups}
        if len(units) > 1:
            raise RuntimeError(
                "Time groups use inconsistent units: " + ", ".join(sorted(units))
            )
        time_unit = next(iter(units)) if units else ""
        times = np.asarray([item.value for item in time_groups], dtype=float)

        print(f"Geomechanics HDF5: {h5_path}")
        print(f"Mesh vertices: {node_count:,}")
        print(f"Time groups: {len(time_groups)}")
        print(f"Requested regions: {len(regions)}")

        for region in regions:
            center = region.center_xyz
            print(
                f"  {region.name}: nodes={region.indices.size:,}, "
                f"center=({center[0]:.6f}, {center[1]:.6f}, "
                f"{center[2]:.6f})"
            )
            if region.name in axes:
                print(f"    axis={axes[region.name].tolist()}")

        for time_index, time_group in enumerate(time_groups):
            group = h5[time_group.hdf5_path]
            arrays: dict[str, np.ndarray] = {}

            for dataset_name, dataset in group.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                values = compatible_nodal_array(dataset, node_count)
                if values is None:
                    continue

                output_name = safe_name(dataset_name)
                base_name = output_name
                counter = 2
                while output_name in arrays:
                    output_name = f"{base_name}_{counter}"
                    counter += 1

                arrays[output_name] = values
                field_name_map[dataset_name] = output_name

            missing_strain = [
                name
                for name, _ in STRAIN_COMPONENTS
                if find_array(arrays, name) is None
            ]

            strain_norm = None
            if not missing_strain:
                strain_norm = tensor_strain_norm_values(arrays)
            if missing_strain and not args.allow_missing_strain_components:
                raise RuntimeError(
                    f"Time group {time_group.hdf5_path!r} is missing required "
                    "strain components: " + ", ".join(missing_strain)
                )

            representative_points = np.vstack(
                [region.center_xyz for region in regions]
            )
            point_data: dict[str, np.ndarray] = {
                "Region_ID": np.arange(1, len(regions) + 1, dtype=np.int32),
                "Region_Node_Count": np.asarray(
                    [region.indices.size for region in regions],
                    dtype=np.int32,
                ),
            }

            for region_position, region in enumerate(regions):
                flag = np.zeros(len(regions), dtype=np.uint8)
                flag[region_position] = 1
                point_data[f"{region.name}_Flag"] = flag

            # Regional scalar summaries derived from the six strain components.
            if strain_norm is not None:
                for region in regions:
                    stats = scalar_statistics(strain_norm, region.indices)
                    norm_series = all_series[region.name].setdefault(
                        "strain_tensor_norm",
                        {"mean": [], "std": [], "min": [], "max": []},
                    )
                    for statistic in ("mean", "std", "min", "max"):
                        norm_series[statistic].append(stats[statistic])

            # Compact VTU statistics for every compatible field.
            for field_name, values in arrays.items():
                means: list[float] = []
                stds: list[float] = []
                minima: list[float] = []
                maxima: list[float] = []

                for region in regions:
                    stats = scalar_statistics(values, region.indices)
                    means.append(stats["mean"])
                    stds.append(stats["std"])
                    minima.append(stats["min"])
                    maxima.append(stats["max"])

                    region_field = all_series[region.name].setdefault(
                        field_name,
                        {"mean": [], "std": [], "min": [], "max": []},
                    )
                    for statistic in ("mean", "std", "min", "max"):
                        region_field[statistic].append(stats[statistic])

                point_data[f"Mean_{field_name}"] = np.asarray(means, dtype=float)
                point_data[f"Std_{field_name}"] = np.asarray(stds, dtype=float)
                point_data[f"Min_{field_name}"] = np.asarray(minima, dtype=float)
                point_data[f"Max_{field_name}"] = np.asarray(maxima, dtype=float)

            # True vector outputs from the averaged displacement components.
            ux = find_array(arrays, "displacement_x")
            uy = find_array(arrays, "displacement_y")
            uz = find_array(arrays, "displacement_z")
            if ux is not None and uy is not None and uz is not None:
                mean_displacement = np.asarray(
                    [
                        [
                            float(np.mean(np.asarray(ux)[region.indices])),
                            float(np.mean(np.asarray(uy)[region.indices])),
                            float(np.mean(np.asarray(uz)[region.indices])),
                        ]
                        for region in regions
                    ],
                    dtype=float,
                )
                point_data["Mean_Displacement"] = mean_displacement
                point_data["Mean_Displacement_Magnitude"] = np.linalg.norm(
                    mean_displacement,
                    axis=1,
                )

            axial_point_values = np.full(len(regions), np.nan, dtype=float)
            for region_position, region in enumerate(regions):
                if region.name not in axes:
                    continue
                nodal_axial = axial_strain_values(arrays, axes[region.name])
                stats = scalar_statistics(nodal_axial, region.indices)
                axial_point_values[region_position] = stats["mean"]
                for statistic in ("mean", "std", "min", "max"):
                    axial_series[region.name][statistic].append(stats[statistic])

            if np.any(np.isfinite(axial_point_values)):
                point_data["Mean_Axial_Strain"] = axial_point_values

            # Wide and long CSV rows.
            for region_position, region in enumerate(regions):
                row: dict[str, object] = {
                    "time": time_group.value,
                    "time_unit": time_group.unit,
                    "region": region.name,
                    "region_id": region_position + 1,
                    "node_count": int(region.indices.size),
                    "center_x_m": float(region.center_xyz[0]),
                    "center_y_m": float(region.center_xyz[1]),
                    "center_z_m": float(region.center_xyz[2]),
                }

                for field_name, values in arrays.items():
                    stats = scalar_statistics(values, region.indices)
                    row[f"Mean_{field_name}"] = stats["mean"]

                    is_strain = any(
                        normalized_name(field_name) == normalized_name(name)
                        or normalized_name(field_name).startswith(
                            normalized_name(name) + "_"
                        )
                        for name, _ in STRAIN_COMPONENTS
                    ) or normalized_name(field_name).startswith(
                        normalized_name("volumetric_strain")
                    )

                    if args.include_spread or is_strain:
                        row[f"Std_{field_name}"] = stats["std"]
                        row[f"Min_{field_name}"] = stats["min"]
                        row[f"Max_{field_name}"] = stats["max"]

                if region.name in axes:
                    nodal_axial = axial_strain_values(arrays, axes[region.name])
                    stats = scalar_statistics(nodal_axial, region.indices)
                    for statistic, value in stats.items():
                        row[f"{statistic.capitalize()}_Axial_Strain"] = value

                all_rows.append(row)

                for component, _ in STRAIN_COMPONENTS:
                    values = find_array(arrays, component)
                    if values is None:
                        continue
                    stats = scalar_statistics(values, region.indices)
                    strain_long_rows.append(
                        {
                            "time": time_group.value,
                            "time_unit": time_group.unit,
                            "region": region.name,
                            "region_id": region_position + 1,
                            "node_count": int(region.indices.size),
                            "component": component,
                            "mean_strain": stats["mean"],
                            "std_strain": stats["std"],
                            "min_strain": stats["min"],
                            "max_strain": stats["max"],
                        }
                    )

            cells = [
                (
                    "vertex",
                    np.arange(len(regions), dtype=np.int64).reshape(-1, 1),
                )
            ]
            mesh = meshio.Mesh(
                points=representative_points,
                cells=cells,
                point_data=point_data,
            )
            vtu_path = output_dir / f"regions_{time_index:04d}.vtu"
            try:
                meshio.write(
                    vtu_path,
                    mesh,
                    binary=True,
                    compression="zlib",
                )
            except TypeError:
                meshio.write(vtu_path, mesh, binary=True)

            pvd_records.append((time_group.value, vtu_path))
            print(
                f"  wrote {vtu_path.name} at "
                f"time={time_group.value:g} {time_group.unit}"
            )

    # Optional flow-HDF5 pressure extraction.
    #
    # IMPORTANT: LIQUID_PRESSURE is cell-centered in the PFLOTRAN flow grid.
    # Geomechanics displacement/strain remain vertex-centered and continue to use
    # the geomechanics HDF5 and vset node IDs above. For pressure, the injector
    # vset is a mechanics-node region and must be translated through the validated
    # flow-cell <-> mechanics-vertex mapping before indexing LIQUID_PRESSURE.
    if args.flow_h5 is not None and not args.no_pressure_plots:
        flow_path = args.flow_h5.expanduser().resolve()
        mapping_path = args.mapping.expanduser().resolve()
        pressure_region_name = safe_name(args.pressure_region)
        if pressure_region_name not in {region.name for region in regions}:
            raise RuntimeError(
                f"Pressure region {args.pressure_region!r} was not requested. "
                f"Available regions: {', '.join(region.name for region in regions)}"
            )
        pressure_region_position = next(
            index for index, region in enumerate(regions)
            if region.name == pressure_region_name
        )
        pressure_region = regions[pressure_region_position]

        with h5py.File(flow_path, "r") as flow_h5:
            flow_count = infer_dataset_count(flow_h5)
            flow_groups = discover_time_groups(flow_h5, flow_count)
            if not flow_groups:
                raise RuntimeError(
                    f"{flow_path}: no compatible flow time groups found"
                )

            # The mapping contract is flow-cell ID <-> mechanics-vertex ID.
            # read_vset() returns zero-based mechanics indices, and read_mapping()
            # returns zero-based flow/mechanics indices, so the membership test
            # is performed in the same index space.
            flow_ids, mech_ids = read_mapping(
                mapping_path,
                flow_count,
                node_count,
            )
            injector_mech_nodes = read_vset(
                pressure_region.vset_path,
                node_count,
            )
            injector_flow_cells = flow_ids[
                np.isin(mech_ids, injector_mech_nodes)
            ]

            if injector_flow_cells.size == 0:
                raise RuntimeError(
                    "No flow cells found for injection region via mapping."
                )

            # Remove duplicate flow-cell IDs defensively while preserving the
            # mapping's deterministic ordering.
            injector_flow_cells = np.unique(injector_flow_cells)

            print(
                f"Injector mechanics vset nodes: {injector_mech_nodes.size:,}"
            )
            print(
                f"Injector flow cells via mapping: {injector_flow_cells.size:,}"
            )

            baseline_pressures: np.ndarray | None = None
            baseline_time: float | None = None
            center_cell_index: int | None = None
            center_cell_id: int | None = None
            pressure_samples: list[tuple[float, dict[str, float]]] = []

            for flow_group in flow_groups:
                group = flow_h5[flow_group.hdf5_path]
                liquid_pressure, pressure_name = read_flow_scalar_array(
                    group,
                    flow_count,
                    "LIQUID_PRESSURE",
                )
                if liquid_pressure is None:
                    continue

                liquid_pressure = np.asarray(
                    liquid_pressure,
                    dtype=np.float64,
                ).reshape(-1)
                if liquid_pressure.size != flow_count:
                    raise RuntimeError(
                        f"Pressure field {pressure_name!r} has "
                        f"{liquid_pressure.size:,} values, but the flow field has "
                        f"{flow_count:,} cells."
                    )

                injector_pressures = liquid_pressure[injector_flow_cells]

                if baseline_pressures is None:
                    baseline_pressures = injector_pressures.copy()
                    baseline_time = float(flow_group.value)

                    # Choose the representative cell by the median instantaneous
                    # pressure ordering, matching the requested diagnostic. This
                    # is a pressure-representative cell, not a geometric center.
                    center_index = int(
                        np.argsort(injector_pressures)[
                            len(injector_pressures) // 2
                        ]
                    )
                    center_cell_index = center_index
                    center_cell_id = int(
                        injector_flow_cells[center_index] + 1
                    )

                pressure_change = injector_pressures - baseline_pressures

                stats = {
                    "mean": float(np.mean(injector_pressures, dtype=np.float64)),
                    "median": float(np.median(injector_pressures)),
                    "p05": float(np.percentile(injector_pressures, 5.0)),
                    "p95": float(np.percentile(injector_pressures, 95.0)),
                    "min": float(np.min(injector_pressures)),
                    "max": float(np.max(injector_pressures)),
                    "center": float(injector_pressures[center_cell_index]),
                    "change_mean": float(np.mean(pressure_change, dtype=np.float64)),
                    "change_median": float(np.median(pressure_change)),
                    "change_p05": float(np.percentile(pressure_change, 5.0)),
                    "change_p95": float(np.percentile(pressure_change, 95.0)),
                    "change_min": float(np.min(pressure_change)),
                    "change_max": float(np.max(pressure_change)),
                    "change_center": float(pressure_change[center_cell_index]),
                }
                pressure_samples.append((float(flow_group.value), stats))

                pressure_long_rows.append({
                    "time": float(flow_group.value),
                    "time_unit": flow_group.unit,
                    "region": pressure_region.name,
                    "region_id": pressure_region_position + 1,
                    "mechanics_node_count": int(injector_mech_nodes.size),
                    "flow_cell_count": int(injector_flow_cells.size),
                    "representative_cell_id": center_cell_id,
                    "representative_cell_distance_m": float("nan"),
                    "pressure_mean_Pa": stats["mean"],
                    "pressure_median_Pa": stats["median"],
                    "pressure_p05_Pa": stats["p05"],
                    "pressure_p95_Pa": stats["p95"],
                    "pressure_min_Pa": stats["min"],
                    "pressure_max_Pa": stats["max"],
                    "pressure_representative_Pa": stats["center"],
                    "pressure_change_mean_Pa": stats["change_mean"],
                    "pressure_change_median_Pa": stats["change_median"],
                    "pressure_change_p05_Pa": stats["change_p05"],
                    "pressure_change_p95_Pa": stats["change_p95"],
                    "pressure_change_min_Pa": stats["change_min"],
                    "pressure_change_max_Pa": stats["change_max"],
                    "pressure_change_representative_Pa": stats["change_center"],
                    "pressure_baseline_time": baseline_time,
                    "pressure_field": pressure_name,
                    "mapping": str(mapping_path),
                })

            if not pressure_samples:
                print(
                    f"WARNING: no LIQUID_PRESSURE field found in {flow_path}; "
                    "pressure plots will be skipped."
                )
            else:
                pressure_samples.sort(key=lambda item: item[0])
                pressure_times = np.asarray(
                    [item[0] for item in pressure_samples],
                    dtype=float,
                )
                pressure_series[pressure_region.name] = {
                    key: [sample[1][key] for sample in pressure_samples]
                    for key in (
                        "mean", "median", "p05", "p95", "min", "max", "center",
                        "change_mean", "change_median", "change_p05",
                        "change_p95", "change_min", "change_max", "change_center",
                    )
                }
                pressure_series[pressure_region.name]["times"] = pressure_times.tolist()
                pressure_plot_dir = output_dir / "plots" / "pressure"
                pressure_plot_dir.mkdir(parents=True, exist_ok=True)
                plot_outputs.extend(
                    plot_region_pressure(
                        region=pressure_region,
                        times=pressure_times,
                        pressure=pressure_series[pressure_region.name],
                        output_base=pressure_plot_dir
                        / f"{safe_name(pressure_region.name)}_pressure",
                        formats=args.plot_formats,
                        dpi=args.dpi,
                        cell_count=int(injector_flow_cells.size),
                        center_cell_id=int(center_cell_id),
                        baseline_time=float(baseline_time),
                    )
                )

    preferred = (
        "time",
        "time_unit",
        "region",
        "region_id",
        "node_count",
        "center_x_m",
        "center_y_m",
        "center_z_m",
    )
    combined_csv = output_dir / "region_timeseries.csv"
    write_csv_rows(combined_csv, all_rows, preferred)

    for region_name in region_names:
        region_rows = [row for row in all_rows if row["region"] == region_name]
        write_csv_rows(
            output_dir / f"{safe_name(region_name)}_timeseries.csv",
            region_rows,
            preferred,
        )

    strain_long_csv = output_dir / "strain_components_long.csv"
    write_csv_rows(
        strain_long_csv,
        strain_long_rows,
        (
            "time",
            "time_unit",
            "region",
            "region_id",
            "node_count",
            "component",
            "mean_strain",
            "std_strain",
            "min_strain",
            "max_strain",
        ),
    )

    pvd_path = output_dir / "region_timeseries.pvd"
    write_pvd(pvd_path, pvd_records)

    if pressure_long_rows:
        pressure_csv = output_dir / "pressure_timeseries.csv"
        write_csv_rows(
            pressure_csv,
            pressure_long_rows,
            (
                "time", "time_unit", "region", "region_id", "mechanics_node_count", "flow_cell_count",
                "representative_cell_id", "representative_cell_distance_m",
                "pressure_mean_Pa", "pressure_median_Pa", "pressure_p05_Pa", "pressure_p95_Pa",
                "pressure_min_Pa", "pressure_max_Pa", "pressure_representative_Pa",
                "pressure_change_mean_Pa", "pressure_change_median_Pa",
                "pressure_change_p05_Pa", "pressure_change_p95_Pa",
                "pressure_change_min_Pa", "pressure_change_max_Pa",
                "pressure_change_representative_Pa", "pressure_baseline_time", "pressure_field", "mapping",
            ),
        )
    else:
        pressure_csv = None

    # Publication-quality plots: one axes per figure, never subplots.
    by_region_dir = output_dir / "plots" / "by_region"
    by_component_dir = output_dir / "plots" / "by_component"
    displacement_dir = output_dir / "plots" / "displacement"

    for region in regions:
        region_axial: Mapping[str, Sequence[float]] | None = None
        if region.name in axes and axial_series[region.name]["mean"]:
            region_axial = axial_series[region.name]

        plot_outputs.extend(
            plot_region_strains(
                region=region,
                times=times,
                time_unit=time_unit,
                series=all_series[region.name],
                axis_series=region_axial,
                output_base=by_region_dir
                / f"{safe_name(region.name)}_all_strain_components",
                formats=args.plot_formats,
                dpi=args.dpi,
                strain_unit=args.strain_unit,
                spread_mode=args.plot_spread,
                include_volumetric=args.include_volumetric_in_strain_plot,
                symmetric_y=args.symmetric_strain_axis,
            )
        )

        if not args.no_displacement_plots:
            plot_outputs.extend(
                plot_region_displacement(
                    region=region,
                    times=times,
                    time_unit=time_unit,
                    series=all_series[region.name],
                    output_base=displacement_dir
                    / f"{safe_name(region.name)}_displacement_components",
                    formats=args.plot_formats,
                    dpi=args.dpi,
                )
            )

    if not args.no_component_comparison_plots:
        for component, label in STRAIN_COMPONENTS:
            plot_outputs.extend(
                plot_component_across_regions(
                    component=component,
                    label=label,
                    regions=regions,
                    times=times,
                    time_unit=time_unit,
                    all_series=all_series,
                    output_base=by_component_dir
                    / f"{component}_all_regions",
                    formats=args.plot_formats,
                    dpi=args.dpi,
                    strain_unit=args.strain_unit,
                )
            )

    if not args.no_regional_strain_norm_plot:
        plot_outputs.extend(
            plot_region_strain_norm_across_regions(
                regions=regions,
                times=times,
                all_series=all_series,
                output_base=output_dir / "plots" / "high_quality" / "regional_strain_tensor_norm_nanostrain",
                formats=args.plot_formats,
                dpi=args.dpi,
            )
        )

    manifest = {
        "geomechanics_hdf5": str(h5_path),
        "node_count": node_count,
        "time_values": times.tolist(),
        "time_unit": time_unit,
        "aggregation": (
            "Arithmetic mean over all unique 1-based node IDs in each vset. "
            "Standard deviation, minimum, and maximum quantify spatial spread. "
            "Integer fields use the modal value."
        ),
        "interpretation_warning": (
            "HEC and Injection curves are spatial region summaries, not point "
            "measurements. Signed strain components may cancel in a region mean."
        ),
        "strain_plot_components": [name for name, _ in STRAIN_COMPONENTS],
        "strain_plot_unit": args.strain_unit,
        "plot_spread": args.plot_spread,
        "pressure": {
            "enabled": bool(pressure_long_rows),
            "flow_hdf5": str(args.flow_h5.expanduser().resolve()) if args.flow_h5 is not None else None,
            "mapping": str(args.mapping.expanduser().resolve()) if args.mapping is not None else None,
            "region": safe_name(args.pressure_region),
            "csv": pressure_csv.name if pressure_csv is not None else None,
            "sampling": "cell-centered",
            "definition": (
                "LIQUID_PRESSURE is sampled on flow cells selected by taking the "
                "mechanics-node IDs in injection_borehole.vset and joining them "
                "through the validated flow-cell/mechanics-vertex mapping. The "
                "representative cell is selected by median instantaneous pressure "
                "ordering at the first available flow snapshot. Pressure change "
                "is baseline-subtracted cell-by-cell from that same first snapshot."
            ),
        },
        "derived_strain": {
            "tensor_frobenius_norm": "sqrt(exx^2 + eyy^2 + ezz^2 + 2*(exy^2 + eyz^2 + ezx^2))",
            "summary_plot_unit": "nanostrain",
        },
        "regions": [
            {
                "name": region.name,
                "vset": str(region.vset_path),
                "node_count": int(region.indices.size),
                "center_xyz_m": region.center_xyz.tolist(),
                "axis": axes[region.name].tolist()
                if region.name in axes
                else None,
            }
            for region in regions
        ],
        "field_name_mapping": field_name_map,
        "outputs": {
            "combined_csv": combined_csv.name,
            "strain_long_csv": strain_long_csv.name,
            "pressure_csv": pressure_csv.name if pressure_csv is not None else None,
            "pvd": pvd_path.name,
            "vtu_files": [path.name for _, path in pvd_records],
            "plot_files": [
                str(Path(value).relative_to(output_dir))
                for value in plot_outputs
            ],
        },
    }
    manifest_path = output_dir / "region_timeseries_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRegion time-series export complete")
    print(f"  combined CSV:       {combined_csv}")
    print(f"  long strain CSV:    {strain_long_csv}")
    if pressure_csv is not None:
        print(f"  pressure CSV:       {pressure_csv}")
    print(f"  ParaView PVD:       {pvd_path}")
    print(f"  figures written:    {len(plot_outputs)}")
    print(f"  manifest:           {manifest_path}")
    print("\nPer-region six-component plots:")
    print(f"  {by_region_dir}")
    print("\nIn ParaView:")
    print("  Open region_timeseries.pvd")
    print("  Threshold Region_ID or <NAME>_Flag")
    print("  Apply Plot Data Over Time")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
