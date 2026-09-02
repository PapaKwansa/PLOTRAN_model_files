#!/usr/bin/env python3
"""
Quantitatively compare matched one-way and two-way PFLOTRAN runs.

The script compares both geomechanics and, optionally, flow HDF5 outputs at
common output times. It validates that the two geomechanics files use the same
vertices, applies the validated flow-cell-to-mechanics-vertex mapping to vset
regions, and writes:

  * full-field comparison metrics for every common numeric field;
  * region-mean comparison metrics for AVN2, AVN87, AVN31, Injection, HEC, etc.;
  * publication-quality one-way/two-way strain overlays;
  * two-way-minus-one-way strain-difference plots;
  * displacement-magnitude comparisons;
  * liquid-pressure comparisons when flow files are supplied;
  * a machine-readable JSON summary.

The six strain components are kept dimensionless. No nanostrain conversion is
performed unless --strain-unit nanostrain is requested for plotting. CSV values
always remain in their native HDF5 units.

Example
-------
python3 compare_oneway_twoway.py \
  --oneway-geomech north_avant_v5_oneway_injection_smoke-geomech.h5 \
  --twoway-geomech north_avant_v5_twoway_injection_smoke-geomech.h5 \
  --oneway-flow north_avant_v5_oneway_injection_smoke.h5 \
  --twoway-flow north_avant_v5_twoway_injection_smoke.h5 \
  --mapping bartlesville_hec_lime_v5_interfaces_median.mapping \
  --region AVN2=AVN2.vset \
  --region AVN87=AVN87.vset \
  --region AVN31=AVN31.vset \
  --region Injection=injection_borehole.vset \
  --region HEC=hec.vset \
  --output-dir compare_oneway_twoway_v5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TIME_PATTERN = re.compile(
    r"Time\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"\s*([A-Za-z]*)",
    re.IGNORECASE,
)

STRAIN_COMPONENTS = (
    "strain_xx",
    "strain_yy",
    "strain_zz",
    "strain_xy",
    "strain_yz",
    "strain_zx",
)

STRAIN_LABELS = {
    "strain_xx": r"$\varepsilon_{xx}$",
    "strain_yy": r"$\varepsilon_{yy}$",
    "strain_zz": r"$\varepsilon_{zz}$",
    "strain_xy": r"$\varepsilon_{xy}$",
    "strain_yz": r"$\varepsilon_{yz}$",
    "strain_zx": r"$\varepsilon_{zx}$",
}


def safe_name(name: str) -> str:
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
    name, path_text = text.split("=", 1)
    name = safe_name(name)
    path = Path(path_text.strip())
    if not name or not path_text.strip():
        raise argparse.ArgumentTypeError("NAME and PATH must be nonempty")
    return name, path


def parse_time(group_path: str) -> tuple[float, str]:
    match = TIME_PATTERN.search(group_path)
    if not match:
        raise ValueError(f"Could not parse time from {group_path!r}")
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    return value, (match.group(2) or "").strip()


def time_to_hours(value: float, unit: str) -> float:
    key = unit.strip().lower()
    if key in {"", "h", "hr", "hrs", "hour", "hours"}:
        return float(value)
    if key in {"s", "sec", "secs", "second", "seconds"}:
        return float(value) / 3600.0
    if key in {"min", "mins", "minute", "minutes"}:
        return float(value) / 60.0
    if key in {"d", "day", "days"}:
        return float(value) * 24.0
    raise ValueError(f"Unsupported time unit {unit!r}")


def compatible_array(dataset: h5py.Dataset, expected_count: int) -> np.ndarray | None:
    shape = tuple(int(value) for value in dataset.shape)
    if shape == (expected_count,):
        values = np.asarray(dataset[...])
    elif shape == (expected_count, 1):
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
            raise RuntimeError(f"{dataset.name!r} contains NaN or infinity")
    return values


def discover_time_groups(
    h5: h5py.File,
    expected_count: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        if "time" not in name.lower():
            return

        compatible = 0
        for item in obj.values():
            if isinstance(item, h5py.Dataset):
                if compatible_array_shape(item.shape, expected_count):
                    compatible += 1

        if compatible == 0:
            return

        try:
            value, unit = parse_time(name)
            hours = time_to_hours(value, unit)
        except ValueError:
            return

        groups.append(
            {
                "time_native": float(value),
                "time_unit": unit or "h",
                "time_hours": float(hours),
                "path": "/" + name,
                "compatible_dataset_count": compatible,
            }
        )

    h5.visititems(visitor)
    groups.sort(key=lambda item: (item["time_hours"], item["path"]))
    return groups


def compatible_array_shape(shape: tuple[int, ...], expected_count: int) -> bool:
    values = tuple(int(value) for value in shape)
    return values in {(expected_count,), (expected_count, 1)}


def pair_time_groups(
    one_groups: list[dict[str, Any]],
    two_groups: list[dict[str, Any]],
    tolerance_hours: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[int] = set()

    for one in one_groups:
        distances = [
            abs(one["time_hours"] - two["time_hours"])
            if index not in used
            else float("inf")
            for index, two in enumerate(two_groups)
        ]
        if not distances:
            continue
        index = int(np.argmin(distances))
        if distances[index] <= tolerance_hours:
            used.add(index)
            pairs.append((one, two_groups[index]))

    return pairs


def arrays_from_group(
    group: h5py.Group,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    arrays: dict[str, dict[str, Any]] = {}

    for original_name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        values = compatible_array(dataset, expected_count)
        if values is None:
            continue

        key = normalized_name(original_name)
        if key in arrays:
            raise RuntimeError(
                f"Field-name collision after normalization in {group.name}: "
                f"{arrays[key]['original_name']!r} and {original_name!r}"
            )

        arrays[key] = {
            "original_name": original_name,
            "values": values,
            "dtype": str(values.dtype),
        }

    return arrays


def read_vset(path: Path, maximum_id: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    values: set[int] = set()
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        value = int(text)
        if value < 1 or value > maximum_id:
            raise RuntimeError(
                f"{path}: ID {value} is outside valid range 1..{maximum_id}"
            )
        values.add(value)

    if not values:
        raise RuntimeError(f"{path}: no IDs found")

    return np.asarray(sorted(value - 1 for value in values), dtype=np.int64)


def read_mapping(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(f"{path}: mapping needs at least two columns")

    flow_ids = data[:, 0]
    mechanics_ids = data[:, 1]

    if len(np.unique(flow_ids)) != len(flow_ids):
        raise RuntimeError(f"{path}: duplicate flow-cell IDs")
    if len(np.unique(mechanics_ids)) != len(mechanics_ids):
        raise RuntimeError(
            f"{path}: this comparison utility requires a one-to-one mapping"
        )
    if np.min(flow_ids) < 1 or np.min(mechanics_ids) < 1:
        raise RuntimeError(f"{path}: mapping IDs must be one-based positive integers")

    return flow_ids, mechanics_ids


def flow_indices_for_regions(
    mechanics_region_indices: dict[str, np.ndarray],
    mapping_path: Path,
    mechanics_count: int,
) -> tuple[dict[str, np.ndarray], int]:
    flow_ids, mechanics_ids = read_mapping(mapping_path)
    flow_count = int(np.max(flow_ids))

    if int(np.max(mechanics_ids)) > mechanics_count:
        raise RuntimeError(
            f"{mapping_path}: mechanics ID exceeds vertex count {mechanics_count}"
        )

    inverse = np.full(mechanics_count + 1, -1, dtype=np.int64)
    inverse[mechanics_ids] = flow_ids - 1

    output: dict[str, np.ndarray] = {}
    for name, zero_based_mechanics in mechanics_region_indices.items():
        mechanics_one_based = zero_based_mechanics + 1
        flow_zero_based = inverse[mechanics_one_based]
        if np.any(flow_zero_based < 0):
            missing = mechanics_one_based[flow_zero_based < 0][:10]
            raise RuntimeError(
                f"{mapping_path}: region {name} contains unmapped mechanics IDs "
                f"{missing.tolist()}"
            )
        output[name] = np.asarray(flow_zero_based, dtype=np.int64)

    return output, flow_count


def numeric_metrics(one: np.ndarray, two: np.ndarray) -> dict[str, float]:
    a = np.asarray(one, dtype=np.float64)
    b = np.asarray(two, dtype=np.float64)
    delta = b - a

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    norm_delta = float(np.linalg.norm(delta))
    scale = max(norm_a, norm_b, np.finfo(float).tiny)

    return {
        "oneway_mean": float(np.mean(a)),
        "twoway_mean": float(np.mean(b)),
        "mean_delta_twoway_minus_oneway": float(np.mean(delta)),
        "mean_absolute_delta": float(np.mean(np.abs(delta))),
        "rmse_delta": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_delta": float(np.max(np.abs(delta))),
        "oneway_l2_norm": norm_a,
        "twoway_l2_norm": norm_b,
        "delta_l2_norm": norm_delta,
        "symmetric_relative_l2": norm_delta / scale,
        "oneway_min": float(np.min(a)),
        "oneway_max": float(np.max(a)),
        "twoway_min": float(np.min(b)),
        "twoway_max": float(np.max(b)),
    }


def region_metrics(
    one: np.ndarray,
    two: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    a = np.asarray(one, dtype=np.float64)[indices]
    b = np.asarray(two, dtype=np.float64)[indices]
    delta = b - a

    one_mean = float(np.mean(a))
    two_mean = float(np.mean(b))
    denominator = max(abs(one_mean), abs(two_mean), np.finfo(float).tiny)

    return {
        "node_or_cell_count": int(indices.size),
        "oneway_mean": one_mean,
        "twoway_mean": two_mean,
        "mean_delta_twoway_minus_oneway": two_mean - one_mean,
        "symmetric_percent_difference_of_means": (
            100.0 * (two_mean - one_mean) / denominator
        ),
        "oneway_spatial_std": float(np.std(a)),
        "twoway_spatial_std": float(np.std(b)),
        "mean_absolute_nodal_delta": float(np.mean(np.abs(delta))),
        "rmse_nodal_delta": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_nodal_delta": float(np.max(np.abs(delta))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compare_pair(
    one_path: Path,
    two_path: Path,
    expected_count: int,
    region_indices: dict[str, np.ndarray],
    time_tolerance_hours: float,
    kind: str,
) -> dict[str, Any]:
    global_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "time_hours": [],
                "oneway": [],
                "twoway": [],
                "delta": [],
            }
        )
    )
    common_fields_by_time: dict[str, list[str]] = {}

    with h5py.File(one_path, "r") as one_h5, h5py.File(two_path, "r") as two_h5:
        one_groups = discover_time_groups(one_h5, expected_count)
        two_groups = discover_time_groups(two_h5, expected_count)
        pairs = pair_time_groups(
            one_groups,
            two_groups,
            time_tolerance_hours,
        )

        if not pairs:
            raise RuntimeError(
                f"No common {kind} output times were found between "
                f"{one_path.name} and {two_path.name}"
            )

        for one_group_info, two_group_info in pairs:
            time_hours = 0.5 * (
                one_group_info["time_hours"] + two_group_info["time_hours"]
            )

            one_arrays = arrays_from_group(
                one_h5[one_group_info["path"]],
                expected_count,
            )
            two_arrays = arrays_from_group(
                two_h5[two_group_info["path"]],
                expected_count,
            )
            common_keys = sorted(set(one_arrays) & set(two_arrays))

            common_fields_by_time[f"{time_hours:.16g}"] = common_keys

            if not common_keys:
                raise RuntimeError(
                    f"No common numeric {kind} fields at time {time_hours:g} h"
                )

            for key in common_keys:
                one_values = one_arrays[key]["values"]
                two_values = two_arrays[key]["values"]

                metrics = numeric_metrics(one_values, two_values)
                global_rows.append(
                    {
                        "kind": kind,
                        "time_hours": time_hours,
                        "field_key": key,
                        "oneway_field_name": one_arrays[key]["original_name"],
                        "twoway_field_name": two_arrays[key]["original_name"],
                        **metrics,
                    }
                )

                for region_name, indices in region_indices.items():
                    rmetrics = region_metrics(
                        one_values,
                        two_values,
                        indices,
                    )
                    region_rows.append(
                        {
                            "kind": kind,
                            "time_hours": time_hours,
                            "region": region_name,
                            "field_key": key,
                            "oneway_field_name": one_arrays[key]["original_name"],
                            "twoway_field_name": two_arrays[key]["original_name"],
                            **rmetrics,
                        }
                    )

                    series_entry = series[region_name][key]
                    series_entry["time_hours"].append(float(time_hours))
                    series_entry["oneway"].append(
                        float(rmetrics["oneway_mean"])
                    )
                    series_entry["twoway"].append(
                        float(rmetrics["twoway_mean"])
                    )
                    series_entry["delta"].append(
                        float(rmetrics["mean_delta_twoway_minus_oneway"])
                    )

    return {
        "global_rows": global_rows,
        "region_rows": region_rows,
        "series": {
            region: dict(fields)
            for region, fields in series.items()
        },
        "common_fields_by_time": common_fields_by_time,
        "common_time_hours": sorted(
            {
                float(row["time_hours"])
                for row in global_rows
            }
        ),
    }


def find_series_key(
    fields: dict[str, dict[str, list[float]]],
    requested: str,
) -> str | None:
    target = normalized_name(requested)

    if target in fields:
        return target

    candidates = [
        key
        for key in fields
        if key == target or key.startswith(target + "_")
    ]
    return candidates[0] if candidates else None


def strain_multiplier_and_label(unit: str) -> tuple[float, str]:
    if unit == "dimensionless":
        return 1.0, "Strain [dimensionless]"
    if unit == "microstrain":
        return 1.0e6, r"Strain [$\mu\varepsilon$]"
    if unit == "nanostrain":
        return 1.0e9, "Strain [nanostrain]"
    raise ValueError(unit)


def configure_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, linewidth=0.8, color="0.35")
    ax.tick_params(direction="out")


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_region_strain_comparisons(
    series: dict[str, dict[str, dict[str, list[float]]]],
    output_dir: Path,
    strain_unit: str,
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    multiplier, ylabel = strain_multiplier_and_label(strain_unit)
    outputs: list[str] = []

    for region, fields in series.items():
        component_keys: list[tuple[str, str]] = []
        for component in STRAIN_COMPONENTS:
            key = find_series_key(fields, component)
            if key is not None:
                component_keys.append((component, key))

        if not component_keys:
            continue

        fig, ax = plt.subplots(figsize=(11.5, 7.0))
        component_handles: list[Line2D] = []

        for component, key in component_keys:
            values = fields[key]
            time = np.asarray(values["time_hours"], dtype=float)
            one = multiplier * np.asarray(values["oneway"], dtype=float)
            two = multiplier * np.asarray(values["twoway"], dtype=float)

            line, = ax.plot(
                time,
                two,
                marker="o",
                linewidth=2.0,
                label=STRAIN_LABELS[component],
            )
            ax.plot(
                time,
                one,
                marker="x",
                linestyle="--",
                linewidth=1.6,
                color=line.get_color(),
            )
            component_handles.append(line)

        configure_axis(ax)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{region}: one-way versus two-way mean strain components"
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        component_legend = ax.legend(
            handles=component_handles,
            title="Component",
            loc="best",
            ncol=2,
        )
        ax.add_artist(component_legend)
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], linestyle="-", marker="o",
                    color="0.15", label="Two-way",
                ),
                Line2D(
                    [0], [0], linestyle="--", marker="x",
                    color="0.15", label="One-way",
                ),
            ],
            title="Coupling",
            loc="upper left",
        )

        stem = output_dir / f"{safe_name(region)}_strain_oneway_twoway"
        save_figure(fig, stem, dpi)
        outputs.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])

        fig, ax = plt.subplots(figsize=(11.5, 7.0))
        for component, key in component_keys:
            values = fields[key]
            time = np.asarray(values["time_hours"], dtype=float)
            delta = multiplier * np.asarray(values["delta"], dtype=float)
            ax.plot(
                time,
                delta,
                marker="o",
                linewidth=2.0,
                label=STRAIN_LABELS[component],
            )

        configure_axis(ax)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(
            "Two-way minus one-way " + ylabel.lower()
        )
        ax.set_title(
            f"{region}: coupling-feedback strain difference"
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.legend(title="Component", loc="best", ncol=2)

        stem = output_dir / f"{safe_name(region)}_strain_delta"
        save_figure(fig, stem, dpi)
        outputs.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])

    return outputs


def compute_magnitude_series(
    fields: dict[str, dict[str, list[float]]],
    prefix: str,
) -> dict[str, np.ndarray] | None:
    keys = [
        find_series_key(fields, f"{prefix}_x"),
        find_series_key(fields, f"{prefix}_y"),
        find_series_key(fields, f"{prefix}_z"),
    ]
    if any(key is None for key in keys):
        return None

    assert all(key is not None for key in keys)
    selected = [fields[key] for key in keys if key is not None]
    time = np.asarray(selected[0]["time_hours"], dtype=float)

    one = np.sqrt(
        sum(np.asarray(item["oneway"], dtype=float) ** 2 for item in selected)
    )
    two = np.sqrt(
        sum(np.asarray(item["twoway"], dtype=float) ** 2 for item in selected)
    )

    return {
        "time_hours": time,
        "oneway": one,
        "twoway": two,
        "delta": two - one,
    }


def plot_displacement_comparison(
    series: dict[str, dict[str, dict[str, list[float]]]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    plotted = False

    for region, fields in series.items():
        magnitude = compute_magnitude_series(fields, "displacement")
        if magnitude is None:
            continue
        plotted = True
        line, = ax.plot(
            magnitude["time_hours"],
            magnitude["twoway"],
            marker="o",
            linewidth=2.0,
            label=region,
        )
        ax.plot(
            magnitude["time_hours"],
            magnitude["oneway"],
            marker="x",
            linestyle="--",
            linewidth=1.6,
            color=line.get_color(),
        )

    if not plotted:
        plt.close(fig)
        return outputs

    configure_axis(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Magnitude of region-mean displacement vector [m]")
    ax.set_title("One-way versus two-way displacement response")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    region_legend = ax.legend(title="Region", loc="best", ncol=2)
    ax.add_artist(region_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="0.15", marker="o", label="Two-way"),
            Line2D(
                [0], [0], color="0.15", marker="x",
                linestyle="--", label="One-way",
            ),
        ],
        title="Coupling",
        loc="upper left",
    )

    stem = output_dir / "displacement_magnitude_oneway_twoway"
    save_figure(fig, stem, dpi)
    outputs.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))])
    return outputs


def choose_pressure_key(
    series: dict[str, dict[str, dict[str, list[float]]]],
) -> str | None:
    all_keys: set[str] = set()
    for fields in series.values():
        all_keys.update(fields)

    preferred = [
        key for key in sorted(all_keys)
        if "liquid_pressure" in key
    ]
    if preferred:
        return preferred[0]

    generic = [
        key for key in sorted(all_keys)
        if "pressure" in key and "reference" not in key
    ]
    return generic[0] if generic else None


def plot_pressure_comparison(
    series: dict[str, dict[str, dict[str, list[float]]]],
    output_dir: Path,
    dpi: int,
) -> tuple[list[str], str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure_key = choose_pressure_key(series)
    if pressure_key is None:
        return [], None

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for region, fields in series.items():
        if pressure_key not in fields:
            continue
        values = fields[pressure_key]
        time = np.asarray(values["time_hours"], dtype=float)
        one = np.asarray(values["oneway"], dtype=float)
        two = np.asarray(values["twoway"], dtype=float)

        line, = ax.plot(
            time,
            two,
            marker="o",
            linewidth=2.0,
            label=region,
        )
        ax.plot(
            time,
            one,
            marker="x",
            linestyle="--",
            linewidth=1.6,
            color=line.get_color(),
        )

    configure_axis(ax)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Region-mean liquid pressure [Pa]")
    ax.set_title("One-way versus two-way flow-pressure response")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    region_legend = ax.legend(title="Region", loc="best", ncol=2)
    ax.add_artist(region_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="0.15", marker="o", label="Two-way"),
            Line2D(
                [0], [0], color="0.15", marker="x",
                linestyle="--", label="One-way",
            ),
        ],
        title="Coupling",
        loc="upper left",
    )

    stem = output_dir / "liquid_pressure_oneway_twoway"
    save_figure(fig, stem, dpi)
    return [
        str(stem.with_suffix(".png")),
        str(stem.with_suffix(".pdf")),
    ], pressure_key


def top_latest_metrics(
    rows: list[dict[str, Any]],
    count: int = 12,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    latest = max(float(row["time_hours"]) for row in rows)
    candidates = [
        row for row in rows
        if math.isclose(
            float(row["time_hours"]),
            latest,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    return sorted(
        candidates,
        key=lambda row: float(row["maximum_absolute_delta"]),
        reverse=True,
    )[:count]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare matched PFLOTRAN one-way and two-way outputs."
    )
    parser.add_argument("--oneway-geomech", type=Path, required=True)
    parser.add_argument("--twoway-geomech", type=Path, required=True)
    parser.add_argument("--oneway-flow", type=Path)
    parser.add_argument("--twoway-flow", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument(
        "--region",
        action="append",
        required=True,
        type=parse_named_path,
        metavar="NAME=VSET",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compare_oneway_twoway"),
    )
    parser.add_argument(
        "--time-tolerance-hours",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument(
        "--strain-unit",
        choices=("dimensionless", "microstrain", "nanostrain"),
        default="dimensionless",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    one_geomech = args.oneway_geomech.expanduser().resolve()
    two_geomech = args.twoway_geomech.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (one_geomech, two_geomech):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(one_geomech, "r") as one_h5, h5py.File(two_geomech, "r") as two_h5:
        if "/Domain/Vertices" not in one_h5 or "/Domain/Vertices" not in two_h5:
            raise RuntimeError("Both geomechanics files require /Domain/Vertices")

        one_vertices = np.asarray(one_h5["/Domain/Vertices"][...], dtype=float)
        two_vertices = np.asarray(two_h5["/Domain/Vertices"][...], dtype=float)

        if one_vertices.shape != two_vertices.shape:
            raise RuntimeError(
                f"Geomechanics vertex shape mismatch: "
                f"{one_vertices.shape} versus {two_vertices.shape}"
            )

        maximum_vertex_difference = float(
            np.max(np.abs(one_vertices - two_vertices))
        )

    mechanics_count = int(one_vertices.shape[0])
    mechanics_regions = {
        name: read_vset(path.expanduser().resolve(), mechanics_count)
        for name, path in args.region
    }

    print("One-way/two-way comparison")
    print("==========================")
    print(f"Mechanics vertices: {mechanics_count:,}")
    print(
        "Maximum one-way/two-way vertex difference: "
        f"{maximum_vertex_difference:.6e} m"
    )
    for name, indices in mechanics_regions.items():
        print(f"  region {name}: {indices.size:,} vertices")

    geomech = compare_pair(
        one_geomech,
        two_geomech,
        mechanics_count,
        mechanics_regions,
        args.time_tolerance_hours,
        "geomechanics",
    )

    write_csv(
        output_dir / "geomechanics_global_field_metrics.csv",
        geomech["global_rows"],
    )
    write_csv(
        output_dir / "geomechanics_region_field_metrics.csv",
        geomech["region_rows"],
    )

    figures: list[str] = []
    figures.extend(
        plot_region_strain_comparisons(
            geomech["series"],
            output_dir / "plots" / "strain_by_region",
            args.strain_unit,
            args.dpi,
        )
    )
    figures.extend(
        plot_displacement_comparison(
            geomech["series"],
            output_dir / "plots" / "displacement",
            args.dpi,
        )
    )

    flow: dict[str, Any] | None = None
    chosen_pressure_key: str | None = None

    flow_files_supplied = args.oneway_flow is not None or args.twoway_flow is not None
    if flow_files_supplied:
        if args.oneway_flow is None or args.twoway_flow is None:
            raise RuntimeError(
                "Supply both --oneway-flow and --twoway-flow, or neither"
            )
        if args.mapping is None:
            raise RuntimeError(
                "--mapping is required when flow files are compared"
            )

        one_flow = args.oneway_flow.expanduser().resolve()
        two_flow = args.twoway_flow.expanduser().resolve()
        mapping = args.mapping.expanduser().resolve()

        for path in (one_flow, two_flow, mapping):
            if not path.is_file():
                raise FileNotFoundError(path)

        flow_regions, flow_count = flow_indices_for_regions(
            mechanics_regions,
            mapping,
            mechanics_count,
        )

        print(f"Flow cells from mapping: {flow_count:,}")

        flow = compare_pair(
            one_flow,
            two_flow,
            flow_count,
            flow_regions,
            args.time_tolerance_hours,
            "flow",
        )

        write_csv(
            output_dir / "flow_global_field_metrics.csv",
            flow["global_rows"],
        )
        write_csv(
            output_dir / "flow_region_field_metrics.csv",
            flow["region_rows"],
        )

        pressure_figures, chosen_pressure_key = plot_pressure_comparison(
            flow["series"],
            output_dir / "plots" / "pressure",
            args.dpi,
        )
        figures.extend(pressure_figures)

    summary = {
        "oneway_geomechanics": str(one_geomech),
        "twoway_geomechanics": str(two_geomech),
        "mechanics_vertex_count": mechanics_count,
        "maximum_vertex_difference_m": maximum_vertex_difference,
        "regions": {
            name: int(indices.size)
            for name, indices in mechanics_regions.items()
        },
        "geomechanics_common_times_hours": geomech["common_time_hours"],
        "geomechanics_common_fields_by_time": geomech[
            "common_fields_by_time"
        ],
        "flow_common_times_hours": (
            flow["common_time_hours"] if flow is not None else []
        ),
        "flow_common_fields_by_time": (
            flow["common_fields_by_time"] if flow is not None else {}
        ),
        "chosen_pressure_field_key": chosen_pressure_key,
        "figures": figures,
    }

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCommon geomechanics times [h]:")
    for value in geomech["common_time_hours"]:
        print(f"  {value:.12g}")

    print("\nLargest full-field geomechanics differences at latest common time:")
    for row in top_latest_metrics(geomech["global_rows"]):
        print(
            f"  {row['field_key']:<34s} "
            f"max|delta|={float(row['maximum_absolute_delta']):.6e} "
            f"relL2={float(row['symmetric_relative_l2']):.6e}"
        )

    if flow is not None:
        print("\nCommon flow times [h]:")
        for value in flow["common_time_hours"]:
            print(f"  {value:.12g}")

        print("\nLargest full-field flow differences at latest common time:")
        for row in top_latest_metrics(flow["global_rows"]):
            print(
                f"  {row['field_key']:<34s} "
                f"max|delta|={float(row['maximum_absolute_delta']):.6e} "
                f"relL2={float(row['symmetric_relative_l2']):.6e}"
            )

    print("\nComparison complete")
    print(f"  output directory: {output_dir}")
    print(
        "  geomechanics metrics: "
        f"{output_dir / 'geomechanics_global_field_metrics.csv'}"
    )
    if flow is not None:
        print(
            "  flow metrics: "
            f"{output_dir / 'flow_global_field_metrics.csv'}"
        )
    print(f"  figures written: {len(figures)}")
    print(f"  summary: {output_dir / 'comparison_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
