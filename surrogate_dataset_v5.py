#!/usr/bin/env python3
"""Generate a North Avant V5 surrogate-training dataset.

This version is tailored to the validated V5 continuous 96-hour
flow + two-way geomechanics workflow.

Surrogate inputs
----------------
Only five hydraulic permeability scalars are sampled by Latin hypercube:
    overburden, bartlesville_sand, basal_layer, underburden, hec

The mechanical model is inherited unchanged from the deck template.  In the
current production baseline this includes the site-specific AVN87 mechanics:
    E = 15 GPa, nu = 0.28, Biot = 0.8

Injector pressure observable
----------------------------
The authoritative injector flow-cell definition is the flow HDF5 MATERIAL_ID
field, with MATERIAL_ID == 6 because the production deck defines
injection_borehole as material ID 6.

For those flow cells:
    Delta_p_i(t) = LIQUID_PRESSURE_i(t) - LIQUID_PRESSURE_i(0)

The script reports mean, median, P05, P95, min, max, and standard deviation
of Delta-p over the injection-borehole cells.

A mechanics-to-flow mapping audit is also performed when possible.  It is an
independent check only; it does not define the pressure cells.

Strain observables
------------------
AVN2, AVN31, and AVN87 are kept as separate stations.  Each station uses its
own vset and stores all six strain-tensor components plus volumetric strain.

Numerical execution
-------------------
Each realization is one full 96-hour PFLOTRAN run using the current production
deck, normally launched as 1 node / 64 MPI ranks.  Per-sample NPZ files make
the dataset resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# Surrogate inputs
# -----------------------------------------------------------------------------

MATERIALS = [
    "overburden",
    "bartlesville_sand",
    "basal_layer",
    "underburden",
    "hec",
]

# Baseline flow tensors from the validated V5 production deck.
# Each LHS scalar changes the horizontal permeability and preserves the
# baseline anisotropy ratio in z.
BASE_TENSORS = {
    "overburden": (9.869233e-18, 9.869233e-18, 9.869233e-19),
    "bartlesville_sand": (4.9346165e-15, 4.9346165e-15, 4.9346165e-17),
    "basal_layer": (9.869233e-18, 9.869233e-18, 9.869233e-19),
    "underburden": (9.869233e-18, 9.869233e-18, 9.869233e-19),
    "hec": (4.9346165e-13, 4.9346165e-13, 9.869233e-17),
}

# Conservative initial hydraulic ranges from the previous surrogate workflow.
# These are sensitivity ranges, not claims of exact in-situ values.
LOG10_TARGET_BOUNDS = {
    "overburden": (-18.0, -16.0),
    "bartlesville_sand": (-14.0, -12.0),
    "basal_layer": (-19.0, -17.0),
    "underburden": (-18.0, -16.0),
    "hec": (-13.0, -12.0),
}

DECK_DEFAULT = "north_avant_v5_twoway_production_96h_final.in"

STATIC_FILES = [
    "bartlesville_hec_lime_v5_interfaces_median.uge",
    "bartlesville_hec_lime_v5_interfaces.ugi",
    "bartlesville_hec_lime_v5_interfaces_median.mapping",
    "bartlesville_hec_lime_v5_interfaces_material_ids.h5",
    "boundary_ex_v5/top.ex",
    "boundary_ex_v5/bottom.ex",
    "boundary_ex_v5/north.ex",
    "boundary_ex_v5/south.ex",
    "boundary_ex_v5/east.ex",
    "boundary_ex_v5/west.ex",
    "top.vset",
    "bottom.vset",
    "north.vset",
    "south.vset",
    "east.vset",
    "west.vset",
    "overburden.vset",
    "bartlesville_sand.vset",
    "basal_layer.vset",
    "underburden.vset",
    "hec.vset",
    "injection_borehole.vset",
    "strainmeter_sensors.vset",
    "AVN2.vset",
    "AVN87.vset",
    "AVN31.vset",
]

PRESSURE_OBSERVATION_VSET = "injection_borehole.vset"
STRAIN_OBSERVATION_VSETS = {
    "AVN2": "AVN2.vset",
    "AVN31": "AVN31.vset",
    "AVN87": "AVN87.vset",
}

# The flow deck defines injection_borehole as ID 6.
INJECTION_MATERIAL_ID = 6

STRAIN_COMPONENTS = (
    "strain_xx",
    "strain_yy",
    "strain_zz",
    "strain_xy",
    "strain_yz",
    "strain_zx",
)

STRAIN_DATASET_CANDIDATES = [
    "strain_xx", "strain_yy", "strain_zz",
    "strain_xy", "strain_yz", "strain_zx",
    "GEOMECH_STRAIN_XX", "GEOMECH_STRAIN_YY", "GEOMECH_STRAIN_ZZ",
    "GEOMECH_STRAIN_XY", "GEOMECH_STRAIN_YZ", "GEOMECH_STRAIN_ZX",
    "STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
    "STRAIN_XY", "STRAIN_YZ", "STRAIN_ZX",
]

PRESSURE_DATASET_CANDIDATES = [
    "LIQUID_PRESSURE",
    "Liquid Pressure [Pa]",
    "Liquid Pressure",
]

MATERIAL_ID_DATASET_CANDIDATES = [
    "MATERIAL_ID",
    "Material ID",
    "Material ID []",
    "MaterialID",
]

TARGET_TIMES_H = np.array([
    0.0, 18.0, 18.94, 19.06, 19.25,
    19.50, 20.0, 22.0, 24.0, 36.0,
    48.0, 60.0, 72.0, 84.0, 96.0,
], dtype=float)

TIME_TOL_H = 2.0e-5


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate North Avant V5 surrogate-training data with automatic failed-sample retries."
    )
    p.add_argument("--model-dir", default=".")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--nprocs",
        type=int,
        default=int(os.environ.get("SLURM_NTASKS", "64")),
    )
    p.add_argument(
        "--pflotran-bin",
        default=os.environ.get("PFLOTRAN_BIN", "pflotran"),
    )
    p.add_argument(
        "--mpiexec",
        default=os.environ.get(
            "MPIEXEC",
            "/home/harhin/PFLOTRAN/petsc/arch-linux-c-opt/bin/mpiexec.hydra",
        ),
    )
    p.add_argument("--deck-template", default=DECK_DEFAULT)
    p.add_argument("--copy-static", action="store_true")
    p.add_argument("--keep-runs", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of additional attempts for a failed sample (default: 2).",
    )
    p.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="Disable automatic retries of failed samples.",
    )
    p.add_argument("--allow-failures", action="store_true")
    p.add_argument(
        "--skip-mapping-audit",
        action="store_true",
        help="Do not perform the independent mechanics-to-flow mapping audit.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(str(path))
    else:
        path.unlink()


def link_or_copy(src: Path, dst: Path, copy_mode: bool) -> None:
    safe_unlink(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode:
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    else:
        os.symlink(str(src.resolve()), str(dst))


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# -----------------------------------------------------------------------------
# VSET / mapping parsing
# -----------------------------------------------------------------------------

def parse_integer_tokens(path: Path) -> List[int]:
    values: List[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.split("#", 1)[0]
            for token in line.split():
                if re.fullmatch(r"[+-]?\d+", token):
                    values.append(int(token))
    return values


def load_vset_ids(path: Path) -> np.ndarray:
    """Load node IDs from a vset, handling the common count-first format.

    If the first integer equals the number of remaining integers, it is treated
    as a count/header rather than a node ID.  Otherwise all integers are treated
    as IDs.  The returned IDs remain in the file's original 1-based/0-based
    convention; conversion is handled later where the target array size is known.
    """
    values = parse_integer_tokens(path)
    if not values:
        raise RuntimeError("No integer IDs found in {}".format(path))
    if len(values) >= 2 and values[0] == len(values) - 1:
        values = values[1:]
    return np.unique(np.asarray(values, dtype=np.int64))


def convert_ids_to_zero_based(ids: np.ndarray, upper_bound: int, label: str) -> Tuple[np.ndarray, str]:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        raise RuntimeError("{} contains no IDs".format(label))

    if ids.min() >= 0 and ids.max() < upper_bound:
        return ids.copy(), "0-based"
    if ids.min() >= 1 and ids.max() <= upper_bound:
        return ids - 1, "1-based"

    raise IndexError(
        "{} contains IDs outside valid range 0..{} or 1..{}; min={}, max={}".format(
            label,
            upper_bound - 1,
            upper_bound,
            int(ids.min()),
            int(ids.max()),
        )
    )


def read_mapping(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError("Mapping must have at least two columns")

    flow_ids = data[:, 0]
    mech_ids = data[:, 1]

    if np.unique(flow_ids).size != flow_ids.size:
        raise RuntimeError("Duplicate flow IDs in mapping")
    if np.unique(mech_ids).size != mech_ids.size:
        raise RuntimeError("Duplicate mechanics IDs in mapping")
    if flow_ids.min() < 1 or mech_ids.min() < 1:
        raise RuntimeError("Mapping IDs are expected to be 1-based positive integers")

    return flow_ids - 1, mech_ids - 1


def audit_injector_mapping(
    model_dir: Path,
    mapping_path: Path,
    injector_vset: Path,
    material_flow_ids: np.ndarray,
) -> Dict[str, object]:
    """Independent audit: mechanics vset -> mapping -> flow IDs.

    This audit never defines the pressure cells.  The pressure cells are always
    obtained from MATERIAL_ID == 6 in the flow HDF5.
    """
    flow_ids, mech_ids = read_mapping(mapping_path)

    # The mapping is 0-based after read_mapping.  Infer the mechanics ID space
    # from the maximum mapped ID and convert the vset conservatively.
    mech_count = int(mech_ids.max()) + 1
    inj_mech_raw = load_vset_ids(injector_vset)
    inj_mech_zero, vset_base = convert_ids_to_zero_based(
        inj_mech_raw,
        mech_count,
        "{} mechanics IDs".format(injector_vset.name),
    )

    lookup = {int(m): int(f) for f, m in zip(flow_ids, mech_ids)}
    mapped = [lookup[int(m)] for m in inj_mech_zero if int(m) in lookup]
    mapped = np.unique(np.asarray(mapped, dtype=np.int64))

    material_set = set(int(x) for x in np.asarray(material_flow_ids, dtype=np.int64))
    mapped_set = set(int(x) for x in mapped)

    intersection = mapped_set.intersection(material_set)
    mapping_only = mapped_set.difference(material_set)
    material_only = material_set.difference(mapped_set)

    return {
        "vset_indexing": vset_base,
        "injector_mechanics_vertices": int(inj_mech_zero.size),
        "mapped_flow_cells": int(mapped.size),
        "material_id_flow_cells": int(material_flow_ids.size),
        "intersection": int(len(intersection)),
        "mapping_only": int(len(mapping_only)),
        "material_only": int(len(material_only)),
        "sets_match": mapped_set == material_set,
        "mapped_flow_cell_ids_0based": mapped,
    }


# -----------------------------------------------------------------------------
# Latin hypercube
# -----------------------------------------------------------------------------

def generate_lhs_unit_samples(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    unit = np.empty((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        cuts = np.linspace(0.0, 1.0, n_samples + 1)
        pts = cuts[:-1] + rng.rand(n_samples) * (cuts[1:] - cuts[:-1])
        rng.shuffle(pts)
        unit[:, j] = pts
    return unit


def generate_lhs_log10_samples(
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    names = MATERIALS[:]
    low = np.array([bounds[m][0] for m in names], dtype=float)
    high = np.array([bounds[m][1] for m in names], dtype=float)
    unit = generate_lhs_unit_samples(n_samples, len(names), seed)
    return low + unit * (high - low), names


# -----------------------------------------------------------------------------
# Deck editing
# -----------------------------------------------------------------------------

def replace_perm_tensor_in_block(
    text: str,
    material_name: str,
    perm_x: float,
    perm_y: float,
    perm_z: float,
) -> str:
    lines = text.splitlines(True)
    start = None
    block_re = re.compile(r"^\s*MATERIAL_PROPERTY\s+{}\s*$".format(re.escape(material_name)))

    for i, line in enumerate(lines):
        if block_re.match(line):
            start = i
            break

    if start is None:
        raise RuntimeError("Missing MATERIAL_PROPERTY block: {}".format(material_name))

    end = None
    for j in range(start + 1, len(lines)):
        if re.match(r"^\s*END\s*$", lines[j]):
            end = j
            break

    if end is None:
        raise RuntimeError("Missing END for material block: {}".format(material_name))

    found = {"x": False, "y": False, "z": False}
    vals = {"x": perm_x, "y": perm_y, "z": perm_z}

    for k in range(start + 1, end):
        for axis in ("x", "y", "z"):
            if re.match(r"^\s*PERM_{}\b".format(axis.upper()), lines[k]):
                indent = re.match(r"^(\s*)", lines[k]).group(1)
                lines[k] = "{}PERM_{} {:.9e}\n".format(
                    indent, axis.upper(), vals[axis]
                )
                found[axis] = True

    if not all(found.values()):
        raise RuntimeError("Incomplete permeability block: {}".format(material_name))

    return "".join(lines)


# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------

def parse_time_from_group_name(name: str) -> Optional[float]:
    m = re.search(
        r"Time\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*h",
        name,
    )
    if not m:
        return None
    return float(m.group(1).replace("D", "E").replace("d", "e"))


def find_dataset_in_group(group: h5py.Group, candidates: Sequence[str]) -> np.ndarray:
    norm = [normalize_name(c) for c in candidates]
    result = None

    def visitor(name, obj):
        nonlocal result
        if result is not None or not isinstance(obj, h5py.Dataset):
            return
        leaf = normalize_name(Path(name).name)
        if any(c == leaf or c in leaf or leaf in c for c in norm):
            result = np.asarray(obj[...], dtype=float).reshape(-1)

    group.visititems(visitor)
    if result is None:
        raise KeyError(
            "No dataset found in '{}' for candidates {}".format(group.name, candidates)
        )
    return result


def find_dataset_global(h5obj: h5py.File, candidates: Sequence[str]) -> np.ndarray:
    """Search the entire HDF5 file for a dataset matching candidates."""
    norm = [normalize_name(c) for c in candidates]
    result = None

    def visitor(name, obj):
        nonlocal result
        if result is not None or not isinstance(obj, h5py.Dataset):
            return
        leaf = normalize_name(Path(name).name)
        if any(c == leaf or c in leaf or leaf in c for c in norm):
            result = np.asarray(obj[...], dtype=float).reshape(-1)

    h5obj.visititems(visitor)
    if result is None:
        raise KeyError("No dataset found anywhere for candidates {}".format(candidates))
    return result


def find_time_groups(
    h5obj: h5py.File,
    required_candidates: Sequence[str],
) -> List[Tuple[float, str]]:
    norm = [normalize_name(c) for c in required_candidates]
    found: List[Tuple[float, str]] = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Group) or "Time" not in name:
            return

        has = False
        for child in obj.keys():
            if normalize_name(child) in norm:
                has = True
                break

        if not has:
            def nested(subname, subobj):
                nonlocal has
                if has or not isinstance(subobj, h5py.Dataset):
                    return
                leaf = normalize_name(Path(subname).name)
                if any(c == leaf or c in leaf or leaf in c for c in norm):
                    has = True

            obj.visititems(nested)

        if has:
            t = parse_time_from_group_name(Path(name).name)
            if t is not None:
                found.append((t, name))

    h5obj.visititems(visitor)
    return sorted(found, key=lambda x: x[0])


def select_target_groups(
    groups: List[Tuple[float, str]],
    targets: np.ndarray,
    label: str,
) -> List[Tuple[float, str]]:
    if not groups:
        raise RuntimeError("{} contains no recognized time groups".format(label))

    available = np.asarray([g[0] for g in groups], dtype=float)
    selected: List[Tuple[float, str]] = []

    for target in targets:
        i = int(np.argmin(np.abs(available - target)))
        err = abs(float(available[i] - target))
        if err > TIME_TOL_H:
            raise RuntimeError(
                "{} missing requested time {} h; nearest is {} h".format(
                    label, target, available[i]
                )
            )
        selected.append(groups[i])

    return selected


# -----------------------------------------------------------------------------
# Injector pressure extraction
# -----------------------------------------------------------------------------

def get_flow_pressure_and_material_id(
    h5: h5py.File,
    group_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    group = h5[group_path]

    pressure = find_dataset_in_group(group, PRESSURE_DATASET_CANDIDATES)

    try:
        material_id = find_dataset_in_group(group, MATERIAL_ID_DATASET_CANDIDATES)
    except KeyError:
        # Some PFLOTRAN layouts can store a static material ID array outside
        # the time group.  Use that only as a fallback.
        material_id = find_dataset_global(h5, MATERIAL_ID_DATASET_CANDIDATES)

    return pressure, material_id


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "p05": float(np.nanpercentile(values, 5.0)),
        "p95": float(np.nanpercentile(values, 95.0)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "std": float(np.nanstd(values)),
    }


def extract_pressure_delta_from_material_id(
    flow_h5: Path,
    mapping_path: Optional[Path],
    injector_vset: Optional[Path],
    do_mapping_audit: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Extract injector Delta-p using MATERIAL_ID == 6 as authoritative."""

    with h5py.File(str(flow_h5), "r") as fh:
        groups = find_time_groups(
            fh,
            PRESSURE_DATASET_CANDIDATES,
        )
        selected = select_target_groups(groups, TARGET_TIMES_H, "Flow")

        pressure0, material0 = get_flow_pressure_and_material_id(
            fh, selected[0][1]
        )

        if pressure0.size == 0:
            raise RuntimeError("Empty liquid-pressure array")

        if material0.size != pressure0.size:
            raise RuntimeError(
                "Material-ID array size {} does not match pressure array size {}".format(
                    material0.size, pressure0.size
                )
            )

        # Material IDs should be integral-valued.
        if not np.all(np.isfinite(material0)):
            raise RuntimeError("Material-ID array contains non-finite values")
        material0_int = np.rint(material0).astype(np.int64)

        injector_flow = np.flatnonzero(
            material0_int == INJECTION_MATERIAL_ID
        ).astype(np.int64)

        if injector_flow.size == 0:
            unique_ids = np.unique(material0_int)
            raise RuntimeError(
                "No flow cells with MATERIAL_ID == {}. Available IDs include {}".format(
                    INJECTION_MATERIAL_ID,
                    unique_ids[:50].tolist(),
                )
            )

        baseline = pressure0[injector_flow].copy()

        stats = {
            key: []
            for key in ("mean", "median", "p05", "p95", "min", "max", "std")
        }
        times: List[float] = []

        for t, group_path in selected:
            pressure, material = get_flow_pressure_and_material_id(fh, group_path)

            if pressure.size != pressure0.size:
                raise RuntimeError(
                    "Flow-cell count changed at {} h: {} vs {}".format(
                        t, pressure.size, pressure0.size
                    )
                )

            if material.size != material0.size:
                raise RuntimeError(
                    "Material-ID count changed at {} h: {} vs {}".format(
                        t, material.size, material0.size
                    )
                )

            current_material = np.rint(material).astype(np.int64)
            if not np.array_equal(current_material, material0_int):
                raise RuntimeError(
                    "MATERIAL_ID field changed between snapshot times; refusing to use a moving injector definition."
                )

            dp = pressure[injector_flow] - baseline
            s = compute_stats(dp)
            times.append(float(t))
            for key in stats:
                stats[key].append(s[key])

        for key in stats:
            stats[key] = np.asarray(stats[key], dtype=float)

        audit: Dict[str, object] = {
            "authoritative_definition": "MATERIAL_ID == {}".format(INJECTION_MATERIAL_ID),
            "material_id": INJECTION_MATERIAL_ID,
            "flow_cell_count": int(pressure0.size),
            "injector_flow_cell_count": int(injector_flow.size),
            "mapping_audit_performed": False,
        }

        if do_mapping_audit:
            if mapping_path is None or injector_vset is None:
                raise RuntimeError("Mapping audit requested but mapping/vset paths were not supplied")
            audit_result = audit_injector_mapping(
                flow_h5.parent,
                mapping_path,
                injector_vset,
                injector_flow,
            )
            audit.update(audit_result)
            audit["mapping_audit_performed"] = True

        return np.asarray(times, dtype=float), injector_flow, stats, audit


# -----------------------------------------------------------------------------
# Geomechanics extraction
# -----------------------------------------------------------------------------

def read_geomech_coordinates(h5: h5py.File) -> np.ndarray:
    required = ("/Domain/X", "/Domain/Y", "/Domain/Z")
    if not all(name in h5 for name in required):
        raise RuntimeError("Geomechanics HDF5 lacks /Domain/X,/Y,/Z")
    xyz = np.column_stack([
        np.asarray(h5[name][...], dtype=float).reshape(-1)
        for name in required
    ])
    if xyz.shape[0] == 0 or not np.all(np.isfinite(xyz)):
        raise RuntimeError("Invalid geomechanics coordinates")
    return xyz


def extract_strain_series(
    geomech_h5: Path,
    station_vsets: Dict[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    with h5py.File(str(geomech_h5), "r") as gh:
        xyz = read_geomech_coordinates(gh)
        groups = find_time_groups(gh, STRAIN_DATASET_CANDIDATES)
        selected = select_target_groups(groups, TARGET_TIMES_H, "Geomechanics")

        stations = list(station_vsets.keys())
        n_s = len(stations)
        n_t = len(selected)
        n_c = len(STRAIN_COMPONENTS)

        mean_arr = np.empty((n_s, n_t, n_c), dtype=float)
        std_arr = np.empty((n_s, n_t, n_c), dtype=float)
        vol_arr = np.empty((n_s, n_t), dtype=float)
        counts: Dict[str, int] = {}

        for si, station in enumerate(stations):
            raw_ids = load_vset_ids(station_vsets[station])
            idx, _base = convert_ids_to_zero_based(
                raw_ids,
                xyz.shape[0],
                "{} vset IDs".format(station),
            )
            if idx.size == 0:
                raise RuntimeError("Empty vset for {}".format(station))
            counts[station] = int(idx.size)

            for ti, (_t, group_path) in enumerate(selected):
                grp = gh[group_path]
                vals = np.empty((idx.size, n_c), dtype=float)

                for ci, comp in enumerate(STRAIN_COMPONENTS):
                    candidates = [
                        comp,
                        comp.upper(),
                        comp.replace("_", " "),
                        "GEOMECH_" + comp.upper(),
                        "GEOMECH_" + comp.upper().replace("_", " "),
                    ]
                    arr = find_dataset_in_group(grp, candidates)
                    if idx.max() >= arr.size:
                        raise IndexError(
                            "{} indices exceed {} array".format(station, comp)
                        )
                    vals[:, ci] = arr[idx]

                mean_arr[si, ti, :] = np.nanmean(vals, axis=0)
                std_arr[si, ti, :] = np.nanstd(vals, axis=0)
                vol_arr[si, ti] = float(np.sum(mean_arr[si, ti, 0:3]))

        audit = {
            "station_node_counts": counts,
            "station_order": stations,
            "strain_components": list(STRAIN_COMPONENTS),
        }

        return (
            np.asarray([x[0] for x in selected]),
            mean_arr,
            std_arr,
            vol_arr,
            audit,
        )


# -----------------------------------------------------------------------------
# PFLOTRAN run preparation/execution
# -----------------------------------------------------------------------------

def prepare_sample_run_dir(
    model_dir: Path,
    run_root: Path,
    sample_id: int,
    k_map: Dict[str, float],
    deck_template_name: str,
    copy_static: bool,
) -> Path:
    sample_dir = run_root / "sample_{:04d}".format(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for fname in STATIC_FILES:
        src = model_dir / fname
        if not src.exists():
            raise FileNotFoundError("Missing required static file: {}".format(src))
        link_or_copy(src, sample_dir / fname, copy_static)

    deck_src = model_dir / deck_template_name
    text = deck_src.read_text(encoding="utf-8")

    for material in MATERIALS:
        _bx, _by, _bz = BASE_TENSORS[material]
        scale = k_map[material] / _bx
        text = replace_perm_tensor_in_block(
            text,
            material,
            _bx * scale,
            _by * scale,
            _bz * scale,
        )

    # Use the deck stem explicitly so PFLOTRAN outputs have predictable names.
    prefix = Path(deck_template_name).stem
    (sample_dir / (prefix + ".in")).write_text(text, encoding="utf-8")
    return sample_dir


def run_pflotran(
    run_dir: Path,
    pflotran_bin: str,
    mpiexec: str,
    nprocs: int,
    input_prefix: str,
) -> None:
    cmd = [
        mpiexec,
        "-n",
        str(nprocs),
        pflotran_bin,
        "-input_prefix",
        input_prefix,
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Preserve the known-good PETSc/PFLOTRAN C++ runtime selection by removing
    # Anaconda's libstdc++ from LD_LIBRARY_PATH when launching PFLOTRAN.
    ld = env.get("LD_LIBRARY_PATH", "")
    if ld:
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in ld.split(":") if "anaconda3" not in p.lower()
        )

    with (run_dir / "pflotran_stdout.log").open("w", encoding="utf-8") as out:
        subprocess.run(
            cmd,
            cwd=str(run_dir),
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )


# -----------------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------------

def load_completed_samples(out_dir: Path, n_samples: int) -> Dict[int, Dict[str, np.ndarray]]:
    done: Dict[int, Dict[str, np.ndarray]] = {}
    for i in range(1, n_samples + 1):
        p = out_dir / "sample_outputs" / "sample_{:04d}.npz".format(i)
        if not p.exists():
            continue
        try:
            with np.load(str(p), allow_pickle=False) as z:
                done[i] = {k: z[k] for k in z.files}
        except Exception:
            continue
    return done


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def prepare_named_sample_run_dir(
    model_dir: Path,
    run_root: Path,
    run_name: str,
    k_map: Dict[str, float],
    deck_template_name: str,
    copy_static: bool,
) -> Path:
    """Prepare a clean named attempt directory for retries."""
    sample_dir = run_root / run_name
    if sample_dir.exists() or sample_dir.is_symlink():
        safe_unlink(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for fname in STATIC_FILES:
        src = model_dir / fname
        if not src.exists():
            raise FileNotFoundError("Missing required static file: {}".format(src))
        link_or_copy(src, sample_dir / fname, copy_static)

    deck_src = model_dir / deck_template_name
    text = deck_src.read_text(encoding="utf-8")

    for material in MATERIALS:
        _bx, _by, _bz = BASE_TENSORS[material]
        scale = k_map[material] / _bx
        text = replace_perm_tensor_in_block(
            text,
            material,
            _bx * scale,
            _by * scale,
            _bz * scale,
        )

    prefix = Path(deck_template_name).stem
    (sample_dir / (prefix + ".in")).write_text(text, encoding="utf-8")
    return sample_dir


def read_existing_manifest(path: Path) -> Dict[int, Dict[str, str]]:
    """Read a prior manifest when resuming an interrupted/failed dataset run."""
    rows: Dict[int, Dict[str, str]] = {}
    if not path.exists():
        return rows
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("sample_id"):
                    continue
                try:
                    sid = int(row["sample_id"])
                except ValueError:
                    continue
                rows[sid] = row
    except Exception:
        return {}
    return rows


def load_retry_history(path: Path) -> Dict[str, object]:
    """Load retry history, tolerating the absence of a history file."""
    if not path.exists():
        return {"version": 1, "max_retries": 2, "samples": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "max_retries": 2, "samples": {}}
    if not isinstance(data, dict):
        return {"version": 1, "max_retries": 2, "samples": {}}
    samples = data.get("samples")
    if not isinstance(samples, dict):
        data["samples"] = {}
    return data


def save_retry_history(path: Path, history: Dict[str, object]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def history_entries(history: Dict[str, object], sid: int) -> List[Dict[str, object]]:
    samples = history.setdefault("samples", {})
    key = str(sid)
    entries = samples.setdefault(key, [])
    if not isinstance(entries, list):
        entries = []
        samples[key] = entries
    return entries


def write_checkpoint_manifest(
    out_dir: Path,
    lhs_log10: np.ndarray,
    names: Sequence[str],
    sample_status: Dict[int, Dict[str, object]],
    n_samples: int,
) -> None:
    """Write a resumable manifest after each sample attempt."""
    manifest = out_dir / "sample_manifest.csv"
    fieldnames = [
        "sample_id",
        "status",
        "attempts",
        "last_error",
        *[m + "_k" for m in names],
        "injector_material_id",
        "injector_flow_cells",
        "mapping_audit_sets_match",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid in range(1, n_samples + 1):
            k_map = {
                m: float(10.0 ** lhs_log10[sid - 1, j])
                for j, m in enumerate(names)
            }
            rec = sample_status.get(sid, {})
            row = {
                "sample_id": sid,
                "status": rec.get("status", "pending"),
                "attempts": rec.get("attempts", 0),
                "last_error": rec.get("last_error", ""),
                **{m + "_k": k_map[m] for m in names},
                "injector_material_id": INJECTION_MATERIAL_ID,
                "injector_flow_cells": rec.get("injector_flow_cells", ""),
                "mapping_audit_sets_match": rec.get("mapping_audit_sets_match", ""),
            }
            w.writerow(row)


def get_audit_fields(out_dir: Path, sid: int) -> Dict[str, object]:
    audit_path = out_dir / "sample_outputs" / "sample_{:04d}_audit.json".format(sid)
    if not audit_path.exists():
        return {}
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pressure = data.get("pressure_audit", {})
    if not isinstance(pressure, dict):
        pressure = {}
    return {
        "injector_flow_cells": pressure.get("injector_flow_cell_count", ""),
        "mapping_audit_sets_match": pressure.get("sets_match", ""),
    }


def save_master_dataset(
    out_dir: Path,
    names: Sequence[str],
    all_ok: List[Tuple[int, Dict[str, np.ndarray]]],
) -> Path:
    """Assemble the compact master NPZ from all successful samples."""
    all_ok.sort(key=lambda x: x[0])
    k_log10 = np.vstack([d["k_log10"] for _, d in all_ok])
    k_values = np.vstack([d["k_values"] for _, d in all_ok])
    dp_mean = np.vstack([d["injector_dp_mean_pa"] for _, d in all_ok])
    dp_median = np.vstack([d["injector_dp_median_pa"] for _, d in all_ok])
    dp_p05 = np.vstack([d["injector_dp_p05_pa"] for _, d in all_ok])
    dp_p95 = np.vstack([d["injector_dp_p95_pa"] for _, d in all_ok])
    dp_min = np.vstack([d["injector_dp_min_pa"] for _, d in all_ok])
    dp_max = np.vstack([d["injector_dp_max_pa"] for _, d in all_ok])
    dp_std = np.vstack([d["injector_dp_std_pa"] for _, d in all_ok])
    strain_mean = np.stack([d["strain_mean"] for _, d in all_ok], axis=0)
    strain_std = np.stack([d["strain_std"] for _, d in all_ok], axis=0)
    vol_strain = np.stack([d["volumetric_strain"] for _, d in all_ok], axis=0)

    path = out_dir / "dataset_master.npz"
    np.savez_compressed(
        str(path),
        material_names=np.asarray(names, dtype="U"),
        station_names=np.asarray(list(STRAIN_OBSERVATION_VSETS.keys()), dtype="U"),
        strain_component_names=np.asarray(STRAIN_COMPONENTS, dtype="U"),
        target_times_h=TARGET_TIMES_H,
        k_log10=k_log10,
        k_values=k_values,
        injector_material_id=np.asarray([INJECTION_MATERIAL_ID], dtype=np.int64),
        injector_dp_mean_pa=dp_mean,
        injector_dp_median_pa=dp_median,
        injector_dp_p05_pa=dp_p05,
        injector_dp_p95_pa=dp_p95,
        injector_dp_min_pa=dp_min,
        injector_dp_max_pa=dp_max,
        injector_dp_std_pa=dp_std,
        strain_mean=strain_mean,
        strain_std=strain_std,
        volumetric_strain=vol_strain,
    )
    return path


def main() -> int:
    args = parse_args()

    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    if args.nprocs < 1:
        raise ValueError("--nprocs must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    model_dir = Path(args.model_dir).resolve()
    if args.out_dir is None:
        args.out_dir = os.environ.get(
            "SURROGATE_OUTDIR",
            "./surrogate_dataset_v5",
        )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_outputs").mkdir(parents=True, exist_ok=True)

    deck_path = model_dir / args.deck_template
    if not deck_path.exists():
        raise FileNotFoundError("Deck template not found: {}".format(deck_path))

    lhs_log10, names = generate_lhs_log10_samples(
        args.n_samples,
        LOG10_TARGET_BOUNDS,
        args.seed,
    )

    sample_output_dir = out_dir / "sample_outputs"
    manifest_path = out_dir / "sample_manifest.csv"
    history_path = out_dir / "retry_history.json"
    previous_manifest = read_existing_manifest(manifest_path)
    history = load_retry_history(history_path)
    history["version"] = 1
    history["max_retries"] = args.max_retries
    history["seed"] = args.seed
    history["n_samples"] = args.n_samples

    completed = load_completed_samples(out_dir, args.n_samples)
    sample_status: Dict[int, Dict[str, object]] = {}

    for sid in range(1, args.n_samples + 1):
        prev = previous_manifest.get(sid, {})
        if sid in completed:
            audit = get_audit_fields(out_dir, sid)
            sample_status[sid] = {
                "status": "ok",
                "attempts": max(1, len(history_entries(history, sid))),
                "last_error": "",
                **audit,
            }
            continue

        prev_status = str(prev.get("status", ""))
        entries = history_entries(history, sid)

        # Migrate a failed sample from the original generator into retry history.
        if not entries and prev_status.startswith("failed:"):
            entries.append({
                "attempt": 1,
                "phase": "initial_existing_run",
                "status": "failed",
                "error": prev_status[len("failed:"):].strip(),
            })

        if entries:
            last = entries[-1]
            sample_status[sid] = {
                "status": "pending_retry" if last.get("status") == "failed" else str(last.get("status", "pending")),
                "attempts": len(entries),
                "last_error": str(last.get("error", "")) if last.get("status") == "failed" else "",
            }
        else:
            sample_status[sid] = {
                "status": "pending",
                "attempts": 0,
                "last_error": "",
            }

    deck_prefix = deck_path.stem
    deck_text = deck_path.read_text(encoding="utf-8")
    avn87_match = re.search(
        r"GEOMECHANICS_MATERIAL_PROPERTY\s+AVN87.*?YOUNGS_MODULUS\s+([0-9.eEdD+-]+).*?"
        r"POISSONS_RATIO\s+([0-9.eEdD+-]+).*?BIOT_COEFFICIENT\s+([0-9.eEdD+-]+).*?END",
        deck_text,
        re.S,
    )
    if avn87_match:
        avn87_E = float(avn87_match.group(1).replace("D", "E").replace("d", "e"))
        avn87_nu = float(avn87_match.group(2).replace("D", "E").replace("d", "e"))
        avn87_biot = float(avn87_match.group(3).replace("D", "E").replace("d", "e"))
    else:
        avn87_E = None
        avn87_nu = None
        avn87_biot = None

    print("=" * 72)
    print("North Avant V5 surrogate dataset generation")
    print("=" * 72)
    print("Model directory:", model_dir)
    print("Deck template  :", args.deck_template)
    print("Output directory:", out_dir)
    print("Samples        :", args.n_samples)
    print("Seed           :", args.seed)
    print("MPI ranks/run  :", args.nprocs)
    print("Injector cells : MATERIAL_ID == {}".format(INJECTION_MATERIAL_ID))
    print("AVN87 E/nu/Biot : {}/{}/{}".format(avn87_E, avn87_nu, avn87_biot))
    print("Automatic retries:", "disabled" if args.no_retry_failed else "{} additional attempt(s)".format(args.max_retries))
    print()

    # -------------------------------------------------------------------------
    # PASS 1: original deterministic LHS experiment. Existing successful NPZs
    # are reused when --resume is supplied. Existing failed samples are not
    # rerun as a new initial attempt; they enter the retry stage below.
    # -------------------------------------------------------------------------
    initial_failures: List[int] = []

    for i in range(args.n_samples):
        sid = i + 1
        sample_log10 = lhs_log10[i]
        k_map = {m: float(10.0 ** sample_log10[j]) for j, m in enumerate(names)}
        sample_npz = sample_output_dir / "sample_{:04d}.npz".format(sid)

        if sid in completed:
            print("[RESUME] sample {:04d}".format(sid), flush=True)
            continue

        entries = history_entries(history, sid)
        prev = previous_manifest.get(sid, {})
        if entries or str(prev.get("status", "")).startswith("failed:"):
            # This sample was already attempted in a previous invocation.
            if entries and entries[-1].get("status") == "failed":
                initial_failures.append(sid)
            continue

        sample_dir = prepare_sample_run_dir(
            model_dir,
            out_dir / "runs",
            sid,
            k_map,
            args.deck_template,
            args.copy_static,
        )

        try:
            run_pflotran(
                sample_dir,
                args.pflotran_bin,
                args.mpiexec,
                args.nprocs,
                deck_prefix,
            )

            flow_h5 = sample_dir / (deck_prefix + ".h5")
            geomech_h5 = sample_dir / (deck_prefix + "-geomech.h5")
            if not flow_h5.exists():
                raise FileNotFoundError(
                    "Flow HDF5 not found after PFLOTRAN run: {}".format(flow_h5)
                )
            if not geomech_h5.exists():
                raise FileNotFoundError(
                    "Geomechanics HDF5 not found after PFLOTRAN run: {}".format(geomech_h5)
                )

            mapping_path = sample_dir / "bartlesville_hec_lime_v5_interfaces_median.mapping"
            injector_vset = sample_dir / PRESSURE_OBSERVATION_VSET
            p_times, injector_flow, p_stats, pressure_audit = extract_pressure_delta_from_material_id(
                flow_h5,
                mapping_path if not args.skip_mapping_audit else None,
                injector_vset if not args.skip_mapping_audit else None,
                do_mapping_audit=not args.skip_mapping_audit,
            )

            station_paths = {k: sample_dir / v for k, v in STRAIN_OBSERVATION_VSETS.items()}
            s_times, s_mean, s_std, s_vol, strain_audit = extract_strain_series(
                geomech_h5,
                station_paths,
            )

            if not np.allclose(p_times, TARGET_TIMES_H, atol=TIME_TOL_H, rtol=0.0):
                raise RuntimeError("Extracted flow time grid does not match requested waypoints")
            if not np.allclose(s_times, TARGET_TIMES_H, atol=TIME_TOL_H, rtol=0.0):
                raise RuntimeError("Extracted geomechanics time grid does not match requested waypoints")

            payload: Dict[str, np.ndarray] = {
                "sample_id": np.asarray([sid], dtype=np.int64),
                "k_log10": sample_log10,
                "k_values": np.asarray([k_map[m] for m in names], dtype=float),
                "pressure_times_h": p_times,
                "injector_flow_cell_ids_0based": injector_flow,
                "injector_material_id": np.asarray([INJECTION_MATERIAL_ID], dtype=np.int64),
                "injector_dp_mean_pa": p_stats["mean"],
                "injector_dp_median_pa": p_stats["median"],
                "injector_dp_p05_pa": p_stats["p05"],
                "injector_dp_p95_pa": p_stats["p95"],
                "injector_dp_min_pa": p_stats["min"],
                "injector_dp_max_pa": p_stats["max"],
                "injector_dp_std_pa": p_stats["std"],
                "strain_times_h": s_times,
                "strain_mean": s_mean,
                "strain_std": s_std,
                "volumetric_strain": s_vol,
            }
            np.savez_compressed(str(sample_npz), **payload)

            (sample_output_dir / "sample_{:04d}_audit.json".format(sid)).write_text(
                json.dumps(
                    {"pressure_audit": pressure_audit, "strain_audit": strain_audit, "k_map": k_map},
                    indent=2,
                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                ),
                encoding="utf-8",
            )

            history_entries(history, sid).append({
                "attempt": 1,
                "phase": "initial",
                "status": "ok",
            })
            sample_status[sid] = {
                "status": "ok",
                "attempts": 1,
                "last_error": "",
                "injector_flow_cells": int(injector_flow.size),
                "mapping_audit_sets_match": pressure_audit.get("sets_match", "not_run"),
            }
            completed[sid] = payload
            print(
                "[OK] sample {:04d} | injector flow cells={} | pressure max={:.6g} MPa".format(
                    sid, injector_flow.size, np.max(p_stats["max"]) / 1.0e6
                ),
                flush=True,
            )
            if not args.keep_runs:
                shutil.rmtree(str(sample_dir), ignore_errors=True)
        except Exception as exc:
            error_text = str(exc)
            history_entries(history, sid).append({
                "attempt": 1,
                "phase": "initial",
                "status": "failed",
                "error": error_text,
            })
            sample_status[sid] = {
                "status": "failed",
                "attempts": 1,
                "last_error": error_text,
            }
            initial_failures.append(sid)
            print("[FAIL] sample {:04d}: {}".format(sid, exc), file=sys.stderr, flush=True)
            # Failed directories are preserved for diagnosis.
        write_checkpoint_manifest(out_dir, lhs_log10, names, sample_status, args.n_samples)
        save_retry_history(history_path, history)

    # Existing failed samples from the previous 32-sample run enter the retry queue.
    for sid in range(1, args.n_samples + 1):
        if sid in completed:
            continue
        entries = history_entries(history, sid)
        if entries and entries[-1].get("status") == "failed" and sid not in initial_failures:
            initial_failures.append(sid)

    initial_failures = sorted(set(initial_failures))
    print()
    print("Initial failed samples:", initial_failures if initial_failures else "none")

    # -------------------------------------------------------------------------
    # PASS 2/3: automatically retry every failed sample up to max_retries times.
    # -------------------------------------------------------------------------
    retry_failures = initial_failures[:]
    if not args.no_retry_failed and args.max_retries > 0:
        for retry_index in range(1, args.max_retries + 1):
            if not retry_failures:
                break
            print()
            print("=" * 72)
            print("Retry pass {}/{}".format(retry_index, args.max_retries))
            print("Samples:", retry_failures)
            print("=" * 72)
            next_failures: List[int] = []

            for sid in retry_failures:
                if sid in completed:
                    continue

                sample_log10 = lhs_log10[sid - 1]
                k_map = {m: float(10.0 ** sample_log10[j]) for j, m in enumerate(names)}
                entries = history_entries(history, sid)
                attempt_no = len(entries) + 1
                if attempt_no > args.max_retries + 1:
                    next_failures.append(sid)
                    continue

                run_name = "sample_{:04d}_retry{:02d}".format(sid, retry_index)
                try:
                    sample_dir = prepare_named_sample_run_dir(
                        model_dir,
                        out_dir / "runs",
                        run_name,
                        k_map,
                        args.deck_template,
                        args.copy_static,
                    )
                    print(
                        "[RETRY {}/{}] sample {:04d} (attempt {})".format(
                            retry_index, args.max_retries, sid, attempt_no
                        ),
                        flush=True,
                    )
                    run_pflotran(
                        sample_dir,
                        args.pflotran_bin,
                        args.mpiexec,
                        args.nprocs,
                        deck_prefix,
                    )

                    flow_h5 = sample_dir / (deck_prefix + ".h5")
                    geomech_h5 = sample_dir / (deck_prefix + "-geomech.h5")
                    if not flow_h5.exists():
                        raise FileNotFoundError("Flow HDF5 not found after PFLOTRAN run: {}".format(flow_h5))
                    if not geomech_h5.exists():
                        raise FileNotFoundError("Geomechanics HDF5 not found after PFLOTRAN run: {}".format(geomech_h5))

                    mapping_path = sample_dir / "bartlesville_hec_lime_v5_interfaces_median.mapping"
                    injector_vset = sample_dir / PRESSURE_OBSERVATION_VSET
                    p_times, injector_flow, p_stats, pressure_audit = extract_pressure_delta_from_material_id(
                        flow_h5,
                        mapping_path if not args.skip_mapping_audit else None,
                        injector_vset if not args.skip_mapping_audit else None,
                        do_mapping_audit=not args.skip_mapping_audit,
                    )
                    station_paths = {k: sample_dir / v for k, v in STRAIN_OBSERVATION_VSETS.items()}
                    s_times, s_mean, s_std, s_vol, strain_audit = extract_strain_series(
                        geomech_h5,
                        station_paths,
                    )

                    if not np.allclose(p_times, TARGET_TIMES_H, atol=TIME_TOL_H, rtol=0.0):
                        raise RuntimeError("Extracted flow time grid does not match requested waypoints")
                    if not np.allclose(s_times, TARGET_TIMES_H, atol=TIME_TOL_H, rtol=0.0):
                        raise RuntimeError("Extracted geomechanics time grid does not match requested waypoints")

                    payload: Dict[str, np.ndarray] = {
                        "sample_id": np.asarray([sid], dtype=np.int64),
                        "k_log10": sample_log10,
                        "k_values": np.asarray([k_map[m] for m in names], dtype=float),
                        "pressure_times_h": p_times,
                        "injector_flow_cell_ids_0based": injector_flow,
                        "injector_material_id": np.asarray([INJECTION_MATERIAL_ID], dtype=np.int64),
                        "injector_dp_mean_pa": p_stats["mean"],
                        "injector_dp_median_pa": p_stats["median"],
                        "injector_dp_p05_pa": p_stats["p05"],
                        "injector_dp_p95_pa": p_stats["p95"],
                        "injector_dp_min_pa": p_stats["min"],
                        "injector_dp_max_pa": p_stats["max"],
                        "injector_dp_std_pa": p_stats["std"],
                        "strain_times_h": s_times,
                        "strain_mean": s_mean,
                        "strain_std": s_std,
                        "volumetric_strain": s_vol,
                    }
                    np.savez_compressed(
                        str(sample_output_dir / "sample_{:04d}.npz".format(sid)),
                        **payload,
                    )
                    (sample_output_dir / "sample_{:04d}_audit.json".format(sid)).write_text(
                        json.dumps(
                            {"pressure_audit": pressure_audit, "strain_audit": strain_audit, "k_map": k_map},
                            indent=2,
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                        ),
                        encoding="utf-8",
                    )
                    completed[sid] = payload
                    history_entries(history, sid).append({
                        "attempt": attempt_no,
                        "phase": "retry_{}".format(retry_index),
                        "status": "ok",
                    })
                    sample_status[sid] = {
                        "status": "ok_after_retry_{}".format(retry_index),
                        "attempts": attempt_no,
                        "last_error": "",
                        "injector_flow_cells": int(injector_flow.size),
                        "mapping_audit_sets_match": pressure_audit.get("sets_match", "not_run"),
                    }
                    print(
                        "[OK-RETRY] sample {:04d} | attempt={} | injector flow cells={} | pressure max={:.6g} MPa".format(
                            sid, attempt_no, injector_flow.size, np.max(p_stats["max"]) / 1.0e6
                        ),
                        flush=True,
                    )
                    if not args.keep_runs:
                        shutil.rmtree(str(sample_dir), ignore_errors=True)
                except Exception as exc:
                    error_text = str(exc)
                    history_entries(history, sid).append({
                        "attempt": attempt_no,
                        "phase": "retry_{}".format(retry_index),
                        "status": "failed",
                        "error": error_text,
                    })
                    sample_status[sid] = {
                        "status": "failed",
                        "attempts": attempt_no,
                        "last_error": error_text,
                    }
                    next_failures.append(sid)
                    print(
                        "[FAIL-RETRY] sample {:04d} attempt {}: {}".format(sid, attempt_no, exc),
                        file=sys.stderr,
                        flush=True,
                    )
                    # Preserve failed retry directory for diagnosis.

                write_checkpoint_manifest(out_dir, lhs_log10, names, sample_status, args.n_samples)
                save_retry_history(history_path, history)

            retry_failures = sorted(set(next_failures))

    # -------------------------------------------------------------------------
    # Assemble the final dataset from all successful per-sample NPZ files.
    # -------------------------------------------------------------------------
    all_ok = []
    final_failed: List[int] = []
    for sid in range(1, args.n_samples + 1):
        sample_npz = sample_output_dir / "sample_{:04d}.npz".format(sid)
        if sample_npz.exists():
            try:
                with np.load(str(sample_npz), allow_pickle=False) as z:
                    all_ok.append((sid, {k: z[k] for k in z.files}))
            except Exception as exc:
                final_failed.append(sid)
                sample_status[sid] = {
                    "status": "failed_invalid_npz",
                    "attempts": len(history_entries(history, sid)),
                    "last_error": str(exc),
                }
        else:
            final_failed.append(sid)

    all_ok.sort(key=lambda x: x[0])
    if not all_ok:
        raise RuntimeError("No successful surrogate samples were generated")

    master_path = save_master_dataset(out_dir, names, all_ok)

    recovered = sum(
        1
        for sid, entries in ((sid, history_entries(history, sid)) for sid in range(1, args.n_samples + 1))
        if sid not in final_failed and any(e.get("phase", "").startswith("retry_") and e.get("status") == "ok" for e in entries)
    )

    write_checkpoint_manifest(out_dir, lhs_log10, names, sample_status, args.n_samples)
    manifest = out_dir / "sample_manifest.csv"

    metadata = {
        "workflow": "North_Avant_V5_single_continuous_96h_two_way_surrogate_training_with_automatic_retries",
        "deck_template": args.deck_template,
        "deck_prefix": deck_prefix,
        "n_requested": args.n_samples,
        "n_successful": len(all_ok),
        "n_failed_final": len(final_failed),
        "n_initial_failures": len(initial_failures),
        "n_recovered_after_retry": recovered,
        "seed": args.seed,
        "nprocs_per_run": args.nprocs,
        "max_retries": args.max_retries,
        "automatic_retry_enabled": not args.no_retry_failed,
        "materials_sampled": names,
        "log10_target_bounds": LOG10_TARGET_BOUNDS,
        "target_times_h": TARGET_TIMES_H.tolist(),
        "mechanical_baseline": {
            "AVN87_youngs_modulus_pa": avn87_E,
            "AVN87_poissons_ratio": avn87_nu,
            "AVN87_biot_coefficient": avn87_biot,
        },
        "pressure_observation": {
            "authoritative_definition": "flow HDF5 MATERIAL_ID == {}".format(INJECTION_MATERIAL_ID),
            "material_name": "injection_borehole",
            "material_id": INJECTION_MATERIAL_ID,
            "quantity": "LIQUID_PRESSURE(t) - LIQUID_PRESSURE(0) over injection_borehole flow cells",
            "statistics": ["mean", "median", "p05", "p95", "min", "max", "std"],
            "mapping_audit": "independent mechanics-vset to flow mapping check; not the pressure-cell definition",
        },
        "strain_observation_vsets": STRAIN_OBSERVATION_VSETS,
        "strain_component_names": list(STRAIN_COMPONENTS),
        "retry_policy": {
            "description": "Each failed sample is retried up to max_retries additional times using the exact original LHS permeability vector.",
            "max_retries": args.max_retries,
            "attempts_total_max": args.max_retries + 1,
            "failed_samples_final": final_failed,
        },
        "notes": [
            "Only five hydraulic permeability scalars are sampled.",
            "Mechanical properties are inherited unchanged from the deck template.",
            "AVN2, AVN31, and AVN87 remain separate observation stations.",
            "Injector pressure uses MATERIAL_ID == 6 in the flow HDF5.",
            "Pressure Delta-p uses the same injection cells at t=0 and all later times.",
            "An independent mapping audit can verify the 84-cell expectation without defining the pressure observable.",
            "Per-sample NPZ files make the dataset resumable after a walltime interruption.",
            "Failed samples are retried automatically without changing their LHS parameter vector.",
            "Failed attempt directories are preserved for diagnosis; successful attempt directories are removed unless --keep-runs is used.",
        ],
        "retry_history_file": str(history_path),
    }

    (out_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_retry_history(history_path, history)

    print()
    print("=" * 72)
    print("Surrogate dataset generation complete")
    print("=" * 72)
    print("Successful samples : {} / {}".format(len(all_ok), args.n_samples))
    print("Final failed       : {}".format(final_failed if final_failed else "none"))
    print("Recovered by retry : {}".format(recovered))
    print("Master dataset     : {}".format(master_path))
    print("Manifest           : {}".format(manifest))
    print("Metadata           : {}".format(out_dir / "dataset_metadata.json"))
    print("Retry history      : {}".format(history_path))

    if final_failed and not args.allow_failures:
        print(
            "Failures remain after retry policy; rerun with --allow-failures only if a partial dataset is acceptable.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
