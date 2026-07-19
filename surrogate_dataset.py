#!/usr/bin/env python3
"""
Build a PFLOTRAN surrogate-training dataset from a single continuous run.

Workflow:
- one coupled PFLOTRAN deck per sample
- injection active from 0 to 19 h
- shut-in from 19 to 96 h in the same run
- no checkpoint / restart handoff
- Latin hypercube sampling over the four layer permeabilities

This version is Python 3.6 compatible:
- no SciPy dependency
- no from __future__ import annotations
- no X | Y type syntax

Assumptions for the current Bartlesville HEC workflow:
- the working deck template is geomech_inj_rec.in
- the mesh files are bartlesville_hec.uge / bartlesville_hec.ugi
- geomechanics gravity is off
- coupling timestep is 1.d-3 hour
- only overburden, bartlesville_sand, basal_layer, underburden are sampled
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


MATERIALS = ["overburden", "bartlesville_sand", "basal_layer", "underburden"]

# Current baseline permeability tensors from the working deck.
BASE_TENSORS = {
    "overburden": (9.869233e-18, 9.869233e-18, 9.869233e-19),
    "bartlesville_sand": (4.9346165e-15, 4.9346165e-15, 4.9346165e-17),
    "basal_layer": (9.869233e-18, 9.869233e-18, 9.869233e-19),
    "underburden": (9.869233e-18, 9.869233e-18, 9.869233e-19),
}

# Absolute target ranges for the scalar permeability magnitude of each layer.
# The anisotropy in the template deck is preserved by scaling the whole tensor.
LOG10_TARGET_BOUNDS = {
    "overburden": (-18.0, -16.0),
    "bartlesville_sand": (-14.0, -12.0),
    "basal_layer": (-19.0, -17.0),
    "underburden": (-18.0, -16.0),
}

# Wellbore HDF5 indices previously identified for the North Avant / Bartlesville model.
WELLBORE_H5_INDICES = np.array([
    354057, 354058, 354059, 354060, 354061, 354062, 354063, 354064,
    375524, 375525, 375526, 375527, 375528, 375529, 375530, 375531,
    437633, 437634, 437635, 437636, 437637, 437638, 437639, 437640,
    471238, 471239, 471240, 471241, 471242, 471243, 471244, 471245,
], dtype=int)

STRAIN_COMPONENTS = [
    "strain_xx", "strain_yy", "strain_zz",
    "strain_xy", "strain_yz", "strain_zx",
]

PRESSURE_DATASET_CANDIDATES = [
    "LIQUID_PRESSURE",
    "Liquid Pressure [Pa]",
    "Liquid Pressure",
]

STRAIN_DATASET_CANDIDATES = [
    "strain_xx", "strain_yy", "strain_zz",
    "strain_xy", "strain_yz", "strain_zx",
    "GEOMECH_STRAIN_XX", "GEOMECH_STRAIN_YY", "GEOMECH_STRAIN_ZZ",
    "GEOMECH_STRAIN_XY", "GEOMECH_STRAIN_YZ", "GEOMECH_STRAIN_ZX",
    "STRAIN_XX", "STRAIN_YY", "STRAIN_ZZ",
    "STRAIN_XY", "STRAIN_YZ", "STRAIN_ZX",
]

STATIC_FILES = [
    # Flow / mechanics mesh and mapping
    "bartlesville_hec.uge",
    "bartlesville_hec.ugi",
    "bartlesville_hec.mapping",
    "bartlesville_hec_material_ids.h5",

    # Flow boundary face sets
    "top.ex",
    "bottom.ex",
    "north.ex",
    "south.ex",
    "east.ex",
    "west.ex",

    # Geomechanics boundary region sets
    "top.vset",
    "bottom.vset",
    "north.vset",
    "south.vset",
    "east.vset",
    "west.vset",

    # Geomechanics strata and special regions
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PFLOTRAN surrogate dataset from a single continuous coupled run."
    )
    parser.add_argument("--model-dir", type=str, default=".", help="Directory containing the PFLOTRAN input files.")
    parser.add_argument("--out-dir", type=str, default="./surrogate_dataset", help="Output dataset directory.")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of Latin hypercube samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for the LHS sampler.")
    parser.add_argument(
        "--nprocs",
        type=int,
        default=int(os.environ.get("SLURM_NTASKS", "64")),
        help="MPI ranks to use per PFLOTRAN run.",
    )
    parser.add_argument(
        "--pflotran-bin",
        type=str,
        default=os.environ.get("PFLOTRAN_BIN", "pflotran"),
        help="Path to the PFLOTRAN executable.",
    )
    parser.add_argument("--mpiexec", type=str, default="mpiexec", help="MPI launcher command.")
    parser.add_argument(
        "--deck-template",
        type=str,
        default="geomech_inj_rec.in",
        help="Template deck filename to patch for each sample.",
    )
    parser.add_argument(
        "--copy-static",
        action="store_true",
        help="Copy static files instead of symlinking them.",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep sample run directories after successful extraction.",
    )
    return parser.parse_args()


def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(str(path))
    else:
        path.unlink()


def link_or_copy(src: Path, dst: Path, copy_mode: bool) -> None:
    safe_unlink(dst)
    if copy_mode:
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    else:
        os.symlink(str(src.resolve()), str(dst))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def generate_lhs_unit_samples(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    u = np.empty((n_samples, n_dim), dtype=float)

    for j in range(n_dim):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        pts = cut[:-1] + rng.rand(n_samples) * (cut[1:] - cut[:-1])
        rng.shuffle(pts)
        u[:, j] = pts

    return u


def generate_lhs_log10_samples(
    n_samples: int,
    bounds_log10: Dict[str, Tuple[float, float]],
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    names = MATERIALS[:]
    lower = np.array([bounds_log10[m][0] for m in names], dtype=float)
    upper = np.array([bounds_log10[m][1] for m in names], dtype=float)

    unit = generate_lhs_unit_samples(n_samples=n_samples, n_dim=len(names), seed=seed)
    scaled = lower + unit * (upper - lower)
    return scaled, names


def replace_perm_tensor_in_block(text: str, material_name: str,
                                 perm_x: float, perm_y: float, perm_z: float) -> str:
    """
    Replace PERM_X / PERM_Y / PERM_Z inside a MATERIAL_PROPERTY block.
    """
    lines = text.splitlines(True)
    start = None
    mat_pat = re.compile(r"^\s*MATERIAL_PROPERTY\s+{}\s*$".format(re.escape(material_name)))
    end_pat = re.compile(r"^\s*END\s*$")
    perm_x_pat = re.compile(r"^\s*PERM_X\b")
    perm_y_pat = re.compile(r"^\s*PERM_Y\b")
    perm_z_pat = re.compile(r"^\s*PERM_Z\b")

    for i, line in enumerate(lines):
        if mat_pat.match(line):
            start = i
            break

    if start is None:
        raise RuntimeError("Could not find MATERIAL_PROPERTY block for material '{}'.".format(material_name))

    end = None
    for j in range(start + 1, len(lines)):
        if end_pat.match(lines[j]):
            end = j
            break

    if end is None:
        raise RuntimeError("Could not find END for MATERIAL_PROPERTY block '{}'.".format(material_name))

    found_x = found_y = found_z = False
    for k in range(start + 1, end):
        if perm_x_pat.match(lines[k]):
            indent = re.match(r"^(\s*)", lines[k]).group(1)
            lines[k] = "{}PERM_X {:.6e}\n".format(indent, perm_x)
            found_x = True
        elif perm_y_pat.match(lines[k]):
            indent = re.match(r"^(\s*)", lines[k]).group(1)
            lines[k] = "{}PERM_Y {:.6e}\n".format(indent, perm_y)
            found_y = True
        elif perm_z_pat.match(lines[k]):
            indent = re.match(r"^(\s*)", lines[k]).group(1)
            lines[k] = "{}PERM_Z {:.6e}\n".format(indent, perm_z)
            found_z = True

    if not (found_x and found_y and found_z):
        raise RuntimeError(
            "Could not find all PERM_X/PERM_Y/PERM_Z lines for material '{}'.".format(material_name)
        )

    return "".join(lines)


def find_time_groups(h5obj: h5py.File, dataset_candidates: Sequence[str]) -> List[Tuple[float, str]]:
    """
    Find time groups that contain one of the candidate datasets.
    Returns [(time_hours, group_path), ...].
    """
    groups = []
    norm_cands = [normalize_name(c) for c in dataset_candidates]

    def parse_time_from_group_name(group_name: str) -> Optional[float]:
        m = re.search(r"Time\s+([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)\s*h", group_name)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def visitor(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        if "Time" not in name:
            return

        has_candidate = False

        for leaf_name in obj.keys():
            leaf_norm = normalize_name(leaf_name)
            if leaf_norm in norm_cands:
                has_candidate = True
                break

        if not has_candidate:
            def leaf_visitor(subname, subobj):
                nonlocal has_candidate
                if has_candidate:
                    return
                if isinstance(subobj, h5py.Dataset):
                    leaf_norm = normalize_name(Path(subname).name)
                    for cand in norm_cands:
                        if cand == leaf_norm or cand in leaf_norm or leaf_norm in cand:
                            has_candidate = True
                            return
            obj.visititems(leaf_visitor)

        if not has_candidate:
            return

        t = parse_time_from_group_name(Path(name).name)
        if t is not None:
            groups.append((t, name))

    h5obj.visititems(visitor)

    dedup = {}
    for t, p in groups:
        dedup[p] = t

    return sorted([(t, p) for p, t in dedup.items()], key=lambda x: x[0])


def find_dataset_in_group(group: h5py.Group, candidates: Sequence[str]) -> np.ndarray:
    """
    Return the first matching dataset inside a group (recursive search).
    """
    norm_cands = [normalize_name(c) for c in candidates]
    found = None

    def visitor(name, obj):
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset):
            leaf_norm = normalize_name(Path(name).name)
            for cand in norm_cands:
                if cand == leaf_norm or cand in leaf_norm or leaf_norm in cand:
                    found = np.asarray(obj, dtype=float)
                    return

    group.visititems(visitor)

    if found is None:
        raise KeyError("None of the candidate datasets were found: {}".format(candidates))
    return found


def compute_well_stats_at_time(arr: np.ndarray, well_idx: np.ndarray) -> Dict[str, float]:
    values = arr[well_idx]
    return {
        "median": float(np.nanmedian(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def extract_pressure_series(h5_path: Path, well_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(h5_path), "r") as f:
        groups = find_time_groups(f, PRESSURE_DATASET_CANDIDATES)
        if not groups:
            raise RuntimeError("No pressure time groups found in {}".format(h5_path))

        times = []
        med = []
        pmin = []
        pmax = []

        for t, group_path in groups:
            grp = f[group_path]
            pressure = find_dataset_in_group(grp, PRESSURE_DATASET_CANDIDATES)
            if well_idx.max() >= len(pressure):
                raise IndexError(
                    "Well index out of bounds for pressure array in {} at time {} h.".format(h5_path, t)
                )
            stats = compute_well_stats_at_time(pressure, well_idx)
            times.append(t)
            med.append(stats["median"])
            pmin.append(stats["min"])
            pmax.append(stats["max"])

    return (
        np.asarray(times, dtype=float),
        np.asarray(med, dtype=float),
        np.asarray(pmin, dtype=float),
        np.asarray(pmax, dtype=float),
    )


def extract_geomech_series(h5_path: Path, well_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(h5_path), "r") as f:
        groups = find_time_groups(f, STRAIN_DATASET_CANDIDATES)
        if not groups:
            raise RuntimeError("No geomechanics time groups found in {}".format(h5_path))

        times = []
        med_strains = []
        vol_strain = []

        for t, group_path in groups:
            grp = f[group_path]
            comp_vals = []
            for comp in STRAIN_COMPONENTS:
                candidates = [
                    comp,
                    comp.upper(),
                    comp.replace("_", " "),
                    "GEOMECH_" + comp.upper(),
                    "GEOMECH_" + comp.upper().replace("_", " "),
                ]
                arr = find_dataset_in_group(grp, candidates)
                if well_idx.max() >= len(arr):
                    raise IndexError(
                        "Well index out of bounds for strain array in {} at time {} h.".format(h5_path, t)
                    )
                comp_vals.append(float(np.nanmedian(arr[well_idx])))

            comp_vals_arr = np.asarray(comp_vals, dtype=float)
            times.append(t)
            med_strains.append(comp_vals_arr)
            vol_strain.append(float(comp_vals_arr[0] + comp_vals_arr[1] + comp_vals_arr[2]))

    return (
        np.asarray(times, dtype=float),
        np.asarray(med_strains, dtype=float),
        np.asarray(vol_strain, dtype=float),
    )


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
            raise FileNotFoundError("Missing required input file: {}".format(src))
        link_or_copy(src, sample_dir / fname, copy_static)

    deck_src = model_dir / deck_template_name
    if not deck_src.exists():
        raise FileNotFoundError("Missing deck template: {}".format(deck_src))

    deck_text = deck_src.read_text(encoding="utf-8")

    for material in MATERIALS:
        base_x, base_y, base_z = BASE_TENSORS[material]
        base_scalar = base_x  # preserve anisotropy by scaling the whole tensor

        target_k = k_map[material]
        scale = target_k / base_scalar
        new_x = base_x * scale
        new_y = base_y * scale
        new_z = base_z * scale

        deck_text = replace_perm_tensor_in_block(deck_text, material, new_x, new_y, new_z)

    write_text(sample_dir / "pflotran.in", deck_text)
    return sample_dir


def run_pflotran(run_dir: Path, pflotran_bin: str, mpiexec: str, nprocs: int) -> None:
    cmd = [mpiexec, "-n", str(nprocs), pflotran_bin]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Keep Anaconda available for Python packages, but remove it from LD_LIBRARY_PATH
    # so PFLOTRAN/PETSc does not pick up the wrong libstdc++.
    ld = env.get("LD_LIBRARY_PATH", "")
    if ld:
        parts = [p for p in ld.split(":") if "anaconda3" not in p]
        env["LD_LIBRARY_PATH"] = ":".join(parts)

    subprocess.run(cmd, cwd=str(run_dir), check=True, env=env)


def read_sample_outputs(sample_dir: Path, well_idx: np.ndarray) -> Dict[str, np.ndarray]:
    flow_h5 = sample_dir / "pflotran.h5"
    geomech_h5 = sample_dir / "pflotran-geomech.h5"

    if not flow_h5.exists():
        raise FileNotFoundError("Flow output missing: {}".format(flow_h5))
    if not geomech_h5.exists():
        raise FileNotFoundError("Geomechanics output missing: {}".format(geomech_h5))

    t_p, p_med, p_min, p_max = extract_pressure_series(flow_h5, well_idx)
    t_s, s_med, ev = extract_geomech_series(geomech_h5, well_idx)

    return {
        "pressure_times": t_p,
        "pressure_median": p_med,
        "pressure_min": p_min,
        "pressure_max": p_max,
        "strain_times": t_s,
        "strain_median": s_med,
        "volumetric_strain": ev,
    }


def main() -> int:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_root = out_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    lhs_log10, names = generate_lhs_log10_samples(args.n_samples, LOG10_TARGET_BOUNDS, args.seed)

    k_log10_all = []
    k_all = []
    pressure_times_ref = None
    strain_times_ref = None
    pressure_median_all = []
    pressure_min_all = []
    pressure_max_all = []
    strain_median_all = []
    volumetric_strain_all = []

    manifest_rows = []
    failures = []

    for i in range(args.n_samples):
        sample_id = i + 1
        sample_log10 = lhs_log10[i]
        k_map = {mat: float(10.0 ** sample_log10[j]) for j, mat in enumerate(names)}

        sample_dir = prepare_sample_run_dir(
            model_dir=model_dir,
            run_root=run_root,
            sample_id=sample_id,
            k_map=k_map,
            deck_template_name=args.deck_template,
            copy_static=args.copy_static,
        )

        try:
            run_pflotran(sample_dir, args.pflotran_bin, args.mpiexec, args.nprocs)
            obs = read_sample_outputs(sample_dir, WELLBORE_H5_INDICES)

            if pressure_times_ref is None:
                pressure_times_ref = obs["pressure_times"]
            elif not np.allclose(pressure_times_ref, obs["pressure_times"]):
                raise RuntimeError("Pressure time grid changed across samples. Check deck output times.")

            if strain_times_ref is None:
                strain_times_ref = obs["strain_times"]
            elif not np.allclose(strain_times_ref, obs["strain_times"]):
                raise RuntimeError("Geomechanics time grid changed across samples. Check deck output times.")

            k_log10_all.append(sample_log10)
            k_all.append(np.array([k_map[m] for m in MATERIALS], dtype=float))
            pressure_median_all.append(obs["pressure_median"])
            pressure_min_all.append(obs["pressure_min"])
            pressure_max_all.append(obs["pressure_max"])
            strain_median_all.append(obs["strain_median"])
            volumetric_strain_all.append(obs["volumetric_strain"])

            manifest_rows.append({
                "sample_id": sample_id,
                "status": "ok",
                "overburden_k": k_map["overburden"],
                "bartlesville_sand_k": k_map["bartlesville_sand"],
                "basal_layer_k": k_map["basal_layer"],
                "underburden_k": k_map["underburden"],
                "run_dir": str(sample_dir),
            })
            print("[OK] sample {:04d}".format(sample_id))

            if not args.keep_runs:
                shutil.rmtree(str(sample_dir), ignore_errors=True)

        except Exception as e:
            failures.append((sample_id, str(e)))
            manifest_rows.append({
                "sample_id": sample_id,
                "status": "failed: {}".format(e),
                "overburden_k": k_map.get("overburden", np.nan),
                "bartlesville_sand_k": k_map.get("bartlesville_sand", np.nan),
                "basal_layer_k": k_map.get("basal_layer", np.nan),
                "underburden_k": k_map.get("underburden", np.nan),
                "run_dir": str(sample_dir),
            })
            print("[FAIL] sample {:04d}: {}".format(sample_id, e), file=sys.stderr)

    if not k_log10_all:
        raise RuntimeError("No successful samples were generated.")

    k_log10_arr = np.asarray(k_log10_all, dtype=float)
    k_arr = np.asarray(k_all, dtype=float)
    pressure_median_arr = np.asarray(pressure_median_all, dtype=float)
    pressure_min_arr = np.asarray(pressure_min_all, dtype=float)
    pressure_max_arr = np.asarray(pressure_max_all, dtype=float)
    strain_median_arr = np.asarray(strain_median_all, dtype=float)
    volumetric_strain_arr = np.asarray(volumetric_strain_all, dtype=float)

    np.savez_compressed(
        out_dir / "dataset_master.npz",
        material_names=np.array(MATERIALS, dtype="U"),
        pressure_times=pressure_times_ref,
        strain_times=strain_times_ref,
        k_log10=k_log10_arr,
        k_values=k_arr,
        pressure_median=pressure_median_arr,
        pressure_min=pressure_min_arr,
        pressure_max=pressure_max_arr,
        strain_median=strain_median_arr,
        volumetric_strain=volumetric_strain_arr,
    )

    manifest_path = out_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id", "status",
                "overburden_k", "bartlesville_sand_k", "basal_layer_k", "underburden_k",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    meta = {
        "workflow": "single_continuous_run_injection_to_shutin",
        "deck_template": args.deck_template,
        "n_requested": args.n_samples,
        "n_successful": int(len(k_log10_arr)),
        "n_failed": int(len(failures)),
        "failures": [{"sample_id": sid, "error": err} for sid, err in failures],
        "materials": MATERIALS,
        "pressure_times": pressure_times_ref.tolist() if pressure_times_ref is not None else None,
        "strain_times": strain_times_ref.tolist() if strain_times_ref is not None else None,
        "wellbore_h5_indices": WELLBORE_H5_INDICES.tolist(),
        "base_tensor_values": {
            m: list(BASE_TENSORS[m]) for m in MATERIALS
        },
        "log10_target_bounds": LOG10_TARGET_BOUNDS,
        "notes": [
            "One coupled PFLOTRAN run per sample.",
            "Injection is active from 0 to 19 h, then set to zero through 96 h.",
            "The four sampled parameters are the layer permeabilities only.",
            "Current deck template: geomech_inj_rec.in.",
        ],
    }
    (out_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print("Done.")
    print("Successful samples: {} / {}".format(len(k_log10_arr), args.n_samples))
    print("Dataset: {}".format(out_dir / "dataset_master.npz"))
    print("Manifest: {}".format(manifest_path))
    print("Metadata: {}".format(out_dir / "dataset_metadata.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())