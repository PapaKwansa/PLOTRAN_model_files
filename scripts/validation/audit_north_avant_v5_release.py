#!/usr/bin/env python3
"""
Read-only release audit for the North Avant V5 PFLOTRAN production pipeline.

This script never deletes, moves, or edits project files. It verifies:
  * required production/source/postprocessing files;
  * required runtime-bundle files from the authoritative manifest;
  * Python syntax and shell syntax;
  * production/preproduction deck identity and core tokens;
  * UGE/UGI/mapping/material-HDF5 count consistency;
  * vset ranges and external-boundary EX positivity;
  * duplicate/ambiguous filenames that should be retired;
  * Git status, tracked generated artifacts, and large tracked files.

Run from the repository root:
    .venv-postprocess/bin/python audit_north_avant_v5_release.py

If the frozen runtime bundle already exists:
    .venv-postprocess/bin/python audit_north_avant_v5_release.py \
      --bundle north_avant_v5_palmetto_bundle

Outputs:
    validation/release_audit/north_avant_v5_release_audit.txt
    validation/release_audit/north_avant_v5_release_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    import h5py
except ImportError:
    h5py = None

try:
    import numpy as np
except ImportError:
    np = None


EXPECTED_RUNTIME = [
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
    "overburden.vset",
    "shallow_limestone.vset",
    "bartlesville_sand.vset",
    "basal_layer.vset",
    "underburden.vset",
    "hec.vset",
    "injection_borehole.vset",
    "strainmeter_sensors.vset",
    "AVN2.vset",
    "AVN87.vset",
    "AVN31.vset",
    "top.vset",
    "bottom.vset",
    "north.vset",
    "south.vset",
    "east.vset",
    "west.vset",
]

REQUIRED_RELEASE = [
    "north_avant_v5_twoway_preproduction_4h.in",
    "north_avant_v5_twoway_production_96h_final.in",
    "run_north_avant_v5_simulation.slurm",
    "postprocess_north_avant_v5_results.slurm",
    "submit_north_avant_v5_pipeline.sh",
    "pflotran_coupled_to_vtu.py",
    "pflotran_region_timeseries_plots.py",
    "preflight_north_avant_v5_bundle.py",
    "north_avant_v5_runtime_manifest.txt",
    "requirements-postprocess.txt",
    "NORTH_AVANT_V5_PRODUCTION_PHYSICS.md",
    "README_PIPELINE.md",
]

REQUIRED_MESH_SOURCE = [
    "build_poly_layers4.py",
    "layers4_get_material_boundary_tags.py",
    "tetgen_quality_report_localized.py",
    "tetgen_to_avs_ugi_canonical.py",
    "material_h5_from_txt.py",
    "validate_uge_and_write_mapping.py",
    "build_boundary_ex.py",
]

REQUIRED_VALIDATION_SOURCE = [
    "compare_oneway_twoway.py",
    "compare_continuous_restart.py",
    "restart_acceptance_gate.py",
    "audit_xmf_h5_v2.py",
]

AMBIGUOUS_OR_SUPERSEDED = [
    "poroelastic.sh",
    "run_north_avant_v5_simulations.slurm",
    "postprocess_north_avant_v5_results.sh",
    "audit_xmf_h5.py",
    "pflotran_geomech_to_vtu.py",
    "pflotran_strainmeter_timeseries.py",
    "workflow.py",
    "workflow_mine.py",
    "build_poly_layers4_mine.py",
    "layers4_get_material_boundary_tags_mine.py",
    "convert_vset_to_ex.py",
    "generate_ugi.py",
    "mapping.py",
    "px.py",
    "h5_outputs.py",
    "input_validation.py",
    "xdmf_outputs.py",
    "delete_node_row.py",
    "delete_ugi_row.py",
    "swept_mesh.py",
]

GENERATED_SUFFIXES = {
    ".h5", ".xmf", ".vtu", ".pvd", ".vtp", ".pvtp",
    ".node", ".ele", ".face", ".edge", ".neigh", ".poly",
    ".uge", ".ugi", ".mapping", ".trn", ".inp", ".chk",
    ".out", ".err", ".log",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Audit:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.hard_failures = 0
        self.warnings = 0

    def pass_(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "PASS", detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "WARN", detail))
        self.warnings += 1

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "FAIL", detail))
        self.hard_failures += 1


def run_capture(command: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def noncomment_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def first_header(path: Path) -> list[str]:
    return next(iter(noncomment_lines(path))).split()


def read_manifest(path: Path) -> list[str]:
    return [line for line in noncomment_lines(path)]


def read_uge_counts(path: Path) -> tuple[int, int]:
    lines = iter(noncomment_lines(path))
    header = next(lines).split()
    if len(header) < 2 or header[0].upper() != "CELLS":
        raise RuntimeError("invalid CELLS header")
    cells = int(header[1])
    for _ in range(cells):
        next(lines)
    conn = next(lines).split()
    if len(conn) < 2 or conn[0].upper() != "CONNECTIONS":
        raise RuntimeError("invalid CONNECTIONS header")
    return cells, int(conn[1])


def read_ugi_counts(path: Path) -> tuple[int, int]:
    header = first_header(path)
    if len(header) < 2:
        raise RuntimeError("invalid UGI header")
    return int(header[0]), int(header[1])


def read_mapping(path: Path) -> tuple[int, bool]:
    if np is None:
        rows = []
        for line in noncomment_lines(path):
            fields = line.split()
            if len(fields) < 2:
                raise RuntimeError("mapping row has fewer than two fields")
            rows.append((int(fields[0]), int(fields[1])))
        identity = all(a == i and b == i for i, (a, b) in enumerate(rows, start=1))
        return len(rows), identity

    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError("mapping has fewer than two columns")
    expected = np.arange(1, data.shape[0] + 1, dtype=np.int64)
    identity = np.array_equal(data[:, 0], expected) and np.array_equal(data[:, 1], expected)
    return int(data.shape[0]), bool(identity)


def check_vset(path: Path, maximum_id: int) -> int:
    ids: list[int] = []
    for line in noncomment_lines(path):
        value = int(line.split()[0])
        if value < 1 or value > maximum_id:
            raise RuntimeError(f"ID {value} outside 1..{maximum_id}")
        ids.append(value)
    if not ids:
        raise RuntimeError("empty vset")
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate vset IDs")
    return len(ids)


def check_ex(path: Path, maximum_id: int) -> tuple[int, float]:
    lines = iter(noncomment_lines(path))
    header = next(lines).split()
    if len(header) < 2 or header[0].upper() != "CONNECTIONS":
        raise RuntimeError("invalid EX header")
    expected = int(header[1])
    count = 0
    area_sum = 0.0
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError("EX row has fewer than five fields")
        cell = int(fields[0])
        area = float(fields[4])
        if cell < 1 or cell > maximum_id:
            raise RuntimeError(f"cell ID {cell} outside 1..{maximum_id}")
        if not (area > 0.0):
            raise RuntimeError(f"nonpositive area {area}")
        count += 1
        area_sum += area
    if count != expected:
        raise RuntimeError(f"header count {expected}, parsed {count}")
    return count, area_sum


def check_python_syntax(path: Path) -> tuple[bool, str]:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True, "syntax compiled"
    except Exception as exc:
        return False, repr(exc)


def file_candidates(root: Path, relative: str) -> list[Path]:
    # Support both a flat development checkout and the clean release layout:
    # decks/, slurm/, scripts/{mesh,validation,postprocess}/, and docs/.
    alternatives = [root / relative]

    if relative.endswith(".in") and relative.startswith("north_avant_v5_"):
        alternatives += [root / "decks" / relative]

    if relative in {"README_PIPELINE.md", "NORTH_AVANT_V5_PRODUCTION_PHYSICS.md"}:
        alternatives += [root / "docs" / relative]

    if relative.endswith(".slurm"):
        alternatives += [root / "slurm" / relative]

    if relative in {"pflotran_coupled_to_vtu.py", "pflotran_region_timeseries_plots.py"}:
        alternatives += [root / "scripts" / "postprocess" / relative]

    if relative in {
        "preflight_north_avant_v5_bundle.py",
        "compare_oneway_twoway.py",
        "compare_continuous_restart.py",
        "restart_acceptance_gate.py",
        "audit_xmf_h5_v2.py",
    }:
        alternatives += [root / "scripts" / "validation" / relative]

    if relative in {
        "build_poly_layers4.py",
        "layers4_get_material_boundary_tags.py",
        "tetgen_quality_report_localized.py",
        "tetgen_to_avs_ugi_canonical.py",
        "material_h5_from_txt.py",
        "validate_uge_and_write_mapping.py",
        "build_boundary_ex.py",
    }:
        alternatives += [root / "scripts" / "mesh" / relative]

    return alternatives


def locate(root: Path, relative: str) -> Path | None:
    for candidate in file_candidates(root, relative):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def require_files(audit: Audit, root: Path, files: list[str], group: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in files:
        path = locate(root, name)
        if path is None:
            missing.append(name)
        else:
            found[name] = path
    if missing:
        audit.fail(group, "missing: " + ", ".join(missing))
    else:
        audit.pass_(group, f"{len(files)} required files present")
    return found


def check_deck(audit: Audit, path: Path, production: bool) -> None:
    text = path.read_text(encoding="utf-8")
    label = path.name
    expected_final = "FINAL_TIME 96.d0 hour" if production else "FINAL_TIME 4.d0 hour"
    tokens = [
        "FLOW_COUPLING TWO_WAY_COUPLED",
        expected_final,
        "INTERPOLATION LINEAR",
        "SNES_TYPE NTR",
        "DTOL 1.d3",
        "PRESSURE_CHANGE_GOVERNOR 5.d4",
        "SATURATION_CHANGE_GOVERNOR 2.d-2",
        "TIMESTEP_REDUCTION_FACTOR 2.5d-1",
        "TIMESTEP_MAXIMUM_GROWTH_FACTOR 1.2d0",
        "NUM_STEPS_AFTER_TS_CUT 10",
        "COUPLING_TIMESTEP_SIZE 5.d-3 hour",
        "bartlesville_hec_lime_v5_interfaces_median.uge",
        "bartlesville_hec_lime_v5_interfaces.ugi",
        "bartlesville_hec_lime_v5_interfaces_median.mapping",
        "bartlesville_hec_lime_v5_interfaces_material_ids.h5",
        "shallow_limestone",
        "GEOMECHANICS_OUTPUT",
        "FORMAT HDF5",
    ]
    missing = [token for token in tokens if token not in text]
    if missing:
        audit.fail(f"deck:{label}", "missing tokens: " + ", ".join(missing))
    else:
        audit.pass_(f"deck:{label}", f"two-way, LINEAR ramp schedule, stabilized controls, {expected_final}, V5 paths, flow snapshots, default geomechanics output, checkpoints disabled")

    # NAV5 NO-MIDRUN-CHECKPOINT CONTRACT
    active_text = "\n".join(
        raw.split("#", 1)[0]
        for raw in text.splitlines()
    )

    if re.search(
        r"(?m)^[ \t]*CHECKPOINT[ \t]*$",
        active_text,
    ):
        audit.fail(
            f"deck:{label}:checkpoint",
            "contains an active CHECKPOINT block; parallel HDF5 restart writes are disabled",
        )

    geomech_output = re.search(
        r"(?ms)"
        r"^[ \t]*GEOMECHANICS_OUTPUT[ \t]*$"
        r".*?"
        r"^[ \t]*END[ \t]*$",
        active_text,
    )

    if geomech_output is None:
        audit.fail(
            f"deck:{label}:geomechanics-output",
            "GEOMECHANICS_OUTPUT block is missing",
        )
    elif re.search(
        r"(?m)^[ \t]*TIMES[ \t]+",
        geomech_output.group(0),
    ):
        audit.fail(
            f"deck:{label}:geomechanics-times",
            "GEOMECHANICS_OUTPUT contains explicit TIMES",
        )

    if "ONE_WAY_COUPLED" in text:
        audit.fail(f"deck:{label}:oneway", "contains ONE_WAY_COUPLED")
    if re.search(r"\bbartlesville_hec\.(uge|ugi|mapping)\b", text):
        audit.fail(f"deck:{label}:old-grid", "contains old bartlesville_hec grid/mapping reference")


def git_inventory(audit: Audit, root: Path) -> dict[str, object]:
    info: dict[str, object] = {}
    rc, top = run_capture(["git", "rev-parse", "--show-toplevel"], root)
    if rc != 0:
        audit.warn("git repository", "not a Git checkout")
        return info

    git_root = Path(top.strip()).resolve()
    info["git_root"] = str(git_root)
    rc, head = run_capture(["git", "rev-parse", "HEAD"], root)
    info["head"] = head.strip() if rc == 0 else "unknown"
    rc, status = run_capture(["git", "status", "--short", "--untracked-files=all"], root)
    info["status"] = status.splitlines()
    if status.strip():
        audit.warn("git working tree", f"{len(status.splitlines())} changed/untracked paths")
    else:
        audit.pass_("git working tree", "clean")

    rc, tracked = run_capture(["git", "ls-files", "-z"], root)
    tracked_paths = [item for item in tracked.split("\0") if item] if rc == 0 else []
    info["tracked_count"] = len(tracked_paths)

    tracked_generated: list[str] = []
    tracked_large: list[dict[str, object]] = []
    for rel in tracked_paths:
        path = git_root / rel
        if path.suffix.lower() in GENERATED_SUFFIXES:
            tracked_generated.append(rel)
        if path.is_file() and path.stat().st_size >= 50 * 1024 * 1024:
            tracked_large.append({"path": rel, "bytes": path.stat().st_size})

    info["tracked_generated"] = tracked_generated
    info["tracked_large"] = tracked_large

    if tracked_large:
        audit.warn(
            "large tracked files",
            ", ".join(f"{item['path']} ({item['bytes']/1024**2:.1f} MiB)" for item in tracked_large),
        )
    else:
        audit.pass_("large tracked files", "none >= 50 MiB")

    if tracked_generated:
        audit.warn("tracked generated artifacts", f"{len(tracked_generated)} generated/binary paths tracked")
    else:
        audit.pass_("tracked generated artifacts", "none detected")

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit North Avant V5 release readiness.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Frozen runtime bundle; default uses <repo>/north_avant_v5_palmetto_bundle if present, otherwise repo root.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("validation/release_audit"),
    )
    args = parser.parse_args()

    root = args.repo.expanduser().resolve()
    report_dir = (root / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    audit = Audit()

    release = require_files(audit, root, REQUIRED_RELEASE, "release pipeline files")
    require_files(audit, root, REQUIRED_MESH_SOURCE, "V5 mesh source files")
    require_files(audit, root, REQUIRED_VALIDATION_SOURCE, "validation source files")

    for key in ("north_avant_v5_twoway_preproduction_4h.in", "north_avant_v5_twoway_production_96h_final.in"):
        path = release.get(key)
        if path:
            check_deck(audit, path, production="96h" in key)

    # Syntax checks.
    for name, path in release.items():
        if path.suffix == ".py":
            ok, detail = check_python_syntax(path)
            (audit.pass_ if ok else audit.fail)(f"python syntax:{name}", detail)
        elif path.suffix in {".sh", ".slurm"}:
            rc, output = run_capture(["bash", "-n", str(path)], root)
            if rc == 0:
                audit.pass_(f"shell syntax:{name}", "bash -n passed")
            else:
                audit.fail(f"shell syntax:{name}", output.strip())

    for name in REQUIRED_MESH_SOURCE + REQUIRED_VALIDATION_SOURCE:
        path = locate(root, name)
        if path:
            ok, detail = check_python_syntax(path)
            (audit.pass_ if ok else audit.fail)(f"python syntax:{name}", detail)

    # Manifest and bundle.
    manifest_path = release.get("north_avant_v5_runtime_manifest.txt")
    manifest_entries: list[str] = []
    if manifest_path:
        manifest_entries = read_manifest(manifest_path)
        missing_expected = sorted(set(EXPECTED_RUNTIME) - set(manifest_entries))
        extra = sorted(set(manifest_entries) - set(EXPECTED_RUNTIME))
        if missing_expected or extra:
            audit.fail(
                "runtime manifest contents",
                f"missing={missing_expected}; extra={extra}",
            )
        elif len(manifest_entries) != len(set(manifest_entries)):
            audit.fail("runtime manifest contents", "duplicate entries")
        else:
            audit.pass_("runtime manifest contents", f"exact authoritative {len(EXPECTED_RUNTIME)}-file manifest")

    default_bundle = root / "north_avant_v5_palmetto_bundle"
    if args.bundle is not None:
        bundle = args.bundle.expanduser()
        if not bundle.is_absolute():
            bundle = root / bundle
        bundle = bundle.resolve()
    else:
        bundle = default_bundle if default_bundle.is_dir() else root

    runtime_missing = [name for name in EXPECTED_RUNTIME if not (bundle / name).is_file() or (bundle / name).stat().st_size == 0]
    if runtime_missing:
        audit.fail("runtime bundle files", f"bundle={bundle}; missing: " + ", ".join(runtime_missing))
    else:
        audit.pass_("runtime bundle files", f"all {len(EXPECTED_RUNTIME)} files present in {bundle}")

    for deck_name in ("north_avant_v5_twoway_preproduction_4h.in", "north_avant_v5_twoway_production_96h_final.in"):
        if not (bundle / deck_name).is_file():
            audit.warn("bundle deck", f"{deck_name} is not yet copied into {bundle}")

    for pp in ("scripts/postprocess/pflotran_coupled_to_vtu.py", "scripts/postprocess/pflotran_region_timeseries_plots.py"):
        if not (bundle / pp).is_file():
            audit.warn("bundle postprocessor", f"{pp} is not yet copied into {bundle}")

    # Runtime consistency checks.
    if not runtime_missing:
        try:
            cells, connections = read_uge_counts(bundle / EXPECTED_RUNTIME[0])
            tetrahedra, vertices = read_ugi_counts(bundle / EXPECTED_RUNTIME[1])
            mapping_rows, identity = read_mapping(bundle / EXPECTED_RUNTIME[2])
            if cells == vertices == mapping_rows == 140456 and tetrahedra == 802245 and connections == 959340 and identity:
                audit.pass_(
                    "runtime topology counts",
                    f"cells={cells}, connections={connections}, vertices={vertices}, tets={tetrahedra}, identity mapping",
                )
            else:
                audit.fail(
                    "runtime topology counts",
                    f"cells={cells}, connections={connections}, vertices={vertices}, tets={tetrahedra}, mapping_rows={mapping_rows}, identity={identity}",
                )

            if h5py is None:
                audit.warn("material HDF5", "h5py unavailable; skipped")
            else:
                with h5py.File(bundle / EXPECTED_RUNTIME[3], "r") as h5:
                    ids = h5["/Materials/Cell Ids"]
                    mats = h5["/Materials/Material Ids"]
                    observed = sorted({int(value) for value in mats[...]})
                    if ids.shape == (cells,) and mats.shape == (cells,) and observed == list(range(1, 11)):
                        audit.pass_("material HDF5", f"{cells} rows; material IDs 1..10")
                    else:
                        audit.fail("material HDF5", f"shapes={ids.shape},{mats.shape}; IDs={observed}")

            vset_details = []
            for rel in EXPECTED_RUNTIME:
                if rel.endswith(".vset"):
                    count = check_vset(bundle / rel, vertices)
                    vset_details.append(f"{rel}:{count}")
            audit.pass_("vset validation", "; ".join(vset_details))

            ex_details = []
            for rel in EXPECTED_RUNTIME:
                if rel.endswith(".ex"):
                    count, area_sum = check_ex(bundle / rel, cells)
                    ex_details.append(f"{rel}:{count},A={area_sum:.6e}")
            audit.pass_("boundary EX validation", "; ".join(ex_details))
        except Exception as exc:
            audit.fail("runtime consistency", repr(exc))

    # Ambiguities / leftovers.
    leftovers = [name for name in AMBIGUOUS_OR_SUPERSEDED if (root / name).exists()]
    if leftovers:
        audit.warn("ambiguous or superseded files", ", ".join(leftovers))
    else:
        audit.pass_("ambiguous or superseded files", "none at repository root")

    dangerous = []
    for pattern in (
        "bartlesville_hec_lime_v2*",
        "bartlesville_hec_lime_v3*",
        "bartlesville_hec_lime_v4*",
        "*UNVALIDATED*",
        "voronoi_mesh_proc*.vtp",
        "north_avant_v5_*smoke*.h5",
        "north_avant_v5_*restart_stage*.h5",
    ):
        dangerous.extend(str(path.relative_to(root)) for path in root.glob(pattern))
    dangerous = sorted(set(dangerous))
    if dangerous:
        audit.warn("generated/legacy artifacts still in root", f"{len(dangerous)} paths; archive before release")
    else:
        audit.pass_("generated/legacy artifacts still in root", "none detected")

    git_info = git_inventory(audit, root)

    # Required-file hashes.
    hashes = {}
    for name, path in sorted(release.items()):
        hashes[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    result = {
        "repo": str(root),
        "bundle": str(bundle),
        "hard_failures": audit.hard_failures,
        "warnings": audit.warnings,
        "checks": [asdict(check) for check in audit.checks],
        "release_file_hashes": hashes,
        "git": git_info,
    }

    json_path = report_dir / "north_avant_v5_release_audit.json"
    txt_path = report_dir / "north_avant_v5_release_audit.txt"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "NORTH AVANT V5 RELEASE AUDIT",
        "=" * 100,
        f"Repository: {root}",
        f"Runtime bundle: {bundle}",
        "",
    ]
    for check in audit.checks:
        lines.append(f"{check.status:4s}  {check.name}")
        lines.append(f"      {check.detail}")
    lines += [
        "",
        "=" * 100,
        f"Hard failures: {audit.hard_failures}",
        f"Warnings:      {audit.warnings}",
        f"Result:        {'PASS' if audit.hard_failures == 0 else 'FAIL'}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")

    return 0 if audit.hard_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
