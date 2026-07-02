#!/usr/bin/env python3
"""One-command North Avant / Bartlesville HEC + local borehole-mesh-zone workflow.

Run from this folder:
    python3 workflow.py

The HEC remains material-ID 5 tag-only geometry. It is not a TetGen PLC
surface or region. Five boreholes are locally resolved solid material-tag mesh zones created from smooth cylindrical-shell Part-1 point fields.
They are not hollow, not Part-3 TetGen holes, and have no internal PLC shell facets.

The grey injection borehole reaches the HEC. The four green strainmeters are
outside the HEC footprint and occupy only the top 100 m of the domain.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# Update only these paths if your local installation differs.
# Update only these paths if your local installation differs.
TETGEN_EXE = os.environ.get(
    "TETGEN_EXE",
    "/home/kwesi/Tetgen/build/tetgen",
)

VORONOI_EXE = os.environ.get(
    "VORONOI_EXE",
    "/home/kwesi/voronoi/src/voronoi",
)

MESH_NAME = "bartlesville_hec"
SCRIPT_DIR = Path(__file__).resolve().parent
TETGEN_DIAGNOSE = False
SKIP_PX = False
KEEP_OLD_OUTPUTS = False
MAX_TETRAHEDRA = int(os.environ.get("BARTLESVILLE_MAX_TETS", "6000000"))

VSET_FILES: List[str] = [
    "top.vset", "bottom.vset", "north.vset", "south.vset", "east.vset", "west.vset",
    "overburden.vset", "bartlesville_sand.vset", "basal_layer.vset", "underburden.vset",
    "hec.vset",
    "injection_borehole.vset",
    "strainmeter_1.vset", "strainmeter_2.vset", "strainmeter_3.vset", "strainmeter_4.vset",
    "strainmeter_boreholes.vset", "boreholes.vset",
]

# The last four helpers come from your existing PFLOTRAN workflow.
REQUIRED_HELPERS: List[str] = [
    "build_poly_layers4.py",
    "layers4_get_material_boundary_tags.py",
    "tetgen_to_avs.py",
    "material_h5_from_txt.py",
    "convert_vset_to_ex.py",
    "generate_ugi.py",
    "mapping.py",
]


def nonempty(path: Path, min_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def require_executable(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} executable was not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")
    return path


def require_helpers() -> None:
    missing = [name for name in REQUIRED_HELPERS if not (SCRIPT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing helper scripts:\n  " + "\n  ".join(missing))


def run(command: Iterable[object]) -> None:
    values = [str(value) for value in command]
    print("[CMD]", " ".join(values))
    subprocess.run(values, check=True)


def run_voronoi(command: Iterable[object], log_path: Path) -> None:
    values = [str(value) for value in command]
    print("[CMD]", " ".join(values))
    result = subprocess.run(values, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"VORONOI failed with exit code {result.returncode}. See {log_path.name}.")


def normalize_tetgen_ascii(path: Path) -> None:
    """Remove TetGen comments for legacy helper scripts that need line-1 headers."""
    if not path.is_file():
        return
    kept: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                kept.append(line + "\n")
    if not kept:
        raise RuntimeError(f"No numeric TetGen data found in {path}")
    path.write_text("".join(kept), encoding="utf-8")
    print(f"    normalized {path.name}")


def first_numeric_header_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                return int(line.split()[0])
    raise RuntimeError(f"No numeric header in {path}")


def write_mesh_count_summary() -> tuple[int, int]:
    node_file = SCRIPT_DIR / f"{MESH_NAME}.1.node"
    ele_file = SCRIPT_DIR / f"{MESH_NAME}.1.ele"
    nodes = first_numeric_header_count(node_file)
    tetrahedra = first_numeric_header_count(ele_file)
    output = SCRIPT_DIR / f"{MESH_NAME}_mesh_counts.txt"
    output.write_text(
        "quantity,count\n"
        f"nodes,{nodes}\n"
        f"tetrahedra,{tetrahedra}\n"
        f"postprocess_tet_limit,{MAX_TETRAHEDRA}\n",
        encoding="utf-8",
    )
    print(f"    nodes      : {nodes:,}")
    print(f"    tetrahedra : {tetrahedra:,}")
    print(f"    guard      : {MAX_TETRAHEDRA:,}")
    return nodes, tetrahedra


def clean_generated_outputs() -> None:
    print("--> Removing generated outputs from a previous run...")
    suffixes = {
        ".poly", ".node", ".ele", ".face", ".edge", ".neigh", ".inp", ".uge", ".ugi", ".h5",
        ".mapping", ".trn", ".txt", ".xyz", ".csv", ".json", ".xmf", ".pvtp", ".vtp", ".vtu", ".log",
    }
    for path in SCRIPT_DIR.iterdir():
        if path.is_file() and path.name.startswith(MESH_NAME) and path.suffix in suffixes:
            path.unlink()
            print(f"    deleted {path.name}")
    for vset_name in VSET_FILES + ["hec_surface.vset"]:
        vset = SCRIPT_DIR / vset_name
        ex = vset.with_suffix(".ex")
        if vset.exists():
            vset.unlink()
            print(f"    deleted {vset.name}")
        if ex.exists():
            ex.unlink()
            print(f"    deleted {ex.name}")


def must_exist(path: Path, label: str, min_bytes: int = 1) -> None:
    if not nonempty(path, min_bytes=min_bytes):
        raise RuntimeError(f"{label} was not created or is empty: {path.name}")


def main() -> None:
    os.chdir(SCRIPT_DIR)
    print("\n" + "*" * 102)
    print("North Avant / Bartlesville HEC: tag-only HEC + five smoothly graded cylindrical borehole mesh zones")
    print("Domain: 10 km x 10 km x 750 m; HEC: tag-only 580 m x 300 m x 5 m at (5000,5000,530) m")
    print("HEC: horizontal; long axis 5 degrees east (+x) of north (+y); material 5 tagged at z=530 m")
    print("Boreholes: grey injection R=5 m runs from HEC top z=532.5 m to z=750 m")
    print("Four green strainmeters: R=2 m, about 2.3 km from the HEC centre, z=650--750 m (upper 100 m only)")
    print("All boreholes use dense cylindrical-shell Part-1 refinement halos and material tags; they are not hollow and not TetGen holes. Run: python3 workflow.py")
    print("*" * 102 + "\n")

    require_helpers()
    tetgen = require_executable(TETGEN_EXE, "TetGen")
    voronoi = require_executable(VORONOI_EXE, "VORONOI")
    if not KEEP_OLD_OUTPUTS:
        clean_generated_outputs()

    node_file = f"{MESH_NAME}.1.node"
    ele_file = f"{MESH_NAME}.1.ele"
    avs_file = f"{MESH_NAME}.inp"
    material_file = f"{MESH_NAME}_materials.txt"
    h5_file = f"{MESH_NAME}_material_ids.h5"
    uge_file = SCRIPT_DIR / f"{MESH_NAME}.uge"
    ugi_file = f"{MESH_NAME}.ugi"

    print("\n--> Step 1/9: writing PLC and running TetGen")
    build_command: List[object] = [sys.executable, SCRIPT_DIR / "build_poly_layers4.py", MESH_NAME, tetgen]
    if TETGEN_DIAGNOSE:
        build_command.append("--diagnose")
    run(build_command)

    print("\n--> Step 2/9: normalizing TetGen output")
    for suffix in (".1.node", ".1.ele", ".1.face", ".1.edge", ".1.neigh"):
        normalize_tetgen_ascii(SCRIPT_DIR / f"{MESH_NAME}{suffix}")

    print("\n--> Step 3/9: checking mesh size")
    _, tetrahedra = write_mesh_count_summary()
    if tetrahedra > MAX_TETRAHEDRA:
        raise RuntimeError(
            f"TetGen created {tetrahedra:,} tetrahedra, above the safety limit of {MAX_TETRAHEDRA:,}.\n"
            "Reduce the borehole shell density or increase the local z spacing in build_poly_layers4.py and rerun."
        )

    print("\n--> Step 4/9: material IDs, boundary tags, HEC tag, and borehole mesh-zone vsets")
    tag_command: List[object] = [sys.executable, SCRIPT_DIR / "layers4_get_material_boundary_tags.py", MESH_NAME]
    if SKIP_PX:
        tag_command.append("--skip-px")
    run(tag_command)
    for vset_name in ("hec.vset", "injection_borehole.vset", "strainmeter_boreholes.vset"):
        must_exist(SCRIPT_DIR / vset_name, vset_name)

    print("\n--> Step 5/9: strict TetGen -> AVS conversion and connectivity validation")
    run([sys.executable, SCRIPT_DIR / "tetgen_to_avs.py", node_file, ele_file, avs_file])
    must_exist(SCRIPT_DIR / avs_file, "AVS mesh", min_bytes=100)

    print("\n--> Step 6/9: writing PFLOTRAN material HDF5")
    run([sys.executable, SCRIPT_DIR / "material_h5_from_txt.py", node_file, h5_file, material_file])
    must_exist(SCRIPT_DIR / h5_file, "Material HDF5", min_bytes=100)

    print("\n--> Step 7/9: generating PFLOTRAN UGE with VORONOI")
    voronoi_log = SCRIPT_DIR / f"{MESH_NAME}_voronoi.log"
    run_voronoi([voronoi, "-avs", avs_file, "-type", "pflotran", "-o", uge_file.name], voronoi_log)
    must_exist(uge_file, "PFLOTRAN UGE", min_bytes=100)

    print("\n--> Step 8/9: converting vsets to PFLOTRAN .ex files")
    for vset_name in VSET_FILES:
        vset = SCRIPT_DIR / vset_name
        if nonempty(vset):
            run([sys.executable, SCRIPT_DIR / "convert_vset_to_ex.py", uge_file.name, vset_name])
        else:
            print(f"WARNING: {vset_name} is missing or empty; skipped.")

    print("\n--> Step 9/9: generating UGI and mapping")
    run([sys.executable, SCRIPT_DIR / "generate_ugi.py", ele_file, node_file, ugi_file])
    run([sys.executable, SCRIPT_DIR / "mapping.py", MESH_NAME])

    print("\n" + "*" * 102)
    print("Workflow completed successfully.")
    print("Material 5 is the rotated tag-only HEC; materials 6--10 are the five solid borehole mesh zones.")
    print("*" * 102 + "\n")


if __name__ == "__main__":
    main()
