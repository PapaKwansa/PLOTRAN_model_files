#!/usr/bin/env python3
"""
Preflight a North Avant V5 PFLOTRAN runtime bundle before Palmetto transfer.

Checks:
  * every runtime-manifest file exists and is nonempty;
  * every file path referenced by the deck exists;
  * deck is TWO_WAY_COUPLED with explicit rate interpolation and checkpointing;
  * UGE, UGI, mapping, and material-HDF5 counts agree;
  * all vset IDs are valid;
  * external-boundary EX records have positive areas;
  * SHA256SUMS and a JSON report are written.

Usage:
  python3 preflight_north_avant_v5_bundle.py \
    north_avant_v5_twoway_preproduction_3h.in \
    north_avant_v5_runtime_manifest.txt

Run it again with the 96-hour deck before the production submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def noncomment_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[Path]:
    files = []
    for line in noncomment_lines(path):
        files.append(Path(line))
    if not files:
        raise RuntimeError(f"{path}: manifest is empty")
    return files


def read_ugi_header(path: Path) -> tuple[int, int]:
    first = next(noncomment_lines(path)).split()
    if len(first) < 2:
        raise RuntimeError(f"{path}: malformed UGI header")
    return int(first[0]), int(first[1])


def read_uge_counts(path: Path) -> tuple[int, int]:
    lines = noncomment_lines(path)
    first = next(lines).split()
    if len(first) < 2 or first[0].upper() != "CELLS":
        raise RuntimeError(f"{path}: malformed UGE CELLS header")
    cells = int(first[1])
    for _ in range(cells):
        next(lines)
    connection_header = next(lines).split()
    if len(connection_header) < 2 or connection_header[0].upper() != "CONNECTIONS":
        raise RuntimeError(f"{path}: malformed UGE CONNECTIONS header")
    return cells, int(connection_header[1])


def check_mapping(path: Path) -> tuple[int, bool]:
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(f"{path}: mapping requires two columns")
    expected = np.arange(1, data.shape[0] + 1, dtype=np.int64)
    identity = (
        np.array_equal(data[:, 0], expected)
        and np.array_equal(data[:, 1], expected)
    )
    return int(data.shape[0]), bool(identity)


def check_vset(path: Path, maximum_id: int) -> int:
    values = []
    for line in noncomment_lines(path):
        value = int(line.split()[0])
        if value < 1 or value > maximum_id:
            raise RuntimeError(
                f"{path}: ID {value} outside valid range 1..{maximum_id}"
            )
        values.append(value)
    if not values:
        raise RuntimeError(f"{path}: empty vset")
    if len(values) != len(set(values)):
        raise RuntimeError(f"{path}: duplicate vset IDs")
    return len(values)


def check_ex(path: Path, maximum_id: int) -> tuple[int, float]:
    lines = noncomment_lines(path)
    header = next(lines).split()
    if len(header) < 2 or header[0].upper() != "CONNECTIONS":
        raise RuntimeError(f"{path}: malformed EX header")
    count = int(header[1])
    area_sum = 0.0
    rows = 0
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError(f"{path}: malformed EX row")
        cell_id = int(fields[0])
        area = float(fields[4])
        if cell_id < 1 or cell_id > maximum_id:
            raise RuntimeError(f"{path}: invalid cell ID {cell_id}")
        if not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(f"{path}: nonpositive/nonfinite area {area}")
        area_sum += area
        rows += 1
    if rows != count:
        raise RuntimeError(f"{path}: header count {count}, parsed {rows}")
    return rows, area_sum


def referenced_paths(deck_text: str) -> set[Path]:
    paths: set[Path] = set()
    patterns = [
        r"^\s*TYPE\s+UNSTRUCTURED_EXPLICIT\s+(\S+)",
        r"^\s*TYPE\s+UNSTRUCTURED\s+(\S+)",
        r"^\s*GEOMECHANICS_MAPPING_FILE\s+(\S+)",
        r"^\s*FILE\s+(\S+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, deck_text, flags=re.MULTILINE):
            token = match.group(1).strip().strip('"').strip("'")
            if token.startswith("__"):
                continue
            paths.add(Path(token))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("north_avant_v5_bundle_preflight.json"),
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=Path("north_avant_v5_bundle_SHA256SUMS"),
    )
    args = parser.parse_args()

    root = Path.cwd()
    deck = args.deck.resolve()
    manifest = args.manifest.resolve()

    for path in (deck, manifest):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest_paths = read_manifest(manifest)
    missing = [
        str(path)
        for path in manifest_paths
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("Missing/empty manifest files:\n  " + "\n  ".join(missing))

    deck_text = deck.read_text(encoding="utf-8")
    required_tokens = {
        "two-way coupling": "FLOW_COUPLING TWO_WAY_COUPLED",
        "explicit interpolation": "INTERPOLATION ",
        "checkpointing": "CHECKPOINT",
        "median UGE": "bartlesville_hec_lime_v5_interfaces_median.uge",
        "canonical UGI": "bartlesville_hec_lime_v5_interfaces.ugi",
        "validated mapping": "bartlesville_hec_lime_v5_interfaces_median.mapping",
        "limestone": "shallow_limestone",
    }
    absent = [label for label, token in required_tokens.items() if token not in deck_text]
    if absent:
        raise RuntimeError("Deck is missing required features: " + ", ".join(absent))

    references = referenced_paths(deck_text)
    missing_refs = [str(path) for path in sorted(references) if not path.is_file()]
    if missing_refs:
        raise RuntimeError("Deck references missing files:\n  " + "\n  ".join(missing_refs))

    uge = Path("bartlesville_hec_lime_v5_interfaces_median.uge")
    ugi = Path("bartlesville_hec_lime_v5_interfaces.ugi")
    mapping = Path("bartlesville_hec_lime_v5_interfaces_median.mapping")
    materials = Path("bartlesville_hec_lime_v5_interfaces_material_ids.h5")

    flow_cells, flow_connections = read_uge_counts(uge)
    tetrahedra, mechanics_vertices = read_ugi_header(ugi)
    mapping_rows, mapping_identity = check_mapping(mapping)

    with h5py.File(materials, "r") as h5:
        cell_ids = np.asarray(h5["/Materials/Cell Ids"][...])
        material_ids = np.asarray(h5["/Materials/Material Ids"][...])

    if not (
        flow_cells
        == mechanics_vertices
        == mapping_rows
        == cell_ids.size
        == material_ids.size
    ):
        raise RuntimeError(
            "Count mismatch: "
            f"UGE={flow_cells}, UGI vertices={mechanics_vertices}, "
            f"mapping={mapping_rows}, material rows={material_ids.size}"
        )

    if not np.array_equal(
        cell_ids,
        np.arange(1, flow_cells + 1, dtype=cell_ids.dtype),
    ):
        raise RuntimeError("Material HDF5 cell IDs are not contiguous 1..N")

    vset_counts = {}
    ex_reports = {}
    for path in manifest_paths:
        if path.suffix == ".vset":
            vset_counts[str(path)] = check_vset(path, mechanics_vertices)
        elif path.suffix == ".ex":
            count, area_sum = check_ex(path, flow_cells)
            ex_reports[str(path)] = {
                "records": count,
                "area_sum_m2": area_sum,
            }

    checksum_files = [deck, manifest] + [path.resolve() for path in manifest_paths]
    checksum_lines = [f"{sha256(path)}  {path.relative_to(root)}" for path in checksum_files]
    args.checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    report = {
        "status": "passed",
        "deck": str(deck),
        "manifest": str(manifest),
        "counts": {
            "flow_cells": flow_cells,
            "flow_connections": flow_connections,
            "mechanics_vertices": mechanics_vertices,
            "tetrahedra": tetrahedra,
            "mapping_rows": mapping_rows,
        },
        "mapping_identity": mapping_identity,
        "material_ids": sorted(int(value) for value in np.unique(material_ids)),
        "vset_counts": vset_counts,
        "external_boundary_reports": ex_reports,
        "deck_references": sorted(str(path) for path in references),
        "checksums_file": str(args.checksums.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("North Avant V5 runtime-bundle preflight: PASSED")
    print(f"  flow cells       : {flow_cells:,}")
    print(f"  flow connections : {flow_connections:,}")
    print(f"  mechanics nodes  : {mechanics_vertices:,}")
    print(f"  tetrahedra       : {tetrahedra:,}")
    print(f"  mapping identity : {mapping_identity}")
    print(f"  material IDs     : {report['material_ids']}")
    print(f"  report           : {args.report}")
    print(f"  checksums        : {args.checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
