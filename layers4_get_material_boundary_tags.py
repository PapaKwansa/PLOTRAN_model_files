#!/usr/bin/env python3
"""Assign material IDs, outer-boundary tags, and tag-only HEC vsets.

This updated version no longer expects any borehole mesh zones.
The HEC is material 5 and is selected directly from the tag-only rotated
matrix lattice. The small injection source region is handled later by
make_wellbore_vset.py from the TetGen .node file.

Outputs:
  * <mesh>_materials.txt
  * top/bottom/north/south/east/west.vset
  * overburden/bartlesville_sand/basal_layer/underburden.vset
  * hec.vset
  * <mesh>_hec_tagged_nodes.xyz
  * <mesh>_hec_tag_report.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

OUTER_FACE_MARKERS: Dict[int, str] = {
    1: "top",
    2: "bottom",
    3: "north",
    4: "south",
    5: "east",
    6: "west",
}

BASE_MATERIAL_VSETS: Dict[int, str] = {
    1: "overburden",
    2: "bartlesville_sand",
    3: "basal_layer",
    4: "underburden",
    5: "hec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Bartlesville materials, boundaries, and HEC vsets."
    )

    parser.add_argument(
        "mesh_prefix",
        help="Mesh prefix, e.g. bartlesville_hec"
    )

    parser.add_argument(
        "--skip-px",
        action="store_true",
        help="Do not invoke px.py"
    )

    return parser.parse_args()


def iter_data_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            value = raw.strip()
            if value and not value.startswith("#"):
                yield value


def read_node_xyz(path: Path) -> np.ndarray:
    lines = iter_data_lines(path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"No TetGen node header in {path}") from exc
    count = int(header[0])
    values = np.empty((count, 3), dtype=float)
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"Expected {count} node rows in {path}; stopped at {row}.") from exc
        if len(fields) < 4:
            raise RuntimeError(f"Malformed node row {row + 1} in {path}.")
        values[row] = (float(fields[1]), float(fields[2]), float(fields[3]))
    return values


def write_vset(path: Path, ids_1based: Iterable[int]) -> int:
    values = sorted({int(value) for value in ids_1based})
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")
    return len(values)


def assign_base_materials(z: np.ndarray, layers: list[dict], tolerance: float) -> np.ndarray:
    result = np.full(z.size, -999, dtype=int)
    ordered = sorted(layers, key=lambda layer: float(layer["z_min"]))
    for index, layer in enumerate(ordered):
        lower = float(layer["z_min"])
        upper = float(layer["z_max"])
        material_id = int(layer["material_id"])
        if index == len(ordered) - 1:
            mask = (z >= lower - tolerance) & (z <= upper + tolerance)
        else:
            mask = (z >= lower - tolerance) & (z < upper - tolerance)
        result[mask] = material_id
    return result


def pure_outer_face_masks(points: np.ndarray, domain: Mapping[str, object], tolerance: float) -> Dict[str, np.ndarray]:
    minimum, maximum = np.asarray(domain["min"], dtype=float), np.asarray(domain["max"], dtype=float)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    top = np.isclose(z, maximum[2], atol=tolerance)
    bottom = np.isclose(z, minimum[2], atol=tolerance)
    north = np.isclose(y, maximum[1], atol=tolerance)
    south = np.isclose(y, minimum[1], atol=tolerance)
    east = np.isclose(x, maximum[0], atol=tolerance)
    west = np.isclose(x, minimum[0], atol=tolerance)
    hit_count = top.astype(int) + bottom.astype(int) + north.astype(int) + south.astype(int) + east.astype(int) + west.astype(int)
    return {
        "top": top & (hit_count == 1),
        "bottom": bottom & (hit_count == 1),
        "north": north & (hit_count == 1),
        "south": south & (hit_count == 1),
        "east": east & (hit_count == 1),
        "west": west & (hit_count == 1),
    }


def hec_local_coordinates(points: np.ndarray, hec: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(hec["center"], dtype=float)
    axes = hec["axes"]
    length_axis = np.asarray(axes["length"], dtype=float)
    width_axis = np.asarray(axes["width"], dtype=float)
    up_axis = np.asarray(axes["normal_up"], dtype=float)
    rel = points - center
    return rel @ length_axis, rel @ width_axis, rel @ up_axis


def strict_hec_mask(points: np.ndarray, hec: Mapping[str, object], tolerance: float) -> np.ndarray:
    local_length, local_width, local_up = hec_local_coordinates(points, hec)
    return (
        np.isclose(local_up, 0.0, atol=tolerance)
        & (np.abs(local_length) <= 0.5 * float(hec["length_m"]) - 1.0e-9)
        & (np.abs(local_width) <= 0.5 * float(hec["width_m"]) - 1.0e-9)
    )


def write_hec_reports(prefix: str, points: np.ndarray, selected: np.ndarray, geometry: Mapping[str, object]) -> None:
    hec = geometry["hec"]
    u, v, w = hec_local_coordinates(points[selected], hec)
    with Path(f"{prefix}_hec_tagged_nodes.xyz").open("w", encoding="utf-8") as handle:
        handle.write("# id x_m y_m z_m local_length_m local_width_m local_up_m\n")
        for node_id, point, uu, vv, ww in zip(selected + 1, points[selected], u, v, w):
            handle.write(f"{int(node_id)} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {uu:.10f} {vv:.10f} {ww:.10f}\n")
    with Path(f"{prefix}_hec_tag_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        writer.writerow(["representation", hec["representation"]])
        writer.writerow(["center_m", hec["center"]])
        writer.writerow(["length_m", hec["length_m"]])
        writer.writerow(["width_m", hec["width_m"]])
        writer.writerow(["thickness_m", hec["thickness_m"]])
        writer.writerow(["azimuth_deg_east_of_north", hec["azimuth_deg_east_of_north"]])
        writer.writerow(["tagged_node_count", int(selected.size)])
        writer.writerow(["tagged_z_unique_m", ";".join(f"{value:.10f}" for value in np.unique(points[selected, 2]))])
        writer.writerow(["minimum_local_length_m", f"{np.min(u):.10f}"])
        writer.writerow(["maximum_local_length_m", f"{np.max(u):.10f}"])
        writer.writerow(["minimum_local_width_m", f"{np.min(v):.10f}"])
        writer.writerow(["maximum_local_width_m", f"{np.max(v):.10f}"])


def write_px_inputs(prefix: str, materials: np.ndarray, skip_px: bool) -> None:
    Path(f"{prefix}.trn").write_text("0.000000000000E+00  0.000000000000E+00  0.000000000000E+00\n", encoding="utf-8")
    summary = Path(f"{prefix}_material_flags_summary.txt")
    with summary.open("w", encoding="utf-8") as handle:
        handle.write(f"{materials.size} 1\n")
        np.savetxt(handle, materials, fmt="%d")
    if skip_px:
        print("--> px.py skipped.")
        return
    if not Path("px.py").is_file():
        print("--> px.py not found; visualization inputs were written only.")
        return
    try:
        import subprocess
        import sys
        subprocess.run([sys.executable, "px.py", "-f", prefix, str(summary), "meshtags", "0.0", "tags"], check=True)
    except Exception as exc:
        print(f"WARNING: px.py failed but tags were written: {exc}")


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix.removesuffix(".poly")
    geometry_path = Path(f"{prefix}_geometry.json")
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Missing geometry sidecar: {geometry_path}")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    points = read_node_xyz(Path(f"{prefix}.1.node"))
    tolerance = 1.0e-6

    materials = assign_base_materials(points[:, 2], geometry["layers"], tolerance)
    if np.any(materials == -999):
        raise RuntimeError("Some nodes were not assigned a geological material.")

    hec_mask = strict_hec_mask(points, geometry["hec"], tolerance)
    hec_mask &= materials == int(geometry["hec"]["host_material_id"])
    hec_ids = np.where(hec_mask)[0]
    if hec_ids.size == 0:
        raise RuntimeError("No material-5 HEC nodes found. The z=530 rotated lattice is missing.")
    materials[hec_mask] = int(geometry["hec"]["material_id"])

    np.savetxt(f"{prefix}_materials.txt", materials, fmt="%d")
    print(f"--> Wrote {prefix}_materials.txt")

    for material_id, name in BASE_MATERIAL_VSETS.items():
        count = write_vset(Path(f"{name}.vset"), np.where(materials == material_id)[0] + 1)
        print(f"    {name}.vset: {count} nodes")

    boundary_tags = np.full(points.shape[0], -999, dtype=int)
    face_masks = pure_outer_face_masks(points, geometry["domain"], tolerance)
    for marker, name in OUTER_FACE_MARKERS.items():
        ids = np.where(face_masks[name])[0] + 1
        count = write_vset(Path(f"{name}.vset"), ids)
        boundary_tags[ids - 1] = marker
        print(f"    {name}.vset: {count} nodes")
    np.savetxt(f"{prefix}_boundaries.txt", boundary_tags, fmt="%d")
    print(f"--> Wrote {prefix}_boundaries.txt")

    write_hec_reports(prefix, points, hec_ids, geometry)
    print(f"--> Wrote {prefix}_hec_tag_report.csv")
    write_px_inputs(prefix, materials, args.skip_px)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
