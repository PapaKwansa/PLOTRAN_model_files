#!/usr/bin/env python3
"""Assign materials, boundary tags, and PFLOTRAN vsets for the current
Bartlesville tag-only HEC mesh.

The mesh builder writes the active local mesh zones to
``<prefix>_geometry.json`` under ``refinement_targets``.  This tagger reads
that list dynamically; it does not assume a fixed number of boreholes or
strainmeters.

Supported target shapes
-----------------------
* ``vertical_cylinder``: used by ``injection_borehole``.
* ``sphere``: used by the AVN2, AVN87, and AVN31 strainmeter sensor pods.

Material IDs are assigned after TetGen:
  1--4  geological layers
  5     tag-only HEC
  6     injection borehole
  7--9  AVN2, AVN87, AVN31 sensor pods

No target is a TetGen hole or an internal PLC region.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

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
        description="Write Bartlesville materials, boundary tags, and PFLOTRAN vsets."
    )
    parser.add_argument("mesh_prefix", help="Mesh prefix, e.g. bartlesville_hec")
    parser.add_argument("--skip-px", action="store_true", help="Do not invoke px.py")
    return parser.parse_args()


def iter_data_lines(path: Path):
    """Yield non-empty, non-comment ASCII TetGen lines."""
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            value = raw.strip()
            if value and not value.startswith("#"):
                yield value


def read_node_xyz(path: Path) -> np.ndarray:
    """Read a TetGen .node file and return coordinates in node-file order."""
    lines = iter_data_lines(path)
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"No TetGen node header in {path}") from exc
    if len(header) < 4:
        raise RuntimeError(f"Malformed TetGen node header in {path}: {' '.join(header)!r}")

    count = int(header[0])
    dimension = int(header[1])
    if dimension != 3:
        raise RuntimeError(f"Expected a 3-D TetGen .node file, received dimension={dimension}.")

    values = np.empty((count, 3), dtype=float)
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"Expected {count} node rows in {path}; stopped at row {row}.") from exc
        if len(fields) < 4:
            raise RuntimeError(f"Malformed node row {row + 1} in {path}.")
        values[row] = (float(fields[1]), float(fields[2]), float(fields[3]))
    return values


def write_vset(path: Path, ids_1based: Iterable[int]) -> int:
    """Write unique 1-based node IDs, one per line, and return the count."""
    values = sorted({int(value) for value in ids_1based})
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")
    return len(values)


def assign_base_materials(z: np.ndarray, layers: Sequence[Mapping[str, Any]], tolerance: float) -> np.ndarray:
    """Assign material IDs using the layer intervals in the JSON sidecar."""
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


def pure_outer_face_masks(
    points: np.ndarray,
    domain: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, np.ndarray]:
    """Return masks for the open part of each outer face, excluding edges/corners."""
    minimum = np.asarray(domain["min"], dtype=float)
    maximum = np.asarray(domain["max"], dtype=float)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    top = np.isclose(z, maximum[2], atol=tolerance)
    bottom = np.isclose(z, minimum[2], atol=tolerance)
    north = np.isclose(y, maximum[1], atol=tolerance)
    south = np.isclose(y, minimum[1], atol=tolerance)
    east = np.isclose(x, maximum[0], atol=tolerance)
    west = np.isclose(x, minimum[0], atol=tolerance)
    hit_count = (
        top.astype(int)
        + bottom.astype(int)
        + north.astype(int)
        + south.astype(int)
        + east.astype(int)
        + west.astype(int)
    )
    return {
        "top": top & (hit_count == 1),
        "bottom": bottom & (hit_count == 1),
        "north": north & (hit_count == 1),
        "south": south & (hit_count == 1),
        "east": east & (hit_count == 1),
        "west": west & (hit_count == 1),
    }


def hec_local_coordinates(
    points: np.ndarray,
    hec: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(hec["center"], dtype=float)
    axes = hec["axes"]
    length_axis = np.asarray(axes["length"], dtype=float)
    width_axis = np.asarray(axes["width"], dtype=float)
    up_axis = np.asarray(axes["normal_up"], dtype=float)
    relative = points - center
    return relative @ length_axis, relative @ width_axis, relative @ up_axis


def strict_hec_mask(points: np.ndarray, hec: Mapping[str, Any], tolerance: float) -> np.ndarray:
    """Select existing mesh points on the exact HEC mid-plane and footprint."""
    local_length, local_width, local_up = hec_local_coordinates(points, hec)
    return (
        np.isclose(local_up, 0.0, atol=tolerance)
        & (np.abs(local_length) <= 0.5 * float(hec["length_m"]) - 1.0e-9)
        & (np.abs(local_width) <= 0.5 * float(hec["width_m"]) - 1.0e-9)
    )


def validate_refinement_targets(targets: Sequence[Mapping[str, Any]]) -> None:
    """Fail early for malformed or duplicate target definitions."""
    if not targets:
        raise RuntimeError(
            "No refinement_targets were found in the geometry JSON. "
            "Use the current build_poly_layers4.py before running this tagger."
        )

    names: set[str] = set()
    material_ids: set[int] = set()
    for target in targets:
        name = str(target.get("name", "")).strip()
        if not name:
            raise RuntimeError("A refinement target has no name.")
        if name in names:
            raise RuntimeError(f"Duplicate refinement target name: {name}")
        names.add(name)

        material_id = int(target["material_id"])
        if material_id in material_ids:
            raise RuntimeError(f"Duplicate material_id among refinement targets: {material_id}")
        material_ids.add(material_id)

        center = np.asarray(target.get("center_xyz_m"), dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise RuntimeError(f"Target {name} has an invalid center_xyz_m definition.")
        radius = float(target["tag_radius_m"])
        if not np.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(f"Target {name} has invalid tag_radius_m={radius}.")

        shape = str(target.get("tag_shape", "")).strip()
        if shape not in {"vertical_cylinder", "sphere"}:
            raise RuntimeError(
                f"Target {name} has unsupported tag_shape={shape!r}; "
                "supported values are 'vertical_cylinder' and 'sphere'."
            )
        if shape == "vertical_cylinder":
            z_min = target.get("tag_z_min_m")
            z_max = target.get("tag_z_max_m")
            if z_min is None or z_max is None or float(z_min) > float(z_max):
                raise RuntimeError(f"Cylinder target {name} has invalid tag_z_min_m/tag_z_max_m.")


def refinement_target_mask(
    points: np.ndarray,
    target: Mapping[str, Any],
    tolerance: float,
) -> np.ndarray:
    """Select a target by its physical post-mesh tag geometry."""
    center = np.asarray(target["center_xyz_m"], dtype=float)
    radius = float(target["tag_radius_m"])
    shape = str(target["tag_shape"])

    if shape == "vertical_cylinder":
        radial2 = (points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2
        z_min = float(target["tag_z_min_m"])
        z_max = float(target["tag_z_max_m"])
        return (
            radial2 <= (radius + tolerance) ** 2
        ) & (points[:, 2] >= z_min - tolerance) & (points[:, 2] <= z_max + tolerance)

    if shape == "sphere":
        delta = points - center
        return np.einsum("ij,ij->i", delta, delta) <= (radius + tolerance) ** 2

    raise RuntimeError(f"Unsupported target shape encountered after validation: {shape!r}")


def write_hec_reports(
    prefix: str,
    points: np.ndarray,
    selected: np.ndarray,
    geometry: Mapping[str, Any],
) -> None:
    hec = geometry["hec"]
    local_length, local_width, local_up = hec_local_coordinates(points[selected], hec)

    with Path(f"{prefix}_hec_tagged_nodes.xyz").open("w", encoding="utf-8") as handle:
        handle.write("# id x_m y_m z_m local_length_m local_width_m local_up_m\n")
        for node_id, point, uu, vv, ww in zip(
            selected + 1,
            points[selected],
            local_length,
            local_width,
            local_up,
        ):
            handle.write(
                f"{int(node_id)} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f} "
                f"{uu:.10f} {vv:.10f} {ww:.10f}\n"
            )

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
        writer.writerow([
            "tagged_z_unique_m",
            ";".join(f"{value:.10f}" for value in np.unique(points[selected, 2])),
        ])
        writer.writerow(["minimum_local_length_m", f"{np.min(local_length):.10f}"])
        writer.writerow(["maximum_local_length_m", f"{np.max(local_length):.10f}"])
        writer.writerow(["minimum_local_width_m", f"{np.min(local_width):.10f}"])
        writer.writerow(["maximum_local_width_m", f"{np.max(local_width):.10f}"])

    with Path(f"{prefix}_hec_topview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x_m", "y_m", "z_m", "local_length_m", "local_width_m"])
        for node_id, point, uu, vv in zip(selected + 1, points[selected], local_length, local_width):
            writer.writerow([
                int(node_id),
                f"{point[0]:.10f}",
                f"{point[1]:.10f}",
                f"{point[2]:.10f}",
                f"{uu:.10f}",
                f"{vv:.10f}",
            ])


def write_target_reports(
    prefix: str,
    points: np.ndarray,
    targets: Sequence[Mapping[str, Any]],
    masks: Mapping[str, np.ndarray],
) -> None:
    """Write explicit reports for the injection and all sensor-pod tags."""
    header = [
        "name",
        "kind",
        "purpose",
        "source_sensor_id",
        "material_id",
        "color_hint",
        "tag_shape",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "tag_radius_m",
        "tag_z_min_m",
        "tag_z_max_m",
        "tagged_node_count",
        "minimum_tagged_x_m",
        "maximum_tagged_x_m",
        "minimum_tagged_y_m",
        "maximum_tagged_y_m",
        "minimum_tagged_z_m",
        "maximum_tagged_z_m",
    ]
    rows = []
    for target in targets:
        name = str(target["name"])
        selected = np.where(masks[name])[0]
        tagged = points[selected]
        center = np.asarray(target["center_xyz_m"], dtype=float)
        rows.append([
            name,
            target.get("kind", ""),
            target.get("purpose", ""),
            target.get("source_sensor_id", "") or "",
            int(target["material_id"]),
            target.get("color_hint", ""),
            target["tag_shape"],
            f"{center[0]:.10f}",
            f"{center[1]:.10f}",
            f"{center[2]:.10f}",
            f"{float(target['tag_radius_m']):.10f}",
            "" if target.get("tag_z_min_m") is None else f"{float(target['tag_z_min_m']):.10f}",
            "" if target.get("tag_z_max_m") is None else f"{float(target['tag_z_max_m']):.10f}",
            int(selected.size),
            f"{np.min(tagged[:, 0]):.10f}",
            f"{np.max(tagged[:, 0]):.10f}",
            f"{np.min(tagged[:, 1]):.10f}",
            f"{np.max(tagged[:, 1]):.10f}",
            f"{np.min(tagged[:, 2]):.10f}",
            f"{np.max(tagged[:, 2]):.10f}",
        ])

    # New descriptive report name.
    for report_path in (
        Path(f"{prefix}_refinement_target_tag_report.csv"),
        # Legacy filename retained because older analysis scripts may expect it.
        Path(f"{prefix}_borehole_tag_report.csv"),
    ):
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


def write_px_inputs(prefix: str, materials: np.ndarray, skip_px: bool) -> None:
    Path(f"{prefix}.trn").write_text(
        "0.000000000000E+00  0.000000000000E+00  0.000000000000E+00\n",
        encoding="utf-8",
    )
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
        subprocess.run(
            [sys.executable, "px.py", "-f", prefix, str(summary), "meshtags", "0.0", "tags"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: px.py failed but material tags were written: {exc}")


def main() -> None:
    args = parse_args()
    prefix = args.mesh_prefix.removesuffix(".poly")
    geometry_path = Path(f"{prefix}_geometry.json")
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Missing geometry sidecar: {geometry_path}")

    geometry: Dict[str, Any] = json.loads(geometry_path.read_text(encoding="utf-8"))
    targets = list(geometry.get("refinement_targets", []))
    validate_refinement_targets(targets)

    points = read_node_xyz(Path(f"{prefix}.1.node"))
    tolerance = 1.0e-6

    materials = assign_base_materials(points[:, 2], geometry["layers"], tolerance)
    missing = np.where(materials == -999)[0]
    if missing.size:
        sample = ", ".join(f"{points[index, 2]:.6g}" for index in missing[:10])
        raise RuntimeError(f"{missing.size} nodes received no base material; z examples: {sample}")

    # Material 5: exact HEC mid-plane tag.
    hec_mask = strict_hec_mask(points, geometry["hec"], tolerance)
    hec_mask &= materials == int(geometry["hec"]["host_material_id"])
    hec_ids = np.where(hec_mask)[0]
    if hec_ids.size == 0:
        raise RuntimeError("No material-5 HEC nodes found. The z=530 rotated lattice is missing.")
    materials[hec_mask] = int(geometry["hec"]["material_id"])

    # Material 6 and sensor-pod materials: dynamic list from geometry JSON.
    target_masks: Dict[str, np.ndarray] = {}
    occupied = np.zeros(points.shape[0], dtype=bool)
    for target in targets:
        name = str(target["name"])
        mask = refinement_target_mask(points, target, tolerance)
        if not np.any(mask):
            raise RuntimeError(
                f"No mesh nodes were found in physical tag geometry for {name}. "
                "Increase the innermost refinement resolution or tag radius."
            )
        if np.any(mask & occupied):
            other_names = [other_name for other_name, other_mask in target_masks.items() if np.any(mask & other_mask)]
            raise RuntimeError(f"Target tags overlap for {name}; overlaps {other_names}.")
        target_masks[name] = mask
        occupied |= mask
        # Targets override geological/HEC material only if their physical tags
        # actually overlap. In the supplied setup they are non-overlapping.
        materials[mask] = int(target["material_id"])

    np.savetxt(f"{prefix}_materials.txt", materials, fmt="%d")
    print(f"--> Wrote {prefix}_materials.txt")

    # Base layer and HEC vsets.
    for material_id, name in BASE_MATERIAL_VSETS.items():
        count = write_vset(Path(f"{name}.vset"), np.where(materials == material_id)[0] + 1)
        print(f"    {name}.vset: {count} nodes")

    # One vset per actual target, plus descriptive and legacy-compatible unions.
    all_target_mask = np.zeros(points.shape[0], dtype=bool)
    sensor_mask = np.zeros(points.shape[0], dtype=bool)
    injection_mask = np.zeros(points.shape[0], dtype=bool)
    for target in targets:
        name = str(target["name"])
        material_id = int(target["material_id"])
        material_mask = materials == material_id
        count = write_vset(Path(f"{name}.vset"), np.where(material_mask)[0] + 1)
        print(f"    {name}.vset: {count} nodes")
        all_target_mask |= material_mask
        if str(target.get("kind", "")).startswith("strainmeter"):
            sensor_mask |= material_mask
        if str(target.get("kind", "")) == "injection_borehole":
            injection_mask |= material_mask

    print(f"    refined_targets.vset: {write_vset(Path('refined_targets.vset'), np.where(all_target_mask)[0] + 1)} nodes")
    print(f"    sensor_pods.vset: {write_vset(Path('sensor_pods.vset'), np.where(sensor_mask)[0] + 1)} nodes")
    print(f"    strainmeter_sensors.vset: {write_vset(Path('strainmeter_sensors.vset'), np.where(sensor_mask)[0] + 1)} nodes")
    # Keep these former union names so existing PFLOTRAN post-processing does not break.
    print(f"    strainmeter_boreholes.vset: {write_vset(Path('strainmeter_boreholes.vset'), np.where(sensor_mask)[0] + 1)} nodes")
    print(f"    boreholes.vset: {write_vset(Path('boreholes.vset'), np.where(all_target_mask)[0] + 1)} nodes")
    if not np.any(injection_mask):
        raise RuntimeError("No injection-borehole nodes were tagged.")

    # Outer boundary vsets and numerical boundary marker file.
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
    write_target_reports(prefix, points, targets, target_masks)
    write_px_inputs(prefix, materials, args.skip_px)
    print(f"--> Wrote {prefix}_hec_tag_report.csv")
    print(f"--> Wrote {prefix}_refinement_target_tag_report.csv")
    print(f"--> Wrote {prefix}_borehole_tag_report.csv (compatibility copy)")

    print("\nMaterial ID distribution:")
    values, counts = np.unique(materials, return_counts=True)
    for value, count in zip(values, counts):
        print(f"  material {value:>2}: {count}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
