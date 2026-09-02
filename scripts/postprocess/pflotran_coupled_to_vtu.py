#!/usr/bin/env python3
"""
Convert matched PFLOTRAN flow and geomechanics HDF5 outputs to VTU/PVD.

The canonical UGI supplies tetrahedral topology. The validated mapping places
flow-cell arrays on the corresponding geomechanics vertices. At every common
output time, the VTU contains:

* all compatible geomechanics nodal fields;
* a true three-component Displacement vector and magnitude;
* a Relative_Displacement vector and magnitude when available;
* mapped flow fields prefixed with ``Flow_`` (pressure, saturation, porosity,
  permeability, material ID, etc.);
* ``Flow_Liquid_Pressure_Change_Pa`` relative to the earliest flow snapshot,
  when a liquid-pressure field is available;
* optional nodal flags generated from PFLOTRAN vset files.

This bypasses PFLOTRAN's native XMF reader path.

Example
-------
python3 pflotran_coupled_to_vtu.py \
  run-geomech.h5 run.h5 canonical.ugi validated.mapping \
  --output-dir paraview_coupled \
  --vset HEC=hec.vset \
  --vset Injection=injection_borehole.vset \
  --vset AVN2=AVN2.vset \
  --vset AVN87=AVN87.vset \
  --vset AVN31=AVN31.vset
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

try:
    import meshio
except ImportError as exc:
    raise SystemExit(
        "meshio is required. Install it in the postprocessing environment."
    ) from exc


TIME_PATTERN = re.compile(
    r"Time\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"\s*([A-Za-z]*)",
    re.IGNORECASE,
)


def data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_ugi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = iter(data_lines(path))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty UGI: {path}") from exc

    if len(header) < 2:
        raise RuntimeError(f"Malformed UGI header: {header}")

    element_count = int(header[0])
    node_count = int(header[1])
    connectivity = np.empty((element_count, 4), dtype=np.int64)

    for row in range(element_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(
                f"UGI ended while reading element {row + 1}"
            ) from exc

        if len(fields) < 5 or fields[0].upper() not in {"T", "TET"}:
            raise RuntimeError(
                f"Expected tetrahedron at UGI element row {row + 1}: {fields}"
            )

        ids = np.asarray([int(value) for value in fields[1:5]], dtype=np.int64)
        if ids.min() < 1 or ids.max() > node_count:
            raise RuntimeError(
                f"UGI element {row + 1} has invalid vertex IDs: {ids.tolist()}"
            )
        connectivity[row] = ids - 1

    points = np.empty((node_count, 3), dtype=np.float64)
    for row in range(node_count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(
                f"UGI ended while reading coordinate row {row + 1}"
            ) from exc
        if len(fields) < 3:
            raise RuntimeError(
                f"Malformed UGI coordinate row {row + 1}: {fields}"
            )
        points[row] = [float(fields[0]), float(fields[1]), float(fields[2])]

    return points, connectivity


def safe_name(name: str) -> str:
    value = name.strip().replace("[", "").replace("]", "")
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def normalized_name(name: str) -> str:
    return safe_name(name).lower()


def parse_time(group_path: str) -> tuple[float, str]:
    match = TIME_PATTERN.search(group_path)
    if not match:
        raise ValueError(group_path)
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    return value, match.group(2) or "h"


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
    if key in {"y", "yr", "year", "years"}:
        return float(value) * 365.25 * 24.0
    raise ValueError(f"Unsupported time unit: {unit!r}")


def dataset_to_1d(dataset: h5py.Dataset, count: int) -> np.ndarray | None:
    shape = tuple(int(value) for value in dataset.shape)
    if shape == (count,):
        values = np.asarray(dataset[...])
    elif shape == (count, 1):
        values = np.asarray(dataset[:, 0])
    else:
        return None

    if not (
        np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        return None

    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise RuntimeError(f"Dataset {dataset.name!r} contains NaN or infinity")
    return values


def discover_time_groups(h5: h5py.File, count: int) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []

    def visitor(name: str, obj) -> None:
        if not isinstance(obj, h5py.Group) or "time" not in name.lower():
            return

        compatible = sum(
            1
            for item in obj.values()
            if isinstance(item, h5py.Dataset)
            and tuple(int(value) for value in item.shape) in {(count,), (count, 1)}
        )
        if compatible == 0:
            return

        try:
            native, unit = parse_time(name)
            hours = time_to_hours(native, unit)
        except ValueError:
            return

        groups.append(
            {
                "time_native": native,
                "time_unit": unit,
                "time_hours": hours,
                "path": "/" + name,
            }
        )

    h5.visititems(visitor)
    groups.sort(key=lambda item: (float(item["time_hours"]), str(item["path"])))
    return groups


def load_group_arrays(group: h5py.Group, count: int, prefix: str = "") -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}

    for original_name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        values = dataset_to_1d(dataset, count)
        if values is None:
            continue

        output_name = prefix + safe_name(original_name)
        base = output_name
        suffix = 2
        while output_name in arrays:
            output_name = f"{base}_{suffix}"
            suffix += 1

        arrays[output_name] = values
        names[original_name] = output_name

    return arrays, names


def read_mapping(path: Path, flow_count: int, mechanics_count: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(f"{path}: expected at least two columns")

    flow_ids = data[:, 0]
    mechanics_ids = data[:, 1]

    if data.shape[0] != mechanics_count:
        raise RuntimeError(
            f"{path}: expected {mechanics_count} mapping rows, found {data.shape[0]}"
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


def map_flow_array(values: np.ndarray, flow_ids: np.ndarray, mechanics_ids: np.ndarray, mechanics_count: int) -> np.ndarray:
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


def parse_vset_argument(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--vset requires NAME=PATH")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--vset requires nonempty NAME and PATH")
    return safe_name(name), Path(path)


def read_vset_flag(path: Path, node_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids: set[int] = set()
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        node_id = int(value)
        if node_id < 1 or node_id > node_count:
            raise RuntimeError(f"{path}: invalid node ID {node_id}")
        ids.add(node_id)
    if not ids:
        raise RuntimeError(f"{path}: empty vset")
    flag = np.zeros(node_count, dtype=np.uint8)
    flag[np.asarray(sorted(value - 1 for value in ids), dtype=np.int64)] = 1
    return flag


def find_component(point_data: dict[str, np.ndarray], base_name: str) -> np.ndarray | None:
    target = normalized_name(base_name)
    for name, values in point_data.items():
        normalized = normalized_name(name)
        if normalized == target or normalized.startswith(target + "_"):
            array = np.asarray(values)
            if array.ndim == 1:
                return array.astype(np.float64, copy=False)
    return None


def add_vector(point_data: dict[str, np.ndarray], output_name: str, x_name: str, y_name: str, z_name: str) -> bool:
    x = find_component(point_data, x_name)
    y = find_component(point_data, y_name)
    z = find_component(point_data, z_name)
    if x is None or y is None or z is None:
        return False
    vector = np.column_stack((x, y, z))
    point_data[output_name] = vector
    point_data[output_name + "_Magnitude"] = np.linalg.norm(vector, axis=1)
    return True


def nearest_group(target_hours: float, groups: list[dict[str, object]], tolerance: float) -> dict[str, object] | None:
    if not groups:
        return None
    distances = np.asarray(
        [abs(float(item["time_hours"]) - target_hours) for item in groups],
        dtype=float,
    )
    index = int(np.argmin(distances))
    return groups[index] if distances[index] <= tolerance else None


def find_field_key(arrays: dict[str, np.ndarray], requested: str) -> str | None:
    """Return an exact or unit-suffixed sanitized field key."""
    target = normalized_name(requested)
    exact: list[str] = []
    prefixed: list[str] = []

    for key in arrays:
        normalized = normalized_name(key)
        if normalized == target:
            exact.append(key)
        elif normalized.startswith(target + "_"):
            prefixed.append(key)

    if exact:
        return sorted(exact)[0]
    if prefixed:
        return sorted(prefixed)[0]
    return None


def write_pvd(path: Path, records: list[tuple[float, Path]]) -> None:
    vtkfile = ET.Element(
        "VTKFile",
        {"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
    )
    collection = ET.SubElement(vtkfile, "Collection")
    for time_hours, vtu_path in records:
        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": f"{time_hours:.16g}",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge PFLOTRAN flow and geomechanics HDF5 output into VTU/PVD."
    )
    parser.add_argument("geomech_h5", type=Path)
    parser.add_argument("flow_h5", type=Path)
    parser.add_argument("ugi", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paraview_coupled"))
    parser.add_argument("--prefix")
    parser.add_argument("--vset", action="append", default=[], type=parse_vset_argument)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--time-tolerance-hours", type=float, default=1.0e-6)
    parser.add_argument("--coordinate-atol", type=float, default=1.0e-6)
    args = parser.parse_args()

    geomech_path = args.geomech_h5.expanduser().resolve()
    flow_path = args.flow_h5.expanduser().resolve()
    ugi_path = args.ugi.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (geomech_path, flow_path, ugi_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = safe_name(args.prefix or geomech_path.stem.removesuffix("-geomech") + "_coupled")

    print(f"Reading canonical UGI: {ugi_path}")
    ugi_points, connectivity = read_ugi(ugi_path)
    mechanics_count = int(ugi_points.shape[0])
    element_count = int(connectivity.shape[0])
    print(f"  mechanics vertices: {mechanics_count:,}")
    print(f"  tetrahedra:         {element_count:,}")

    static_flags: dict[str, np.ndarray] = {}
    for name, path in args.vset:
        flag_name = safe_name(name) + "_Flag"
        static_flags[flag_name] = read_vset_flag(path.expanduser().resolve(), mechanics_count)
        print(f"  {flag_name}: {int(static_flags[flag_name].sum()):,} vertices")

    records: list[tuple[float, Path]] = []
    manifest_times: list[dict[str, object]] = []

    with h5py.File(geomech_path, "r") as geomech_h5, h5py.File(flow_path, "r") as flow_h5:
        if "/Domain/Vertices" not in geomech_h5:
            raise RuntimeError(f"{geomech_path}: missing /Domain/Vertices")
        h5_points = np.asarray(geomech_h5["/Domain/Vertices"][...], dtype=float)
        if h5_points.shape != ugi_points.shape:
            raise RuntimeError(
                f"UGI/geomechanics vertex shape mismatch: {ugi_points.shape} vs {h5_points.shape}"
            )
        maximum_difference = float(np.max(np.abs(h5_points - ugi_points)))
        print(f"Maximum UGI/HDF5 vertex difference: {maximum_difference:.6e} m")
        if maximum_difference > args.coordinate_atol:
            raise RuntimeError("UGI/geomechanics coordinates exceed tolerance")

        # Infer flow-cell count from compatible time datasets, then validate mapping.
        possible_counts: list[int] = []
        def flow_shape_visitor(_name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim in {1, 2}:
                if obj.shape and obj.shape[0] > 1:
                    possible_counts.append(int(obj.shape[0]))
        flow_h5.visititems(flow_shape_visitor)
        if not possible_counts:
            raise RuntimeError(f"{flow_path}: no candidate flow-cell datasets found")
        # The modal leading dimension avoids small metadata arrays.
        counts, frequencies = np.unique(possible_counts, return_counts=True)
        flow_count = int(counts[int(np.argmax(frequencies))])
        print(f"Inferred flow-cell count: {flow_count:,}")

        flow_ids, mechanics_ids = read_mapping(
            mapping_path, flow_count, mechanics_count
        )

        geomech_groups = discover_time_groups(geomech_h5, mechanics_count)
        flow_groups = discover_time_groups(flow_h5, flow_count)
        if not geomech_groups:
            raise RuntimeError("No compatible geomechanics time groups found")
        if not flow_groups:
            raise RuntimeError("No compatible flow time groups found")

        # Build a mapped liquid-pressure baseline from the earliest available
        # flow snapshot. This makes the injection signal visible without the
        # large hydrostatic background dominating the color scale.
        baseline_flow_info = flow_groups[0]
        baseline_flow_arrays, _ = load_group_arrays(
            flow_h5[str(baseline_flow_info["path"])],
            flow_count,
            prefix="Flow_",
        )
        baseline_pressure_key = find_field_key(
            baseline_flow_arrays,
            "Flow_Liquid_Pressure",
        )
        baseline_pressure_mapped: np.ndarray | None = None
        if baseline_pressure_key is not None:
            baseline_pressure_mapped = map_flow_array(
                baseline_flow_arrays[baseline_pressure_key],
                flow_ids,
                mechanics_ids,
                mechanics_count,
            ).astype(np.float64, copy=False)
            print(
                "Pressure-change baseline: "
                f"time={float(baseline_flow_info['time_hours']):.12g} h, "
                f"field={baseline_pressure_key}"
            )
        else:
            print(
                "WARNING: no liquid-pressure field was found in the earliest "
                "flow snapshot; pressure-change output will be omitted."
            )

        if args.latest_only:
            geomech_groups = [geomech_groups[-1]]

        print(f"Geomechanics times selected: {len(geomech_groups)}")
        print(f"Flow times available:         {len(flow_groups)}")

        for output_index, geomech_info in enumerate(geomech_groups):
            time_hours = float(geomech_info["time_hours"])
            flow_info = nearest_group(
                time_hours, flow_groups, args.time_tolerance_hours
            )
            if flow_info is None:
                raise RuntimeError(
                    f"No flow output matches geomechanics time {time_hours:.12g} h "
                    f"within {args.time_tolerance_hours:.3e} h"
                )

            print(
                f"[{output_index + 1}/{len(geomech_groups)}] "
                f"time={time_hours:.12g} h, "
                f"geomech={geomech_info['path']!r}, flow={flow_info['path']!r}"
            )

            geomech_arrays, geomech_names = load_group_arrays(
                geomech_h5[str(geomech_info["path"])], mechanics_count
            )
            flow_arrays_raw, flow_names = load_group_arrays(
                flow_h5[str(flow_info["path"])], flow_count, prefix="Flow_"
            )

            point_data = dict(geomech_arrays)
            for name, values in flow_arrays_raw.items():
                point_data[name] = map_flow_array(
                    values, flow_ids, mechanics_ids, mechanics_count
                )

            current_pressure_key = find_field_key(
                point_data,
                "Flow_Liquid_Pressure",
            )
            if (
                baseline_pressure_mapped is not None
                and current_pressure_key is not None
            ):
                current_pressure = np.asarray(
                    point_data[current_pressure_key],
                    dtype=np.float64,
                )
                point_data["Flow_Liquid_Pressure_Change_Pa"] = (
                    current_pressure - baseline_pressure_mapped
                )

            point_data.update(static_flags)

            has_displacement = add_vector(
                point_data,
                "Displacement",
                "displacement_x",
                "displacement_y",
                "displacement_z",
            )
            has_relative = add_vector(
                point_data,
                "Relative_Displacement",
                "relative_displacement_x",
                "relative_displacement_y",
                "relative_displacement_z",
            )
            print(f"  geomechanics fields: {len(geomech_arrays)}")
            print(f"  mapped flow fields:  {len(flow_arrays_raw)}")
            print(f"  Displacement vector: {has_displacement}")
            print(f"  Relative vector:     {has_relative}")

            mesh = meshio.Mesh(
                points=h5_points,
                cells=[("tetra", connectivity)],
                point_data=point_data,
            )
            vtu_path = output_dir / f"{prefix}_{output_index:04d}.vtu"
            try:
                meshio.write(vtu_path, mesh, binary=True, compression="zlib")
            except TypeError:
                meshio.write(vtu_path, mesh, binary=True)
            records.append((time_hours, vtu_path))
            manifest_times.append(
                {
                    "time_hours": time_hours,
                    "geomechanics_group": geomech_info["path"],
                    "flow_group": flow_info["path"],
                    "vtu": vtu_path.name,
                    "geomechanics_field_names": geomech_names,
                    "flow_field_names": flow_names,
                }
            )
            print(f"  wrote {vtu_path.name} ({vtu_path.stat().st_size / 2**20:.1f} MiB)")

    pvd_path = output_dir / f"{prefix}.pvd"
    write_pvd(pvd_path, records)
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "geomechanics_hdf5": str(geomech_path),
                "flow_hdf5": str(flow_path),
                "canonical_ugi": str(ugi_path),
                "mapping": str(mapping_path),
                "mechanics_vertex_count": mechanics_count,
                "tetrahedron_count": element_count,
                "times": manifest_times,
                "vset_flags": sorted(static_flags),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nCoupled conversion complete")
    print(f"  PVD series: {pvd_path}")
    print(f"  manifest:   {manifest_path}")
    print("  Warp By Vector -> Displacement")
    print("  Flow fields are prefixed with Flow_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
