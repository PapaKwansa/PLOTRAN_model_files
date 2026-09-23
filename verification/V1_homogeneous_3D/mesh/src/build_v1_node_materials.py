#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import yaml


def read_tetgen_nodes(path: Path):
    node_ids = []
    xyz = []

    with path.open("r", encoding="utf-8") as handle:
        rows = (
            line.split("#", 1)[0].strip()
            for line in handle
        )
        rows = [line for line in rows if line]

    header = rows[0].split()
    expected = int(header[0])

    for line in rows[1:1 + expected]:
        fields = line.split()

        node_ids.append(int(fields[0]))

        xyz.append(
            [
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
            ]
        )

    if len(xyz) != expected:
        raise RuntimeError(
            f"Expected {expected} nodes, found {len(xyz)}."
        )

    return (
        np.asarray(node_ids, dtype=np.int64),
        np.asarray(xyz, dtype=float),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build node-centered V1 material IDs "
            "for the PFLOTRAN median-dual flow grid."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--node",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    with args.config.open(
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(handle)

    injector = cfg["injector"]
    material_ids = cfg["mesh"]["material_ids"]

    cx = float(injector["center_m"]["x"])
    cy = float(injector["center_m"]["y"])
    cz = float(injector["center_m"]["z"])

    radius = float(injector["radius_m"])
    height = float(injector["height_m"])

    matrix_id = int(material_ids["matrix"])
    injector_id = int(material_ids["injector"])

    node_ids, xyz = read_tetgen_nodes(args.node)

    radial = np.hypot(
        xyz[:, 0] - cx,
        xyz[:, 1] - cy,
    )

    zmin = cz - 0.5 * height
    zmax = cz + 0.5 * height

    tolerance = 1.0e-8

    injector_mask = (
        (radial <= radius + tolerance)
        & (xyz[:, 2] >= zmin - tolerance)
        & (xyz[:, 2] <= zmax + tolerance)
    )

    materials = np.full(
        xyz.shape[0],
        matrix_id,
        dtype=np.int64,
    )

    materials[injector_mask] = injector_id

    if not np.any(injector_mask):
        raise RuntimeError(
            "No injector nodes were identified."
        )

    unique, counts = np.unique(
        materials,
        return_counts=True,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        args.output,
        materials,
        fmt="%d",
    )

    print("V1 node-centered materials generated")
    print("=" * 60)
    print(f"Node file       : {args.node}")
    print(f"Output          : {args.output}")
    print(f"Total nodes     : {xyz.shape[0]:,}")
    print()
    print("Material counts")

    for material_id, count in zip(
        unique,
        counts,
    ):
        print(
            f"  material {material_id}: "
            f"{count:,}"
        )

    injector_xyz = xyz[injector_mask]

    print()
    print("Injector-node coordinate bounds")
    print(
        f"  x : {injector_xyz[:,0].min():.8f} "
        f"to {injector_xyz[:,0].max():.8f}"
    )
    print(
        f"  y : {injector_xyz[:,1].min():.8f} "
        f"to {injector_xyz[:,1].max():.8f}"
    )
    print(
        f"  z : {injector_xyz[:,2].min():.8f} "
        f"to {injector_xyz[:,2].max():.8f}"
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
