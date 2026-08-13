#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import numpy as np

from tetgen_quality_report import (
    parse_tetgen_node,
    parse_tetgen_ele,
    build_quality_records,
)

MESH_PREFIX = "bartlesville_hec"

NODE_FILE = Path(f"{MESH_PREFIX}.1.node")
ELE_FILE = Path(f"{MESH_PREFIX}.1.ele")

OUT_BAD = Path(f"{MESH_PREFIX}_bad_tets_spatial.csv")
OUT_DEPTH = Path(f"{MESH_PREFIX}_bad_tets_by_depth.csv")

HEC_CENTER = np.array([5000.0, 5000.0, -527.5])

HEC_LENGTH = 580.0
HEC_WIDTH = 300.0

# HEC strike = 5 degrees east of north.
angle = math.radians(5.0)

length_axis = np.array([
    math.sin(angle),
    math.cos(angle),
])

width_axis = np.array([
    math.cos(angle),
    -math.sin(angle),
])


def hec_local_coordinates(x, y):
    delta = np.array([x - HEC_CENTER[0], y - HEC_CENTER[1]])
    u = float(np.dot(delta, length_axis))
    v = float(np.dot(delta, width_axis))
    return u, v


def depth_zone(z):
    if z < -535.0:
        return "underburden"
    elif z < -530.0:
        return "basal_layer"
    elif z < -525.0:
        return "HEC_interval"
    elif z < -500.0:
        return "Bartlesville"
    elif z < -400.0:
        return "lower_overburden"
    elif z < -250.0:
        return "middle_overburden"
    elif z < -100.0:
        return "upper_overburden"
    else:
        return "top_overburden"


def main():
    print(f"Reading {NODE_FILE} and {ELE_FILE}...")

    nodes = parse_tetgen_node(NODE_FILE)
    elements = parse_tetgen_ele(ELE_FILE)

    qualities, bad_mask = build_quality_records(nodes, elements)

    bad_count = 0
    depth_counts = {}

    with OUT_BAD.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "element_id",
            "centroid_x_m",
            "centroid_y_m",
            "centroid_z_m",
            "volume_m3",
            "edge_ratio",
            "radius_edge_ratio",
            "min_dihedral_deg",
            "quality_class",
            "distance_from_well_m",
            "hec_u_m",
            "hec_v_m",
            "inside_hec_planform",
            "depth_zone",
        ])

        for i, (tet, quality, is_bad) in enumerate(
            zip(elements, qualities, bad_mask),
            start=1,
        ):
            if not is_bad:
                continue

            points = np.array([nodes[n] for n in tet])
            centroid = points.mean(axis=0)

            x, y, z = centroid

            distance_from_well = float(
                np.hypot(
                    x - HEC_CENTER[0],
                    y - HEC_CENTER[1],
                )
            )

            u, v = hec_local_coordinates(x, y)

            inside_hec_planform = (
                abs(u) <= HEC_LENGTH / 2.0
                and abs(v) <= HEC_WIDTH / 2.0
            )

            zone = depth_zone(z)

            depth_counts.setdefault(
                zone,
                {
                    "bad_total": 0,
                    "below_1deg": 0,
                    "below_5deg": 0,
                    "below_10deg": 0,
                    "edge_ratio_gt_20": 0,
                    "edge_ratio_gt_25": 0,
                },
            )

            stats = depth_counts[zone]
            stats["bad_total"] += 1

            if quality["min_dihedral_deg"] < 1.0:
                stats["below_1deg"] += 1

            if quality["min_dihedral_deg"] < 5.0:
                stats["below_5deg"] += 1

            if quality["min_dihedral_deg"] < 10.0:
                stats["below_10deg"] += 1

            if quality["edge_ratio"] > 20.0:
                stats["edge_ratio_gt_20"] += 1

            if quality["edge_ratio"] > 25.0:
                stats["edge_ratio_gt_25"] += 1

            writer.writerow([
                i,
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{quality['volume_m3']:.8e}",
                f"{quality['edge_ratio']:.6f}",
                f"{quality['radius_edge_ratio']:.6f}",
                f"{quality['min_dihedral_deg']:.6f}",
                quality["quality_class"],
                f"{distance_from_well:.6f}",
                f"{u:.6f}",
                f"{v:.6f}",
                int(inside_hec_planform),
                zone,
            ])

            bad_count += 1

    with OUT_DEPTH.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "depth_zone",
            "bad_total",
            "below_1deg",
            "below_5deg",
            "below_10deg",
            "edge_ratio_gt_20",
            "edge_ratio_gt_25",
        ])

        for zone, stats in depth_counts.items():
            writer.writerow([
                zone,
                stats["bad_total"],
                stats["below_1deg"],
                stats["below_5deg"],
                stats["below_10deg"],
                stats["edge_ratio_gt_20"],
                stats["edge_ratio_gt_25"],
            ])

    print()
    print(f"Bad tetrahedra: {bad_count:,}")
    print(f"Wrote: {OUT_BAD}")
    print(f"Wrote: {OUT_DEPTH}")
    print()

    print("Bad tetrahedra by depth:")
    print(
        f"{'Zone':25s}"
        f"{'Total':>12s}"
        f"{'<1 deg':>12s}"
        f"{'<5 deg':>12s}"
        f"{'<10 deg':>12s}"
    )

    for zone, stats in depth_counts.items():
        print(
            f"{zone:25s}"
            f"{stats['bad_total']:12,d}"
            f"{stats['below_1deg']:12,d}"
            f"{stats['below_5deg']:12,d}"
            f"{stats['below_10deg']:12,d}"
        )


if __name__ == "__main__":
    main()
