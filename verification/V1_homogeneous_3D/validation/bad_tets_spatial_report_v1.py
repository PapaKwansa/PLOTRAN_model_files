import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# Repository root:
# verification/V1_homogeneous_3D/validation/script.py
#                  ^ parents[3] = repository root
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tetgen_quality_report import (
    parse_tetgen_node,
    parse_tetgen_ele,
    build_quality_records,
)


INJECTOR_CENTER = np.array(
    [5000.0, 5000.0, -529.75],
    dtype=float,
)

DOMAIN_MIN = np.array(
    [0.0, 0.0, -1535.0],
    dtype=float,
)

DOMAIN_MAX = np.array(
    [10000.0, 10000.0, 0.0],
    dtype=float,
)


RADIAL_EDGES_M = np.array(
    [
        0.0,
        1.25,
        1.8,
        2.6,
        3.8,
        5.5,
        8.0,
        12.0,
        18.0,
        27.0,
        40.0,
        60.0,
        90.0,
        135.0,
        200.0,
        500.0,
        1000.0,
        2500.0,
        np.inf,
    ],
    dtype=float,
)


def radial_bin_label(index):
    lo = RADIAL_EDGES_M[index]
    hi = RADIAL_EDGES_M[index + 1]

    if np.isinf(hi):
        return f">={lo:g} m"

    return f"{lo:g}-{hi:g} m"


def distance_to_outer_boundary(point):
    lower = point - DOMAIN_MIN
    upper = DOMAIN_MAX - point

    return float(
        np.min(
            np.concatenate(
                [lower, upper]
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Spatial diagnostic of poor/bad tetrahedra "
            "for the V1 homogeneous 3-D verification mesh."
        )
    )

    parser.add_argument(
        "--node",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ele",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_bad = (
        args.outdir
        / "v1_bad_tets_spatial.csv"
    )

    out_radial = (
        args.outdir
        / "v1_bad_tets_by_radius.csv"
    )

    out_boundary = (
        args.outdir
        / "v1_bad_tets_by_boundary_distance.csv"
    )

    print(
        f"Reading {args.node} and {args.ele}..."
    )

    nodes = parse_tetgen_node(
        args.node
    )

    elements = parse_tetgen_ele(
        args.ele
    )

    qualities, bad_mask = (
        build_quality_records(
            nodes,
            elements,
        )
    )

    radial_stats = {}

    for i in range(
        len(RADIAL_EDGES_M) - 1
    ):
        radial_stats[i] = {
            "total": 0,
            "below_1deg": 0,
            "below_5deg": 0,
            "below_10deg": 0,
            "edge_ratio_gt_20": 0,
            "edge_ratio_gt_50": 0,
        }

    boundary_edges = [
        0.0,
        10.0,
        50.0,
        125.0,
        250.0,
        500.0,
        1000.0,
        np.inf,
    ]

    boundary_stats = {}

    for i in range(
        len(boundary_edges) - 1
    ):
        boundary_stats[i] = {
            "total": 0,
            "below_1deg": 0,
            "below_5deg": 0,
        }

    bad_count = 0

    with out_bad.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "element_id",
                "centroid_x_m",
                "centroid_y_m",
                "centroid_z_m",
                "radial_distance_xy_m",
                "distance_3d_from_injector_m",
                "vertical_offset_from_injector_m",
                "distance_to_outer_boundary_m",
                "volume_m3",
                "edge_ratio",
                "radius_edge_ratio",
                "min_dihedral_deg",
                "quality_class",
            ]
        )

        for (
            element_index,
            (tet, quality, is_bad),
        ) in enumerate(
            zip(
                elements,
                qualities,
                bad_mask,
            ),
            start=1,
        ):

            if not is_bad:
                continue

            points = np.array(
                [
                    nodes[node_id]
                    for node_id in tet
                ],
                dtype=float,
            )

            centroid = points.mean(
                axis=0
            )

            delta = (
                centroid
                - INJECTOR_CENTER
            )

            radial_xy = float(
                np.hypot(
                    delta[0],
                    delta[1],
                )
            )

            distance_3d = float(
                np.linalg.norm(delta)
            )

            vertical_offset = float(
                delta[2]
            )

            boundary_distance = (
                distance_to_outer_boundary(
                    centroid
                )
            )

            radial_index = int(
                np.searchsorted(
                    RADIAL_EDGES_M,
                    radial_xy,
                    side="right",
                )
                - 1
            )

            radial_index = max(
                0,
                min(
                    radial_index,
                    len(RADIAL_EDGES_M) - 2,
                ),
            )

            rstats = radial_stats[
                radial_index
            ]

            rstats["total"] += 1

            angle = quality[
                "min_dihedral_deg"
            ]

            edge_ratio = quality[
                "edge_ratio"
            ]

            if angle < 1.0:
                rstats["below_1deg"] += 1

            if angle < 5.0:
                rstats["below_5deg"] += 1

            if angle < 10.0:
                rstats["below_10deg"] += 1

            if edge_ratio > 20.0:
                rstats[
                    "edge_ratio_gt_20"
                ] += 1

            if edge_ratio > 50.0:
                rstats[
                    "edge_ratio_gt_50"
                ] += 1

            boundary_index = int(
                np.searchsorted(
                    boundary_edges,
                    boundary_distance,
                    side="right",
                )
                - 1
            )

            boundary_index = max(
                0,
                min(
                    boundary_index,
                    len(boundary_edges) - 2,
                ),
            )

            bstats = boundary_stats[
                boundary_index
            ]

            bstats["total"] += 1

            if angle < 1.0:
                bstats["below_1deg"] += 1

            if angle < 5.0:
                bstats["below_5deg"] += 1

            writer.writerow(
                [
                    element_index,
                    f"{centroid[0]:.8f}",
                    f"{centroid[1]:.8f}",
                    f"{centroid[2]:.8f}",
                    f"{radial_xy:.8f}",
                    f"{distance_3d:.8f}",
                    f"{vertical_offset:.8f}",
                    f"{boundary_distance:.8f}",
                    f"{quality['volume_m3']:.12e}",
                    f"{edge_ratio:.8f}",
                    f"{quality['radius_edge_ratio']:.8f}",
                    f"{angle:.8f}",
                    quality["quality_class"],
                ]
            )

            bad_count += 1

    with out_radial.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "radial_bin",
                "bad_total",
                "below_1deg",
                "below_5deg",
                "below_10deg",
                "edge_ratio_gt_20",
                "edge_ratio_gt_50",
            ]
        )

        for i, stats in radial_stats.items():

            writer.writerow(
                [
                    radial_bin_label(i),
                    stats["total"],
                    stats["below_1deg"],
                    stats["below_5deg"],
                    stats["below_10deg"],
                    stats[
                        "edge_ratio_gt_20"
                    ],
                    stats[
                        "edge_ratio_gt_50"
                    ],
                ]
            )

    with out_boundary.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "boundary_distance_bin_m",
                "bad_total",
                "below_1deg",
                "below_5deg",
            ]
        )

        for i, stats in boundary_stats.items():

            lo = boundary_edges[i]
            hi = boundary_edges[i + 1]

            if np.isinf(hi):
                label = f">={lo:g}"
            else:
                label = (
                    f"{lo:g}-{hi:g}"
                )

            writer.writerow(
                [
                    label,
                    stats["total"],
                    stats["below_1deg"],
                    stats["below_5deg"],
                ]
            )

    print()
    print(
        f"Poor/bad tetrahedra: "
        f"{bad_count:,}"
    )

    print()
    print(
        "Poor/bad tetrahedra by "
        "horizontal distance from injector:"
    )

    print(
        f"{'Radius':>16s}"
        f"{'Total':>10s}"
        f"{'<1 deg':>10s}"
        f"{'<5 deg':>10s}"
        f"{'<10 deg':>10s}"
        f"{'ER>20':>10s}"
        f"{'ER>50':>10s}"
    )

    for i, stats in radial_stats.items():

        print(
            f"{radial_bin_label(i):>16s}"
            f"{stats['total']:10,d}"
            f"{stats['below_1deg']:10,d}"
            f"{stats['below_5deg']:10,d}"
            f"{stats['below_10deg']:10,d}"
            f"{stats['edge_ratio_gt_20']:10,d}"
            f"{stats['edge_ratio_gt_50']:10,d}"
        )

    print()
    print(
        "Poor/bad tetrahedra by distance "
        "to outer domain boundary:"
    )

    print(
        f"{'Boundary dist.':>18s}"
        f"{'Total':>10s}"
        f"{'<1 deg':>10s}"
        f"{'<5 deg':>10s}"
    )

    for i, stats in boundary_stats.items():

        lo = boundary_edges[i]
        hi = boundary_edges[i + 1]

        if np.isinf(hi):
            label = f">={lo:g} m"
        else:
            label = (
                f"{lo:g}-{hi:g} m"
            )

        print(
            f"{label:>18s}"
            f"{stats['total']:10,d}"
            f"{stats['below_1deg']:10,d}"
            f"{stats['below_5deg']:10,d}"
        )

    print()
    print(f"Wrote: {out_bad}")
    print(f"Wrote: {out_radial}")
    print(f"Wrote: {out_boundary}")


if __name__ == "__main__":
    main()
