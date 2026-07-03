#!/usr/bin/env python3
"""Create a small PFLOTRAN source-region vertex set for the injection well.

This writes wellbore.vset from a TetGen .node file by selecting node ids inside
a cylindrical source tube around the injection center.

Typical use:
    python3 make_wellbore_vset.py bartlesville_hec.1.node wellbore.vset \
        --center-x 5000 --center-y 5000 --radius 10 \
        --z-min 532.5 --z-max 750.0

The PFLOTRAN deck can then keep:
    SOURCE_SINK inj
      FLOW_CONDITION injection
      REGION wellbore
    END

If you also want a wellbore.ex file for visualization/consistency, add
wellbore.vset to the existing vset->ex conversion loop in workflow.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple


def data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                yield line


def read_node_file(node_file: Path) -> List[Tuple[int, float, float, float]]:
    lines = iter(data_lines(node_file))
    try:
        header = next(lines).split()
    except StopIteration as exc:
        raise RuntimeError(f"Empty node file: {node_file}") from exc

    if len(header) < 4:
        raise RuntimeError(f"Malformed node header in {node_file}: {' '.join(header)}")

    count = int(float(header[0]))
    nodes: List[Tuple[int, float, float, float]] = []
    for row in range(count):
        try:
            fields = next(lines).split()
        except StopIteration as exc:
            raise RuntimeError(f"{node_file}: expected {count} node rows, stopped at {row}") from exc
        if len(fields) < 4:
            raise RuntimeError(f"{node_file}: malformed node row {row + 1}: {' '.join(fields)}")
        node_id = int(float(fields[0]))
        x = float(fields[1])
        y = float(fields[2])
        z = float(fields[3])
        nodes.append((node_id, x, y, z))
    return nodes


def build_wellbore_vset(
    node_file: Path,
    output_vset: Path,
    center_x: float,
    center_y: float,
    radius: float,
    z_min: float,
    z_max: float,
) -> List[int]:
    nodes = read_node_file(node_file)
    radius2 = radius * radius
    selected: List[int] = []

    for node_id, x, y, z in nodes:
        if z < z_min or z > z_max:
            continue
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius2:
            selected.append(node_id)

    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError(
            "No nodes were selected for wellbore.vset. "
            "Try a slightly larger radius or check the center coordinates."
        )

    with output_vset.open("w", encoding="utf-8") as handle:
        for node_id in selected:
            handle.write(f"{node_id}\n")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create wellbore.vset from a TetGen .node file.")
    parser.add_argument("node_file", type=Path, help="TetGen .node file, e.g. bartlesville_hec.1.node")
    parser.add_argument("output_vset", type=Path, help="Output vset file, e.g. wellbore.vset")
    parser.add_argument("--center-x", type=float, default=5000.0, help="Injection center x coordinate (m)")
    parser.add_argument("--center-y", type=float, default=5000.0, help="Injection center y coordinate (m)")
    parser.add_argument("--radius", type=float, default=10.0, help="Selection radius in meters")
    parser.add_argument("--z-min", type=float, default=532.5, help="Minimum z coordinate in meters")
    parser.add_argument("--z-max", type=float, default=750.0, help="Maximum z coordinate in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = build_wellbore_vset(
        args.node_file,
        args.output_vset,
        args.center_x,
        args.center_y,
        args.radius,
        args.z_min,
        args.z_max,
    )
    print(f"Wrote {args.output_vset} with {len(selected)} node ids")


if __name__ == "__main__":
    main()
