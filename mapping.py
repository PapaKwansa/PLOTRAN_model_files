#!/usr/bin/env python3
"""
Generate a mesh-consistent mapping file from the current TetGen .node file.

This version does NOT use needed_files/example.mapping.
It writes:
  - <mesh>.mapping
  - <mesh>_all.vset

The mapping is a simple 1..N identity mapping in node-row order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python mapping.py <mesh_name>")

    mesh_name = sys.argv[1].strip()
    node_file = Path(f"{mesh_name}.1.node")

    if not node_file.is_file():
        raise FileNotFoundError(f"Node file not found: {node_file}")

    # Read node count from the header.
    with node_file.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().split()
        if not first_line:
            raise RuntimeError(f"Malformed or empty node file: {node_file}")
        target_rows = int(float(first_line[0]))

    print(f"Detected {target_rows} rows from {node_file}.")

    # Identity mapping in node-row order:
    # col1 = row/node id, col2 = same id.
    ids = pd.DataFrame(
        {
            0: range(1, target_rows + 1),
            1: range(1, target_rows + 1),
        }
    )

    file_two_columns = Path(f"{mesh_name}.mapping")
    file_one_column = Path(f"{mesh_name}_all.vset")

    ids.to_csv(file_two_columns, sep=" ", index=False, header=False)
    ids.iloc[:, 0].to_csv(file_one_column, index=False, header=False)

    print(f"Files generated:\n- {file_two_columns}\n- {file_one_column}")


if __name__ == "__main__":
    main()