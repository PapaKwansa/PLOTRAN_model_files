
#!/usr/bin/env python3
"""
Scan a TetGen-style .node file and count nodes tagged as 1001 and 1002.

Usage:
  python count_fracture_nodes.py --file surf2.1.node
"""

import argparse
from typing import Tuple

def parse_header(line: str) -> Tuple[int, int, int, int]:
    """
    Parse the first non-comment, non-empty header line of a .node file.
    Expected format (TetGen):
        <# of nodes> <dimension> <# of attributes> <# of boundary markers>
    Returns: (num_nodes, dim, n_attr, n_markers)
    """
    # Remove inline comments after '#'
    line = line.split('#', 1)[0].strip()
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(
            f"Header line should have at least 4 columns, got {len(parts)}: {line!r}"
        )
    try:
        num_nodes = int(float(parts[0]))
        dim = int(float(parts[1]))
        n_attr = int(float(parts[2]))
        n_markers = int(float(parts[3]))
    except ValueError as e:
        raise ValueError(f"Failed to parse header numbers from line: {line!r}") from e
    return num_nodes, dim, n_attr, n_markers


def count_fracture_nodes(node_file: str) -> Tuple[int, int, int, int]:
    """
    Count nodes tagged as 1001 and 1002 in a .node file.

    Returns:
        total_nodes_declared (from header),
        count_1001,
        count_1002,
        total_rows_parsed (data lines parsed; may be <= declared)
    """
    with open(node_file, 'r', encoding='utf-8', errors='ignore') as f:
        # Read lines until we find a non-empty, non-comment header line
        header = None
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                continue
            header = stripped
            break

        if header is None:
            raise ValueError("File appears empty or contains only comments/blank lines.")

        total_nodes_declared, dim, n_attr, n_markers = parse_header(header)

        # Counters
        count_1001 = 0
        count_1002 = 0
        data_lines_parsed = 0

        for raw in f:
            # Strip inline comments
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            # A valid node line should at least have: index x y [z] ... tag
            # In your example: index, x, y, z, attr, tag  (6 cols)
            # We'll accept 5+ columns and rely on the last token as the tag.
            if len(parts) < 2:
                # Too short to be useful; skip
                continue

            # Last column is the tag/marker
            try:
                tag = int(float(parts[-1]))
            except ValueError:
                # If the last token isn't numeric, skip this line
                continue

            if tag == 1001:
                count_1001 += 1
            elif tag == 1002:
                count_1002 += 1

            data_lines_parsed += 1

    return total_nodes_declared, count_1001, count_1002, data_lines_parsed


def main():
    parser = argparse.ArgumentParser(description="Count fracture (1001/1002) nodes in a .node file.")
    parser.add_argument(
        "-f", "--file", default="surf2.1.node",
        help="Path to the .node file (default: surf2.1.node)"
    )
    args = parser.parse_args()

    total_nodes_declared, c1001, c1002, parsed = count_fracture_nodes(args.file)
    frac_total = c1001 + c1002

    # Use the declared total from header as the denominator (per your request)
    denom = total_nodes_declared if total_nodes_declared > 0 else parsed
    portion = (frac_total / denom) if denom else 0.0

    print("=== Fracture Node Summary ===")
    print(f"File: {args.file}")
    print(f"Declared total nodes (header): {total_nodes_declared:,}")
    if parsed != total_nodes_declared:
        print(f"Note: data lines parsed: {parsed:,} (may differ from header count)")
    print(f"Tagged 1001 count: {c1001:,}")
    print(f"Tagged 1002 count: {c1002:,}")
    print(f"Total fracture nodes (1001+1002): {frac_total:,}")
    print(f"Portion of fracture nodes over total: {portion:.6%}")


if __name__ == "__main__":
    main()
