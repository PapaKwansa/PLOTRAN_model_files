#!/usr/bin/env python3
import numpy as np
import sys
import os

def write_vset(filename: str, indices_1based: np.ndarray):
    with open(filename, "w") as f:
        for idx in indices_1based:
            f.write(f"{int(idx)}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 boundary_nodes_to_vset.py <mesh_boundaries.txt> [mesh_materials.txt]")
        sys.exit(1)

    bfile = sys.argv[1].strip()
    if not os.path.exists(bfile):
        print(f"Error: not found: {bfile}")
        sys.exit(1)

    # Optional materials file
    mfile = None
    if len(sys.argv) >= 3:
        mfile = sys.argv[2].strip()
        if not os.path.exists(mfile):
            print(f"Error: not found materials file: {mfile}")
            sys.exit(1)
    else:
        # Auto-detect: <pre>_boundaries.txt -> <pre>_materials.txt
        if bfile.endswith("_boundaries.txt"):
            pre = bfile[:-len("_boundaries.txt")]
            candidate = pre + "_materials.txt"
            if os.path.exists(candidate):
                mfile = candidate

    btags = np.loadtxt(bfile, dtype=int)

    # boundary tag convention
    tag_to_name = {
        1: "top",
        2: "bottom",
        3: "north",
        4: "south",
        5: "east",
        6: "west",
        7: "wellbore",
    }

    for tag, name in tag_to_name.items():
        inds = np.where(btags == tag)[0] + 1  # 1-based node ids
        vset_name = f"{name}.vset"
        write_vset(vset_name, inds)
        print(f"Wrote {vset_name}: {inds.size} nodes")

    # Layer vsets from materials
    if mfile is not None:
        matids = np.loadtxt(mfile, dtype=int)

        if matids.shape[0] != btags.shape[0]:
            print(f"WARNING: materials size ({matids.shape[0]}) != boundaries size ({btags.shape[0]}). Skip layer vsets.")
            return

        # material id -> vset name (your requested names)
        layer_map = {
            4: "overburden.vset",
            2: "bartlesville_sand.vset",
            3: "basal_layer.vset",
            1: "underburden.vset",
        }

        print("\n--> Also writing layer vsets (from materials):", os.path.basename(mfile))
        for mid, vname in layer_map.items():
            inds = np.where(matids == mid)[0] + 1
            write_vset(vname, inds)
            print(f"Wrote {vname}: {inds.size} nodes (matid={mid})")
    else:
        print("\nNOTE: No *_materials.txt found (or provided). Layer vsets not generated.")

if __name__ == "__main__":
    main()
