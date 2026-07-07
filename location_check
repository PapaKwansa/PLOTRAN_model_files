import re
from pathlib import Path
import pandas as pd

UGE_FILE = Path("bartlesville_hec.uge")
VSET_FILE = Path("injection_borehole.vset")   # change if needed

def load_uge_cell_table(uge_path):
    with open(uge_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n_cells = None
    header_idx = None
    for i, line in enumerate(lines):
        m = re.search(r"\bCELLS\s+(\d+)\b", line)
        if m:
            n_cells = int(m.group(1))
            header_idx = i
            break
    if n_cells is None:
        raise ValueError("Could not find CELLS header")

    rows = []
    for line in lines[header_idx + 1:]:
        if len(rows) >= n_cells:
            break
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    return pd.DataFrame(rows, columns=["cell_id", "x_m", "y_m", "z_m", "vol_m3"])

def load_vset_ids(vset_path):
    ids = []
    with open(vset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            ids.extend(int(x) for x in re.findall(r"-?\d+", line))

    # If the first integer is a count header, drop it
    if len(ids) > 1 and ids[0] == len(ids) - 1:
        ids = ids[1:]

    return ids

uge_df = load_uge_cell_table(UGE_FILE)
ids = load_vset_ids(VSET_FILE)

# If your vset IDs are 1-based cell IDs, this is correct:
cells = uge_df[uge_df["cell_id"].isin(ids)].copy()

print("Number of cells:", len(cells))
print("z-min:", cells["z_m"].min())
print("z-max:", cells["z_m"].max())

print("\nLowest 10 cells:")
print(cells.sort_values("z_m")[["cell_id", "x_m", "y_m", "z_m"]].head(10).to_string(index=False))

print("\nHighest 10 cells:")
print(cells.sort_values("z_m")[["cell_id", "x_m", "y_m", "z_m"]].tail(10).to_string(index=False))