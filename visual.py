import subprocess
import os
import sys
import glob
import multiprocessing as mp

mesh_name = "surf2"


# Voronoi visualization
user_input = input("Generate Voronoi visualization files (Materials 2 to 11)? (Y/N): ").strip().lower()

if user_input not in ['yes', 'y']:
    print("Visualization generation skipped. Exiting program.")
    sys.exit()

print("--> Generating visualization files in parallel (Materials 2 to 11)...")

def run_viz(mat_id):
    cmd = f"python3 new_tet_voro_vtu_manual.py {mesh_name} {mat_id}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Error Material {mat_id}]: {result.stderr}")
    else:
        print(f"[Material {mat_id}] Completed.")

material_ids = list(range(2, 12))  # explicitly from 2 to 11 inclusive

with mp.Pool(mp.cpu_count()) as pool:
    pool.map(run_viz, material_ids)

print("\nAll visualization files have been generated successfully.")
print('*' * 80)