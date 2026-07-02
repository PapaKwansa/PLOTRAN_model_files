#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Get mesh name from command-line argument or use default
if len(sys.argv) > 1:
    mesh_name = sys.argv[1].strip()  # Strip any accidental whitespace

# Define paths correctly
needed_files_dir = "needed_files"
mapping_filename = "example.mapping"
file_path = os.path.join(needed_files_dir, mapping_filename)  # Ensure the correct file path
node_file = f"{mesh_name}.1.node"  # Mesh node file

# Validate file existence
if not os.path.exists(file_path):
    print(f"Error: Required file '{mapping_filename}' not found in directory '{needed_files_dir}'.")
    sys.exit(1)

if not os.path.exists(node_file):
    print(f"Error: Node file '{node_file}' not found.")
    sys.exit(1)

file_two_columns = f"{mesh_name}.mapping"
file_one_column = f"{mesh_name}_all.vset"

# Check if example.mapping exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found in {needed_files_dir}.")
    sys.exit(1)

# Read the first line of the .1.node file and extract only the first number
if not os.path.exists(node_file):
    print(f"Error: File '{node_file}' not found in the current directory.")
    sys.exit(1)

with open(node_file, "r") as f:
    first_line = f.readline().strip().split()  # Read first line and split by whitespace
    target_rows = int(first_line[0])  # Convert the first value to an integer

print(f"Detected {target_rows} rows from {node_file}.")

# Load the original mapping file with corrected separator
df = pd.read_csv(file_path, sep='\s+', header=None)  # Fix deprecation warning

# Ensure we have enough rows before filtering
if df.shape[0] < target_rows:
    # Determine the number of missing rows
    missing_rows = target_rows - df.shape[0]
    
    # Create additional rows by incrementing the last row's values by 1
    last_values = df.iloc[-1, :2].values
    new_rows = [last_values + np.array([i + 1, i + 1]) for i in range(missing_rows)]
    
    # Convert to DataFrame and append
    df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
else:
    df_extended = df.iloc[:target_rows, :]

# Save the first two columns to a new file
df_extended.iloc[:target_rows, :2].to_csv(file_two_columns, sep=" ", index=False, header=False)

# Save only the first column to another file
df_extended.iloc[:target_rows, 0].to_csv(file_one_column, index=False, header=False)

print(f"Files generated:\n- {file_two_columns}\n- {file_one_column}")
