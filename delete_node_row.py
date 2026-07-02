#!/usr/bin/env python3

import os
import sys

def delete_last_row_of_node_file(mesh_name):
    """
    Delete the last row of the .node file for the given mesh.
    """
    filename = f"{mesh_name}.1.node"  # File is in the current directory

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in the current directory.")
        return
    
    # Read all lines from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Ensure there is more than just a header
    if len(lines) <= 1:
        print("Error: The file does not have enough data to remove a row.")
        return
    
    # Write back the file without the last row
    with open(filename, 'w') as file:
        file.writelines(lines[:-1])  # Write all but the last line
    
    print(f"Successfully removed the last row from '{filename}'.")

if __name__ == "__main__":
    # Get mesh name from command-line argument or use default
    mesh_name = sys.argv[1] if len(sys.argv) > 1 else "TC_filled"
    delete_last_row_of_node_file(mesh_name)
