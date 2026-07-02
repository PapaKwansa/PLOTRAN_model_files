#!/usr/bin/env python3

import os
import sys

def delete_lines_with_keyword(mesh_name):
    """
    Remove lines containing the keyword "Generated" from the .ugi file.
    """
    filename = f"{mesh_name}.ugi"  # File is in the current directory

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in the current directory.")
        return
    
    # Read all lines from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Filter out lines containing the keyword "Generated"
    filtered_lines = [line for line in lines if "Generated" not in line]

    # Write back the file without the removed lines
    with open(filename, 'w') as file:
        file.writelines(filtered_lines)
    
    print(f"Successfully removed lines containing 'Generated' from '{filename}'.")

if __name__ == "__main__":
    # Get mesh name from command-line argument or use default
    if len(sys.argv) > 1:
        mesh_name = sys.argv[1]
    else:
        print("Usage: python3 script.py <mesh_name>")
        sys.exit(1)

    delete_lines_with_keyword(mesh_name)
