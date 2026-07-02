import sys

def generate_ugi(ele_file, node_file, output_ugi):
    # Read .ele file
    with open(ele_file, 'r') as f:
        ele_lines = f.readlines()
    
    # Read .node file
    with open(node_file, 'r') as f:
        node_lines = f.readlines()
    
    # Extract number of elements and nodes
    num_elements = int(ele_lines[0].split()[0])
    num_nodes = int(node_lines[0].split()[0])
    
    # Process .ele data
    elements = []
    for line in ele_lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) > 1:
            elements.append(f"T  {'  '.join(parts[1:-1])}")  # Exclude last material ID column
    
    # Process .node data
    nodes = []
    for line in node_lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) > 1:
            nodes.append(f"{parts[1]}  {parts[2]}  {parts[3]}")
    
    # Write .ugi file
    with open(output_ugi, 'w') as f:
        f.write(f"{num_elements} {num_nodes}\n")
        f.write('\n'.join(elements) + '\n')
        f.write('\n'.join(nodes) + '\n')
    
    print(f"UGI file generated: {output_ugi}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_ugi.py <ele_file> <node_file> <output_ugi>")
        sys.exit(1)
    
    ele_file = sys.argv[1]
    node_file = sys.argv[2]
    output_ugi = sys.argv[3]
    
    generate_ugi(ele_file, node_file, output_ugi)
