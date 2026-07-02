import numpy as np
import argparse

def vset2ex(uge_file, vset_file, boundary_cell_area=10, epsilon=1.e-6):
    """
    Convert vset files into ex format 
    
    
    """
    print('*' * 80)
    print('--> Converting vset files to ex')

    # Opening uge file
    print('\n--> Opening uge file')
    with open(uge_file, 'r') as fuge:
        # Reading cell ids, cells centers and cell volumes
        line = fuge.readline()
        line = line.split()
        num_cells = int(line[1])

        cell_id = np.zeros(num_cells, 'int')
        cell_coord = np.zeros((num_cells, 3), 'float')
        cell_vol = np.zeros(num_cells, 'float')

        for cells in range(num_cells):
            line = fuge.readline()
            line = line.split()
            cell_id[cells] = int(line.pop(0))
            line = [float(id) for id in line]
            cell_vol[cells] = line.pop(3)
            cell_coord[cells] = line

    print('--> Finished processing uge file\n')

    
    # Ex filename
    ex_file = vset_file.replace('vset', 'ex')

    # Opening the input file
    print('--> Opening vset file: ', vset_file)
    with open(vset_file, 'r') as fvset:
        print('--> Reading boundary node ids')
        node_array = fvset.readlines()
        # node_array = node_array.split()
        num_nodes = len(node_array)
        node_array = np.array(node_array, dtype='int')
    print('--> Finished reading vset file')

    Boundary_cell_area_array = np.zeros(num_nodes, 'float')
    for i in range(num_nodes):
        Boundary_cell_area_array[
            i] = boundary_cell_area  # Fix the area to a large number

    print('--> Finished calculating boundary connections')
    boundary_cell_coord = [
        cell_coord[cell_id[i - 1] - 1] for i in node_array
    ]

  
    boundary_cell_coord = [[cell[0], cell[1], cell[2] + epsilon]
                            for cell in boundary_cell_coord]

    ## Write out ex files
    with open(ex_file, 'w') as f:
        f.write('CONNECTIONS\t%i\n' % node_array.size)
        for idx, cell in enumerate(boundary_cell_coord):
            f.write(
                f"{node_array[idx]}\t{cell[0]:.12e}\t{cell[1]:.12e}\t{cell[2]:.12e}\t{Boundary_cell_area_array[idx]:.12e}\n"
            )

    print(
        f'--> Finished writing ex file {ex_file} corresponding to the vset file: {vset_file} \n'
    )

    print('--> Converting vset files to ex complete')
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vset files to PFLOTRAN ex format')
    parser.add_argument('uge_file', help='PFLOTRAN explicit mesh uge file')
    parser.add_argument('vset_file', help='PFLOTRAN vertex set file')
    parser.add_argument('-b', '--boundary_cell_area', type=float, default=10, help='Boundary cell area')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-6, help='Epsilon value')

    args = parser.parse_args()

    vset2ex(args.uge_file, args.vset_file, args.boundary_cell_area, args.epsilon)
