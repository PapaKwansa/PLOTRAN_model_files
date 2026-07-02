import numpy as np
import argparse
from h5py import *


def read_materials_file(material_file):
    """
    Reads a material file with one column of data and returns a numpy array where each row contains the node number
    and material id.

    :param material_file: The path to the materials file.
    :return: A numpy array of shape (num_points, 2) containing the node number and material id.
    """
    with open(material_file, 'r') as f:
        materials = [int(line.strip()) for line in f]  # Read each line as an integer representing material ID

    num_points = len(materials)  # The number of points is the number of lines in the file

    # Create a numpy array with two columns: one for node number and one for material ID
    matids = np.zeros((num_points, 2), dtype=int)

    for i in range(num_points):
        matids[i, 0] = i + 1  # Node number starts from 1
        matids[i, 1] = materials[i]  # Material ID from the file

    return matids


def write_to_h5(matids, file):
    """
    Reads the node number, material id numpy array and writes to h5 file

    :param matids: The numpy array
    """
    h5file = File(file, mode='w')
    h5grp = h5file.create_group('Materials')
    
    dataset_name = 'Materials/Cell Ids'
    h5dset = h5file.create_dataset(dataset_name, data=matids[:,0])
    
    dataset_name = 'Materials/Material Ids'
    h5dset = h5file.create_dataset(dataset_name, data=matids[:,1])
    
    h5file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dumps the material.h5 for PFLOTRAN from TetGen info.')
    parser.add_argument('node_file', type=str, help='The path to the TetGen node file.')
    parser.add_argument('h5_file', default='material_ids.h5', type=str, help='The name of the output material h5 file.')
    parser.add_argument('material_file', type=str, help='The path to the materials file.')

    args = parser.parse_args()

    # Read the materials file
    matids = read_materials_file(args.material_file)

    # Write the data to an h5 file
    write_to_h5(matids, args.h5_file)
