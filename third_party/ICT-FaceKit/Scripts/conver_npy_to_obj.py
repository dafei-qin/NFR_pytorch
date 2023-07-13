import trimesh
import numpy as np
import os
from glob import glob


if __name__ == '__main__':
    path = 'D:/ICT_data_single_random'
    mesh_ref = trimesh.load('../sample_data_out/random_expression00_onlyface.obj', process=False, maintain_order=True)
    neutrals = glob(path + '/*neutral.npy')
    for n in neutrals:
        mesh_ref.vertices = np.load(n)
        mesh_ref.export(n.replace('.npy', '.obj'), 'obj')