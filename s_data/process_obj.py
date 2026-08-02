import numpy as np
import argparse
import trimesh 
import os
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('./')
from myutils.write_pc2 import writePC2
from myutils.tensors import *
from myutils.myutils import transform_apply
import pickle
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--transform', help='The transformation matrix to roughly align the meshes to the center, you can use blender to manually align the mesh, and construct the transformation matrix by calling the myutils.blender2transform() fuction')

    
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    # files = sorted(glob('./BIWI/npy/*npy'))
    with open(args.transform, 'rb') as f:
        R = pickle.load(f)
    save_dir = os.path.join(args.output_dir, 'npy')
    save_dir_pc2 = os.path.join(args.output_dir, 'pc2')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_pc2, exist_ok=True)

    for idx, dirs in enumerate(sorted(os.listdir(args.data_dir))): # Identities
        
        files = sorted(glob(os.path.join(args.data_dir, dirs, '*.obj')))
        if len(files) == 0:
            continue
        mesh_neutral = trimesh.load(files[0], process=False, maintain_order=True) # For the geometry
        mesh_neutral.vertices = transform_apply(mesh_neutral.vertices, R)
        data = []
        for f in files:
            d = trimesh.load(f, process=False, maintain_order=True)
            d = d.vertices
            data.append(d[np.newaxis])
        data = np.concatenate(data, axis=0)
        data = transform_apply(data, R)

        mesh_neutral.export(os.path.join(save_dir, f'{idx:03d}' + '_neutral.obj'))
        np.save(os.path.join(save_dir, f'{idx:03d}' + '.npy'), data)
        writePC2(os.path.join(save_dir_pc2, f'{idx:03d}' + '.pc2'), data)
    
