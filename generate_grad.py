import numpy as np
import pickle
import os

from myutils.myutils import Mesh, get_gradient_info
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
import argparse
class ds(torch.utils.data.Dataset):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = torch.from_numpy(faces)

    def __len__(self):
        return len(self.verts)
    def __getitem__(self, idx):
        return self.verts[idx], self.faces

def render_dataset(dataset_path):
    import potpourri3d as pp3d
    from diffusion_net.geometry import  build_tangent_frames
    import scipy
    counter = 0
    files = sorted(glob(os.path.join(dataset_path, '*', '*.pkl')))
    # print(files)
    for idx, f in enumerate(tqdm(files)):
        datatype = f.split('/')[-2]

        # Neutral
        try:
            neutral_obj = Mesh.load(os.path.join(dataset_path, os.path.basename(f).replace('_processed_matrix.pkl', '_neutral.obj')))
        except:
            neutral_obj = Mesh.load(os.path.join(dataset_path, os.path.basename(f).replace('_processed_matrix.pkl', '.obj')))

        neutral_obj.vertices = neutral_obj.vertices.astype(np.float32)
        
        # Data
        try:
            with open(f, 'rb') as ff:
                face, verts, gradients= pickle.load(ff).values()
        except ValueError:
            with open(f, 'rb') as ff:
                face, verts, gradients, _= pickle.load(ff).values()

        verts= verts.astype(np.float32)
        np.random.shuffle(verts)
        # if datatype != 'train':
        #     continue
        
        gradX_list = []
        gradY_list = []
        mesh = Mesh(verts[0], face)
        N = verts.shape[1]
        vert_edge_outgoing = [[] for i in range(N)]
        L = pp3d.cotan_laplacian(mesh.vertices, mesh.faces, denom_eps=1e-10) # Why use the neutral to calculate cotan laplacian? But actually L is not used in the following code.
        L_coo = L.tocoo()
        inds_row = L_coo.row
        inds_col = L_coo.col
        edges_np = np.stack((inds_row, inds_col), axis=0)
        rhs_mat_list = []
        row_inds = []
        col_inds = []
        for iE in range(edges_np.shape[1]):
            tail_ind = edges_np[0, iE]
            tip_ind = edges_np[1, iE]
            if tip_ind != tail_ind:
                vert_edge_outgoing[tail_ind].append(iE)
        
        for iV in range(N):
            n_neigh = len(vert_edge_outgoing[iV])
            iEs = vert_edge_outgoing[iV]
            jVs = edges_np[1][iEs]
            ind_lookup_ = np.zeros(len(jVs) + 1)
            ind_lookup_[0] = iV
            ind_lookup_[1:] = jVs
            rhs_mat_ = np.eye(n_neigh)
            rhs_mat_ = np.concatenate([np.zeros((n_neigh, 1)) - 1, rhs_mat_], axis=-1)
            rhs_mat_list.append(rhs_mat_)
            for i_neigh in range(n_neigh + 1):
                i_glob = ind_lookup_[i_neigh]

                row_inds.append(iV)
                col_inds.append(i_glob)
        row_inds = np.array(row_inds)        
        col_inds = np.array(col_inds)
        edges = torch.from_numpy(edges_np).long()
        # indices = None
        for idx in range(0, len(verts)):
            counter += 1
            v = verts[idx]
            mesh = Mesh(v, face)
            indices, gradX, gradY = get_gradient_info(mesh, edges, vert_edge_outgoing, rhs_mat_list, row_inds, col_inds)

            gradX_list.append(gradX[np.newaxis])
            gradY_list.append(gradY[np.newaxis])
        gradX_list = torch.from_numpy(np.concatenate(gradX_list, axis=0)).float()
        gradY_list = torch.from_numpy(np.concatenate(gradY_list, axis=0)).float()
        torch.save({'X': gradX_list, 'Y': gradY_list, 'ind': indices, 'shape': (N, N)}, os.path.join(dataset_path, datatype, os.path.basename(f).replace('_processed_matrix.pkl', '_grad.pt')))

    print(counter)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    # prefix = '/media/qindafei/SSD/face/data'
    # prefix = '/home/qindafei/G/face/'
    # datasets = ['ICT_single_coarse', 'ICT_random_50', 'ICT_live_100', 'MF_sen', 'ICT_live2_cont', 'MF_holes', 'ICT_holes', 'ICT_shift_scale', 'MF_shift_scale', 'MF_all_v5']
    # datasets = ['ICT_live']
    # datasets = ['ICT_live2_cont']
    # datasets = [ 'MF_holes']
    # datasets = ['ICT_single_coarse']
    # datasets = [ 'COMA_orig_double_precision']
    # datasets = ['COMA_no_inner', 'Adobe', 'MetaHuman']
    # datasets = ['MetaHuman_cut']
    # datasets = ['MF_sen', 'ICT_live2_cont', 'MF_sen2']
    # datasets = ['ICT_live_100_disk']
    datasets = args.dataset
    # datapaths = [os.path.join(args.prefix, _) for _ in datasets]

    render_dataset(datasets)

