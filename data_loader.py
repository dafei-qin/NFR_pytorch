import platform
from datasets import my_dataset, test_dataset
# import datasets_disk
import sys
import time
sys.path.append('./third_party/diffusion-net/src')
import diffusion_net
from deformation_transfer import Transfer, Mesh

import torch
import numpy as np

from tqdm import tqdm

import os
import pickle
from copy import deepcopy
import logging
from glob import glob


def load_data(neutrals_dir, datasets, batch_size, normalizer, feature_type='cents&norms', use_wks=False, wks_normalizer=None, global_encoder='pn', device='cuda:0', cache_dir='/media/qindafei/SSD/face/cache',zero_mean=False, use_f=True, use_source_v=False, only_iden=False, exp_per_iden=1, use_landmarks=False, landmark_normalizer=None, use_img=False, img_normalizer=None, use_pix2face=False, dfn_info_per_loader=True, random_dfn_info=False, grad=False):

    transfs = []
    neutrals = []
    datasets_torch = {'train': [], 'val': [], 'test': []}

    
    if use_f:
        logging.info('Constructing linear solver...')
        t = time.time()
        for neu in neutrals_dir:
            neutral_mesh = Mesh.load(neu)
            neutral_mesh.vertices = neutral_mesh.vertices.astype(np.float32)
            face = neutral_mesh.faces
            transfs.append(Transfer(neutral_mesh, deepcopy(neutral_mesh), device=device))
            neutrals.append(deepcopy(neutral_mesh))
        logging.debug(f'Linear Solver constructed, time: {time.time() - t:.2f}s')

    total_num_data = {'train':0, 'val': 0, 'test':0}

    logging.info('Loading data...')
    logging.info(f'Using {feature_type} as inputs to the feature extractor')
    
    for dataset_type in datasets.keys(): # train, val, test
        for datafile_name in tqdm(datasets[dataset_type]): # identity files
            if platform.system() == 'Windows':
                neutral_dir = os.path.join(*datafile_name.split('\\')[:-2], os.path.basename(datafile_name) + '_neutral.obj')
                if datafile_name.split('\\')[0] == '':
                    neutral_dir = '\\' + neutral_dir
            else:
                neutral_dir = os.path.join(*datafile_name.split('/')[:-2], os.path.basename(datafile_name) + '_neutral.obj')
                if datafile_name.split('/')[0] == '':
                    neutral_dir = '/' + neutral_dir
            try:
                neutral_mesh = Mesh.load(neutral_dir)
            except FileNotFoundError:
                neutral_dir = neutral_dir.replace('_neutral.obj', '.obj')
                neutral_mesh = Mesh.load(neutral_dir)

            with open(datafile_name + '_processed_matrix.pkl', 'rb') as f:
                data = pickle.load(f)
                face = data['face']
                verts = data['verts']
                jacobians = data['gradients']

                j = torch.from_numpy(jacobians).float()
                j = j.reshape(j.shape[0], j.shape[1], -1)

            wks, neutral_wks, landmarks, neutral_landmarks, img, neutral_img, indices, values, zbuf, neutral_pix2face, neutral_zbuf = None, None, None, None, None, None, None, None, None, None, None
            dfn_info, dfn_info_list = None, None
            gradX, gradY = None, None
            if use_wks:
                wks = np.load(datafile_name + '_wks.npy')
                neutral_wks = np.load(neutral_dir.replace('.obj', '_wks.npy'))

            if use_landmarks:
                landmarks = np.load(datafile_name  + '_landmarks.npy')
                neutral_landmarks = np.load(neutral_dir.replace('.obj', '_landmarks.npy'))

                
                
            if use_img:
                img = np.load(datafile_name + '_imgs.npy')
                neutral_img = np.load(neutral_dir.replace('.obj', '_img.npy'))
                indices, values, zbuf = torch.load(datafile_name + '_imgs.pt').values()
                neutral_pix2face, neutral_zbuf = torch.load(neutral_dir.replace('.obj', '_img.pt')).values()
                img = np.concatenate([img, zbuf.unsqueeze(-1).numpy()], axis=-1)
                neutral_img = np.concatenate([neutral_img, neutral_zbuf.unsqueeze(-1).numpy()], axis=-1)
                if not use_pix2face:
                    neutral_pix2face = None
                    indices, values, zbuf = None, None, None

            if dfn_info_per_loader:
                # Create dfn_info for each neutral identity
                verts_list = torch.from_numpy(neutral_mesh.vertices).unsqueeze(0)
                face_list = torch.from_numpy(neutral_mesh.faces).unsqueeze(0)
                frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)
                dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(neutral_mesh.faces)]

            if grad:
                gradX, gradY, _, _ = torch.load(datafile_name + '_grad.pt').values()
                gradX = gradX.float()
                gradY = gradY.float()

            # Individual dataset for each identity
            d = my_dataset(face, neutral_mesh.vertices, torch.from_numpy(verts).float(), j, normalizer, neutral_wks=neutral_wks, wks=wks, wks_normalizer=wks_normalizer, neutral_landmarks=neutral_landmarks, landmarks=landmarks, landmark_normalizer=landmark_normalizer, img=img, neutral_img=neutral_img, img_normalizer=img_normalizer, feature_type=feature_type, device=device,  use_f=use_f, use_source_v=use_source_v, only_iden=only_iden, exp_per_iden=exp_per_iden, neutral_pix2face=neutral_pix2face, pix2face_indices=indices, pix2face_values=values, dfn_info=dfn_info, dfn_info_list = dfn_info_list, gradX=gradX, gradY=gradY)

            datasets_torch[dataset_type].append(d)
            total_num_data[dataset_type] += len(d)

    verts_list = torch.from_numpy(neutral_mesh.vertices).unsqueeze(0)
    face_list = torch.from_numpy(neutral_mesh.faces).unsqueeze(0)
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    # A default dfn_info
    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(neutral_mesh.faces)]

    dataloaders = {'train':[], 'val': [], 'test':[]}

    for dataset_type in datasets_torch.keys():
        for d in datasets_torch[dataset_type]:
            if dataset_type == 'train':
                dataloaders[dataset_type].append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False))
            else:
                dataloaders[dataset_type].append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=0))
    
    logging.info('Loading data done. Total number of data:' + ''.join([f'{key}: {value:d} ' for key, value in total_num_data.items()]))


    return face, transfs, dataloaders['train'], dataloaders['val'], dataloaders['test'],  dfn_info








def load_data_ict_live(data_dir, data_head, batch_size, n_identity, normalizer, feature_type='cents&norms', only_test=False, global_encoder='pn',  use_wks=False, wks_normalizer=None, device='cuda:0', cache_dir='/media/qindafei/SSD/face/cache', use_chol=False, zero_mean=False, use_f=True, use_source_v=False, use_landmarks=False, landmark_normalizer=None, use_img=False, img_normalizer=None, use_pix2face=False, random_dfn_info=False, grad=False, dfn_info_per_loader=False):
    if os.path.exists(os.path.join(data_dir, data_head, f'expression_vecs_train.npy')):
        exp_vec_train = np.load(os.path.join(data_dir, data_head, f'expression_vecs_train.npy'))
        exp_vec_val = np.load(os.path.join(data_dir, data_head, f'expression_vecs_val.npy'))
    else:
        exp_vec_train = None
        exp_vec_val = None
    exp_vec_test = np.load(os.path.join(data_dir, data_head, f'expression_vecs_test.npy'))
    iden_vec = np.load(os.path.join(data_dir, data_head, f'iden_vecs.npy'))
    
    identities_train = sorted(glob(os.path.join(data_dir, data_head, 'train/*')))
    identities_train = [_[:-8] for _ in identities_train if 'imgs.pt' in _][:n_identity]

    identities_val = sorted(glob(os.path.join(data_dir, data_head, 'val/*')))
    identities_val = [_[:-8] for _ in identities_val if 'imgs.pt' in _]

    identities_test = sorted(glob(os.path.join(data_dir, data_head, 'test/*')))
    identities_test = [_[:-8] for _ in identities_test if 'imgs.pt' in _]
    # identities = [_[:3] for _ in identities if _.endswith('obj') and 'neu' in _]
    if only_test:
        identities = {'val': identities_val, 'test': identities_test}
    else:
        identities = {'train': identities_train, 'val': identities_val, 'test': identities_test}

    if only_test is True:
        datasets = {
            'val': identities_val,
            'test': identities_test
        }
    else:
        datasets = {
            'train': identities_train,
            'val': identities_val,
            'test': identities_test
        }

    identities_idx = []
    for name, v in identities.items():
        if only_test is True:
            if name == 'train':
                continue
        for _ in v:
            identities_idx.append(int(os.path.basename(_)))
    iden_vec = iden_vec[identities_idx]

    neutral_dirs = [os.path.join(data_dir, data_head, os.path.basename(iden) + '_neutral.obj') for iden_list in identities.values() for iden in iden_list]
    for d in neutral_dirs:
        # print(d)
        assert os.path.exists(d)
    
    return iden_vec, exp_vec_train, exp_vec_val, exp_vec_test, *load_data(neutral_dirs, datasets, batch_size=batch_size, normalizer=normalizer, use_wks=use_wks, wks_normalizer=wks_normalizer, feature_type=feature_type, global_encoder=global_encoder, device=device, cache_dir=cache_dir, use_f=use_f, use_source_v=use_source_v, use_landmarks=use_landmarks, landmark_normalizer=landmark_normalizer, use_img=use_img, img_normalizer=img_normalizer, use_pix2face=use_pix2face, random_dfn_info=random_dfn_info, grad=grad, dfn_info_per_loader=dfn_info_per_loader)



def load_data_mf_v3(data_dir, data_head, batch_size, n_identity, normalizer, use_wks=False, wks_normalizer=None, feature_type='cents&norms', only_test=False, global_encoder='pn', positional_encoding=False, number_of_pe=10, use_hks=False, device='cuda:0', cache_dir='/media/qindafei/SSD/face/cache', use_chol=False, zero_mean=False, use_f=True, use_source_v=False, use_landmarks=False, landmark_normalizer=None, use_img=False, img_normalizer=None, use_pix2face=False, shuffle=False, random_dfn_info=False, grad=False, dfn_info_per_loader=False):

    identities_train = sorted(glob(os.path.join(data_dir, data_head, 'train/*')))
    if shuffle:
        import random
        random.shuffle(identities_train)

    identities_train = [_[:-8] for _ in identities_train if 'imgs.pt' in _][:n_identity]

    identities_val = sorted(glob(os.path.join(data_dir, data_head, 'val/*')))
    identities_val = [_[:-8] for _ in identities_val if 'imgs.pt' in _][:n_identity]
    if len(identities_val) > 12:
        identities_val = identities_val[:10]
    identities_test = sorted(glob(os.path.join(data_dir, data_head, 'test/*')))
    identities_test = [_[:-8] for _ in identities_test if 'imgs.pt' in _][:n_identity]
    if len(identities_test) > 12:
        identities_test = identities_test[:10]
    # identities = [_[:3] for _ in identities if _.endswith('obj') and 'neu' in _]
    if only_test:
        # identities = {'val': identities_val, 'test': identities_test}
        identities = { 'test': identities_test}
    else:
        identities = {'train': identities_train, 'val': identities_val, 'test': identities_test}

    # iden_vec = iden_vec[identities_idx]
    if only_test is True:
        # datasets = {
        #     'val': identities_val,
        #     'test': identities_test
        # }
        datasets = {
            
            'test': identities_test
        }
    else:
        datasets = {
            'train': identities_train,
            'val': identities_val,
            'test': identities_test
        }


    neutral_dirs = [os.path.join(data_dir, data_head, os.path.basename(iden) + '_neutral.obj') for iden_list in identities.values() for iden in iden_list]
    for d in neutral_dirs:
        # print(d)
        assert os.path.exists(d)

        

    return load_data(neutral_dirs, datasets, batch_size=batch_size, normalizer=normalizer, use_wks=use_wks, wks_normalizer=wks_normalizer, feature_type=feature_type, global_encoder=global_encoder, device=device, cache_dir=cache_dir, use_f=use_f, use_source_v=use_source_v, use_landmarks=use_landmarks, landmark_normalizer=landmark_normalizer, use_img=use_img, img_normalizer=img_normalizer, use_pix2face=use_pix2face,  random_dfn_info=random_dfn_info, grad=grad, dfn_info_per_loader=dfn_info_per_loader)

