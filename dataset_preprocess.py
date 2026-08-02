
import numpy as np
import subprocess
from copy import deepcopy
from glob import glob
from deformation_transfer import Transfer
import os
from tqdm import tqdm
import argparse
import warnings
import pickle
import multiprocessing
import trimesh
import torch

from myutils.myutils import Mesh, get_dfn_info_except_grad
import warnings
warnings.filterwarnings('ignore')



def get_centriod(fv):
    return fv.mean(axis=-2)
def get_normal(fv):
    span = fv[:, :, 1:, :] - fv[:, :, :1, :]
    norm = np.cross(span[:, :, 0, :], span[:, :, 1, :])
    return norm / np.linalg.norm(norm, axis=-1)[:, :, np.newaxis]

def Loss(y, yhat):
    return np.mean((y - yhat)**2)




def process_all_mesh(identities, save_dir, face, scale, split=False):


    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    
    for iden_idx, iden in enumerate(tqdm(identities)):
        v = np.load(iden)
        v *= scale
        if split:
            v_train = v[:len(v)//10 * 9]
            v_val = v[len(v)//10*9: len(v)//10 * 9 + len(v)//20]
            v_test = v[len(v)//10 * 9 + len(v)//20:]
        else:
            v_train = []
            v_val = []
            v_test =v



        neutral = Mesh(Mesh.load(os.path.join(os.path.dirname(iden), os.path.basename(iden)[:-4] + '_neutral.obj')).vertices, face)
        # neutral.vertices = neutral.vertices
        neutral.vertices = neutral.vertices * scale
        neutral.write(os.path.join(save_dir, os.path.basename(iden)[:-4] + '_neutral.obj'))
        transfer = Transfer(neutral, deepcopy(neutral))


        if split:
            v_ = {'train': v_train, 'val': v_val, 'test': v_test}
        else:
            v_ = {'test': v_test}

        for key, value in v_.items():
            v_new = []
            jacobians = []
            for _, data in enumerate(value):
                v_new.append(data[np.newaxis])
                current_mesh = Mesh(data, neutral.faces)

                j = transfer.deformation_gradient(current_mesh)
                out = transfer(current_mesh)

                try:
                    assert np.max(np.abs((out.vertices - out.vertices.mean(axis=0)) - (data - data.mean(axis=0)))) < 1e-8
                except AssertionError:
                    print(f'Reconstruction error {np.max(np.abs(out.vertices - out.vertices.mean(axis=0) - (data - data.mean(axis=0))) )} is big, please check the meshes in {os.path.join(save_dir, iden)}')
                    out.write(os.path.join(save_dir, f'{key}_{_}_{np.max(np.abs(out.vertices - out.vertices.mean(axis=0) - data)):.4f}.obj'))
                    out.vertices = data
                    out.write(os.path.join(save_dir, f'{key}_{_}.gt.obj'))

                jacobians.append(j[np.newaxis])

            jacobians = np.concatenate(jacobians, axis=0)
            v_new = np.concatenate(v_new, axis=0)
            with open(os.path.join(save_dir, key, os.path.basename(iden)[:-4] + f'_processed_matrix.pkl'), 'wb') as f:
                pickle.dump({'face':face, 'verts':v_new.astype(np.float32), 'gradients':jacobians.reshape(jacobians.shape[0], jacobians.shape[1], 9).astype(np.float32)}, f)


def process_sequences(identities, save_dir, face, scale, split=False):
    identities = [_.replace('.npy', '') for _ in identities]
    for iden_idx, iden in enumerate(tqdm(identities)):
        neutral = Mesh(Mesh.load(iden + '_neutral.obj').vertices, face)
        neutral.vertices = neutral.vertices * scale
        neutral.write(os.path.join(save_dir, os.path.basename(iden) + '_neutral.obj'))
        transfer = Transfer(neutral, deepcopy(neutral))

        datas = glob(iden + '*.npy')
        if split:
            v_ = {'train': datas[:-4], 'val': datas[-4:-2], 'test': datas[-2:]}
        else:
            v_ = {'test': datas}


        for key, value in v_.items():
            for file in value:
                v_new = np.load(file)
                jacobians = []
                for _, data in enumerate(v_new):
                    # v_new.append(data[np.newaxis])
                    current_mesh = Mesh(data, neutral.faces)

                    j = transfer.deformation_gradient(current_mesh)
                    out = transfer(current_mesh)

                    try:
                        assert np.max(np.abs((out.vertices - out.vertices.mean(axis=0)) - (data - data.mean(axis=0)))) < 1e-8
                    except AssertionError:
                        print(f'Reconstruction error {np.max(np.abs(out.vertices - out.vertices.mean(axis=0) - (data - data.mean(axis=0))) )} is big, please check the meshes in {os.path.join(save_dir, iden)}')
                        out.write(os.path.join(save_dir, f'{key}_{_}_{np.max(np.abs(out.vertices - out.vertices.mean(axis=0) - data)):.4f}.obj'))
                        out.vertices = data
                        out.write(os.path.join(save_dir, f'{key}_{_}.gt.obj'))

                    jacobians.append(j[np.newaxis])

                jacobians = np.concatenate(jacobians, axis=0)
                with open(os.path.join(save_dir, key, file.split('/')[-1].replace('.npy', '') + f'_processed_matrix.pkl'), 'wb') as f:
                    pickle.dump({'face':face, 'verts':v_new.astype(np.float32), 'gradients':jacobians.reshape(jacobians.shape[0], jacobians.shape[1], 9).astype(np.float32)}, f)
def calc_gradients_std(identities, save_dir):
    gradients_list = []
    for f in glob(os.path.join(save_dir, 'train/*_matrix.pkl')):
        with open(f, 'rb') as f:
            face, verts, gradients = pickle.load(f).values()
        gradients_list.append(gradients)
    gradients = np.concatenate(gradients_list, axis=0)
    gradients_std = gradients.reshape(-1, 9).astype(np.float64).std(axis=0) # IMPORTANT: Convert to float64 to compute the std for accuracy

    np.save(os.path.join(save_dir, 'gradients_std.npy'), gradients_std)



def visualize(save_dir, n_identity):
    from vedo import trimesh2vedo, screenshot, Plotter, settings
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    settings.screenshot_transparent_background = False
    settings.screeshot_scale = 4
    screeshot_large_image = True


    visualize_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    train_iden = glob(os.path.join(save_dir, 'train', '*.npy'))
    train_iden = [_.split('/')[-1][:-8] for _ in train_iden][:n_identity]
    val_iden = glob(os.path.join(save_dir, 'val', '*.npy'))
    val_iden = [_.split('/')[-1][:-8] for _ in val_iden][:n_identity]

    test_iden = glob(os.path.join(save_dir, 'test', '*.npy'))
    test_iden = [_.split('/')[-1][:-8] for _ in test_iden][:n_identity]

    for datatype, iden_list in zip(['train', 'val', 'test'], [train_iden, val_iden, test_iden]):

        for iden in iden_list:
            if not os.path.exists(os.path.join(visualize_dir, datatype, iden)):
                os.makedirs(os.path.join(visualize_dir, datatype, iden))
            pkl_f = os.path.join(save_dir, datatype, iden + '_processed_matrix.pkl')
            npy_f = os.path.join(save_dir, datatype, iden + '_wks.npy')
            with open(pkl_f, 'rb') as f:
                face, verts, j = pickle.load(f).values()
            wks = np.load(npy_f)

            for idx, v in enumerate(tqdm(verts)):

                vp = Plotter(axes=0, offscreen=True)
                current_mesh = trimesh.Trimesh(verts[idx], face, process=False, maintain_order=True)
                mesh_list = []
                mesh_list.append(current_mesh)
                for i in range(1, 11):
                    mesh_list.append(trimesh.Trimesh(verts[idx] + np.array([(i % 3) *(np.max(verts[idx], axis=0) - np.min(verts[idx], axis=0))[0], (i // 3) *(np.max(verts[idx], axis=0) - np.min(verts[idx], axis=0))[1], 0]), face, process=False, maintain_order=True))
                meshes = [trimesh.util.concatenate(mesh_list)]
                vmeshes = trimesh2vedo(meshes)
                cmap = 'jet'
                scals_diff = ((verts[idx] - verts[0])**2).sum(axis=-1)[:, np.newaxis]
                scals_diff /= (scals_diff.max() + 1e-3)
                scals_diff /= 0.2
                scals_wks = wks[idx]
                scals = np.concatenate([scals_diff, scals_wks], axis=-1)
                vmeshes[0].cmap(cmap, scals.transpose(1, 0).reshape(-1)).lighting('plastic')
                # vmeshes[0].add_scalarbar(title=f"scalarbar #{0}", c='k') 
                vp.show(vmeshes, axes=1)
                screenshot(os.path.join(visualize_dir, datatype, iden, f'{idx:04d}.png'))
                # break
                # print(idx)



if __name__ == '__main__':
    # Pre-requsits
    os.environ["OMP_NUM_THREADS"] = "1" # Prevent numpy from using too much cores.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_data_dir', type=str, required=True)
    parser.add_argument('--n_identity', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--split', default=False, action='store_true')
    parser.add_argument('--process_matrix', '-pm', default=False, action='store_true')
    parser.add_argument('--process_img', '-pi', default=False, action='store_true')
    parser.add_argument('--process_grad', '-pg', default=False, action='store_true')
    parser.add_argument('--process_dfn', '-pd', default=False, action='store_true')
    parser.add_argument('--sequences', default=False, action='store_true')
    args = parser.parse_args()

    if args.split:
        print('Spliting the dataset into 90%, 5%, 5% for training, validating and testing, for each identity')
        os.makedirs(os.path.join(args.save_data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.save_data_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(args.save_data_dir, 'test'), exist_ok=True)
    else:
        print('All data go to the testing directory.')
        os.makedirs(os.path.join(args.save_data_dir, 'test'))


    data_dir = args.data_dir
    save_data_dir = args.save_data_dir

    neutral_objs = sorted(glob(os.path.join(args.data_dir, '*.obj')))

    neutral_mapping_face = neutral_objs[0]

    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    scale = args.scale
    identities = sorted(os.listdir(data_dir))
    identities = [os.path.join(data_dir, _) for _ in identities if 'obj' in _  and 'neutral' in _]
    identities = [_.replace('_neutral.obj', '.npy') for _ in identities]
    if args.n_identity > 0:
        identities = identities[:args.n_identity]
        
    print(identities)

    # Get face mapping and vertex mapping
    neutral_mesh = Mesh.load(neutral_mapping_face)

    face = neutral_mesh.faces

    # Process the expression codes and identity codes
    
    if args.process_matrix:
        print('Calculating deformations...')
        if args.sequences:
            process_sequences(identities, save_data_dir, face, scale, split=args.split)
        else:
            process_all_mesh(identities, save_data_dir, face, scale, split=args.split)
        
    if args.process_img:
        print('Rendering images')
        subprocess.call(['python', 'render_dataset.py', '--dataset', save_data_dir])
    if args.process_grad:
        print('Generating DiffusionNet grads')
        subprocess.call(['python', 'generate_grad.py', '--dataset', save_data_dir])
    # subprocess.call(['python', 'analy_dfn_info.py', '--dataset', save_data_dir])
    if args.process_dfn:
        print(f'Using {args.num_threads} to calculate decompositions (this may takes quite a long time)...')
        # Generate the dfn info for each mesh
        files = sorted(glob(os.path.join(save_data_dir, '*', '*.pkl')))
        for idx, f in enumerate(files):
            datatype = f.split('/')[-2]

            # Neutral
            try:
                neutral_obj = Mesh.load(os.path.join(save_data_dir, os.path.basename(f).replace('_processed_matrix.pkl', '_neutral.obj')))
            except:
                neutral_obj = Mesh.load(os.path.join(save_data_dir, os.path.basename(f).replace('_processed_matrix.pkl', '.obj')))

            neutral_obj.vertices = neutral_obj.vertices.astype(np.float32)
            

            # Data
            try:
                with open(f, 'rb') as ff:
                    face, verts, gradients= pickle.load(ff).values()
            except ValueError:
                with open(f, 'rb') as ff:
                    face, verts, gradients, _= pickle.load(ff).values()

            verts= verts.astype(np.float32)
            result_mass = []
            result_L_idx = []
            result_L_val = []
            result_evals = []
            result_evecs = []
            
            
            pool = multiprocessing.Pool(args.num_threads)
            pbar = tqdm(total=len(verts))

            def update(rst):
                global result_L_idx
                result_mass.append(rst[0].unsqueeze(0))
                result_L_idx = rst[1]
                result_L_val.append(rst[2].unsqueeze(0))
                result_evals.append(rst[4].unsqueeze(0))
                result_evecs.append(rst[5].unsqueeze(0))
                
                pbar.update(1)
            for i in range(len(verts)):
                
                pool.apply_async(get_dfn_info_except_grad, (verts[i],face,), {'k_eig':64,}, callback=update)
                
            pool.close()
            pool.join()
            pbar.close()
            result_mass = torch.cat(result_mass, dim=0)
            result_L_val = torch.cat(result_L_val, dim=0)
            result_evals = torch.cat(result_evals, dim=0)
            result_evecs = torch.cat(result_evecs, dim=0)
            obj = {'mass': result_mass, 'L_idx': result_L_idx, 'L_val': result_L_val, 'evals': result_evals, 'evecs': result_evecs}
            torch.save(obj, f.replace('_processed_matrix.pkl', '_dfn_info.pt'))
        
        
    
