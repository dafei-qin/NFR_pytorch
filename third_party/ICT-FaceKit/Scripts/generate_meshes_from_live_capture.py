import numpy as np
import face_model_io
import os
import trimesh
from copy import deepcopy
from scipy import signal
from glob import glob
from tqdm import tqdm
def downsampling(motion, factor):
    length = motion.shape[0]
    resampled_length = length * factor
    # assert resampled_length == int(resampled_length), print(resampled_length)
    resampled_motion = signal.resample(motion, int(resampled_length))

    return resampled_motion

def read_blendshapes(path, ratio = 6 / 24):
    with open(path, 'r') as f:
        string_list = f.readlines()
    blendshape_names = []
    blendshape_weights_tmp = []
    num_of_frames = -1
    blendshape_weights_all = {}
    line_idx = 0
    while line_idx < len(string_list):
        line = string_list[line_idx]
        while 'attribute' not in line:
            if 'value' in line:
                blendshape_weights_tmp.append(float(line.split(':')[-1]))
            line_idx += 1
            if line_idx == len(string_list):
                break
            line = string_list[line_idx]

            
        blendshape_name = line.split('.')[-1][:-1]
        if num_of_frames == -1:
            num_of_frames = len(blendshape_weights_tmp)

        blendshape_names.append(blendshape_name)
        blendshape_weights_all[blendshape_name] = deepcopy(np.array(blendshape_weights_tmp))
        blendshape_weights_tmp = []
        line_idx += 1

    blendshapes = blendshape_weights_all

    blendshapes = {key: blendshapes[key] for key in list(blendshapes.keys())[:-9]}
    blendshapes = [value[:, np.newaxis] for value in blendshapes.values()]
    blendshapes = np.concatenate(blendshapes, axis=1)
    blendshapes_vec = np.zeros((num_of_frames, 53))
    blendshapes_vec[:, :3] = blendshapes[:, :3]
    blendshapes_vec[:, 3] = blendshapes[:, 2]
    blendshapes_vec[:, 4:7] = blendshapes[:, 3:6]
    blendshapes_vec[:, 7] =  blendshapes[:, 5]
    blendshapes_vec[:, 8:] = blendshapes[:, 6:] 
    blendshapes_vec = downsampling(blendshapes_vec, ratio)
    blendshapes_vec = np.concatenate([np.zeros((1, 53)), blendshapes_vec], axis=0)
    return blendshapes_vec




if __name__ == '__main__':
    files = glob('../../../live_capture/*.anim')

    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    save_dir = "D:\\Live_Captures\\"
    # exp_vecs = read_blendshapes('../../../face/anim_1.txt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mesh_ref = trimesh.load('../sample_data_out/random_expression00_onlyface.obj', process=False, maintain_order=True)
    for idx, f in enumerate(files):
        print(f'{idx+1}/{len(files)}')
        current_save_dir = os.path.join(save_dir, os.path.basename(files[0]).split('.')[0])
        os.makedirs(current_save_dir, exist_ok=True)
        exp_vecs = read_blendshapes(f)
        for idx, v in enumerate(tqdm(exp_vecs)):
            face_model.set_expression(v)
            face_model.deform_mesh()
            vertices = face_model._deformed_vertices[:11248]
            mesh_ref.vertices = vertices
            mesh_ref.export(os.path.join(current_save_dir, f'{idx:04d}.obj'), 'obj')