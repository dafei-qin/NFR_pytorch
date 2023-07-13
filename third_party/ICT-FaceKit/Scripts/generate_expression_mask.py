

"""Example script that samples and writes random identities.
"""
import numpy as np
import face_model_io
import os
from tqdm import tqdm
from copy import deepcopy
import trimesh
import vedo
from vedo import Plotter

import colorcet

exp_names = ['browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimple_L', 'mouthDimple_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthFunnel', 'mouthLeft', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L', 'mouthPress_R', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmile_L', 'mouthSmile_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'noseSneer_L', 'noseSneer_R']

def button_func():
    global idx
    idx = (idx + 1) % masks.shape[0]
    vedo_mesh.cmap(cmap_linear, masks[idx], vmin=0, vmax=1)
    vedo_text.text(f'{exp_names[idx]}')

if __name__ =='__main__':
    """Loads the ICT Face Mode and samples and writes 10 random identities.
    """
    # Create a new FaceModel and load the model
    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    exp_bases = face_model._expression_shape_modes
    exp_bases = exp_bases[:, :11248]
    exp_bases = (exp_bases **2).sum(axis=-1)
    masks = exp_bases > 0.1
    masks = masks.astype(int)
    masks_sum = masks.sum(axis=-1)
    weight  = 1 / (masks_sum + 50) * max(masks_sum)
    np.save('./exp_weights.npy', weight)
    plt = Plotter(N=1)
    mesh_ref = trimesh.load('../sample_data_out/random_expression00_onlyface.obj', process=False, maintain_order=True)
    faces = mesh_ref.faces
    vedo_mesh = vedo.Mesh([np.array(mesh_ref.vertices), np.array(faces.reshape(-1, 3))])
    vedo_text = vedo.Text2D(f'--', pos=[0.1, 1])
    plt += vedo_mesh
    plt += vedo_text
    cmap_linear = colorcet.CET_L18
    idx = 0

    
    plt.at(0).addButton(
        button_func,
        pos=(0.4, 0.1),  # x,y fraction from bottom left corner
        states=['next_mask'],
        c=["g"],
        bc=['w'],  # colors of states
        font="courier",   # arial, courier, times
        size=25,
        bold=True,
        italic=False,
    )
    plt.interactive().close()

