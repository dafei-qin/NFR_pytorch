import numpy as np
import face_model_io
import os
from tqdm import tqdm
from copy import deepcopy
import vedo
from vedo import Plotter
import trimesh
from glob import glob


exp_names = ['browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimple_L', 'mouthDimple_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthFunnel', 'mouthLeft', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L', 'mouthPress_R', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmile_L', 'mouthSmile_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'noseSneer_L', 'noseSneer_R']
def plot(points, iden_idx, exp_idx, exp_vec):
    vedo_mesh.points(points)
    vedo_text.text(''.join([f'{name} : {value:.2f}\n' for name, value in zip(exp_names[:27],exp_vec[:27])]))
    vedo_text2.text(''.join([f'{name} : {value:.2f}\n' for name, value in zip(exp_names[27:],exp_vec[27:])]))
    vedo_idx.text(f'iden: {iden_idx}, exp: {exp_idx}')

def slider_intensity(widget, event):
    global exp_weights, vedo_text, exp_idx
    value = widget.GetRepresentation().GetValue()


    face_model.set_expression(deepcopy(exp_weights * value))
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]


    points = vertices
    plot(points, -1, exp_idx, exp_weights * value)

def slider_idx(widget, event):
    global idx, exp_weights, vedo_text
    value = widget.GetRepresentation().GetValue()

    exp_weights[idx] = value
    face_model.set_expression(deepcopy(exp_weights))
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    points = vertices
    plot(points, -1, idx, exp_weights)

def button_func():
    global iden_idx, exp_idx, exp_weights

    exp_idx = int(input('Please input the expression index:\n'))
    exp_weights = expression_vecs[int(exp_idx)]
    
    points = data_list[iden_idx][exp_idx]

    plot(points, iden_idx, exp_idx, exp_weights)
    
def button_func_next():
    global iden_idx, exp_idx, exp_weights

    exp_idx = (exp_idx + 1) % len(expression_vecs)
    exp_weights = expression_vecs[int(exp_idx)]
    
    points = data_list[iden_idx][exp_idx]
    plot(points, iden_idx, exp_idx, exp_weights)

    # bu.switch() 
def button_iden_idx():
    global iden_idx, exp_idx, iden_weights

    iden_idx = int(input('Please input the identity index:\n'))
    # exp_weights = expression_vecs[int(exp_idx)]
    iden_weights = iden_vecs[iden_idx]
    face_model.set_identity(iden_weights)

    points = data_list[iden_idx][exp_idx]
    plot(points, iden_idx, exp_idx, exp_weights)

def button_idx():
    global idx
    idx = int(input('Please input index for change: '))


if __name__ == '__main__':

    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    save_dir = "D:\\ICT_data_single_random\\"
    data_list = []
    neutral_list = []
    data_files = sorted(glob(os.path.join(save_dir, '*.npy')))
    for iden_idx in range(len(data_files) // 2):
        data_list.append(np.load(os.path.join(save_dir, f'{iden_idx:03d}.npy')))
        neutral_list.append(np.load(os.path.join(save_dir, f'{iden_idx:03d}_neutral.npy')))

    expression_vecs = np.load(os.path.join(save_dir, 'iden_exp_weights_single.npz'))['exp']
    iden_vecs = np.load(os.path.join(save_dir, 'iden_exp_weights_single.npz'))['iden']
    exp_idx = 0
    iden_idx = 3
    idx = 0
    exp_weights = expression_vecs[exp_idx]
    iden_weights = iden_vecs[iden_idx]

    # exp_weights[51] = 0.5
    # exp_weights[18] = 0.47
    # exp_weights[26] = 0.42
    # exp_weights[27] = 0.30
    # exp_weights[26] = 0.275
    vedo_text = vedo.Text2D(''.join([f'{_:.2f}, ' for _ in exp_weights]), pos=[0, 1])
    vedo_text2 = vedo.Text2D(''.join([f'{_:.2f}, ' for _ in exp_weights]), pos=[0.7, 1])
    vedo_idx = vedo.Text2D(f'iden: --, exp: --', pos=[0.1, 0.05])
    mesh_ref = trimesh.load('../sample_data_out/random_expression00_onlyface.obj', process=False, maintain_order=True)
    faces = mesh_ref.faces

    face_model.set_expression(exp_weights)
    face_model.set_identity(iden_weights)
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    vedo_mesh = vedo.Mesh([np.array(vertices), np.array(faces.reshape(-1, 3))])

    plt = Plotter(N=1)
    plt += vedo_mesh
    plt += vedo_text
    plt += vedo_text2
    plt += vedo_idx
    plt.at(0).addSlider2D(
            slider_intensity,
            xmin=-2.00,
            xmax=2.00,
            value=0.00,
            pos=[(0.02, 0.05 + 0.05), (0.15, 0.05 + 0.05)],
            # title="color number",
        )
    plt.at(0).addSlider2D(
            slider_idx,
            xmin=-2.00,
            xmax=2.00,
            value=0.00,
            pos=[(0.02, 0.15), (0.15, 0.15)],
            # title="color number",
        )  
    plt.at(0).addButton(
        button_func,
        pos=(0.4, 0.1),  # x,y fraction from bottom left corner
        states=['new_exp'],
        c=["g"],
        bc=['w'],  # colors of states
        font="courier",   # arial, courier, times
        size=25,
        bold=True,
        italic=False,
    )
    plt.at(0).addButton(
        button_idx,
        pos=(0.6, 0.1),  # x,y fraction from bottom left corner
        states=['new_idx'],
        c=["g"],
        bc=['w'],  # colors of states
        font="courier",   # arial, courier, times
        size=25,
        bold=True,
        italic=False,
    )
    plt.at(0).addButton(
        button_iden_idx,
        pos=(0.8, 0.1),  # x,y fraction from bottom left corner
        states=['new_iden'],
        c=["g"],
        bc=['w'],  # colors of states
        font="courier",   # arial, courier, times
        size=25,
        bold=True,
        italic=False,
    )
    plt.at(0).addButton(
        button_func_next,
        pos=(0.8, 0.05),  # x,y fraction from bottom left corner
        states=['next_idx'],
        c=["g"],
        bc=['w'],  # colors of states
        font="courier",   # arial, courier, times
        size=25,
        bold=True,
        italic=False,
    )
    plt.show().close()