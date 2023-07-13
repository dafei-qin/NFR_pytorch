import numpy as np
import face_model_io
import os
from tqdm import tqdm
from copy import deepcopy
import vedo
from vedo import Plotter
import trimesh



def slider_intensity(widget, event):
    global exp_weights, vedo_text
    value = widget.GetRepresentation().GetValue()

    # exp_weights[idx] = value
    face_model.set_expression(deepcopy(exp_weights * value))
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    points = vertices
    vedo_mesh.points(points)
    vedo_text.text(''.join([f'{_:.2f}, ' for _ in exp_weights]))    


def slider_idx(widget, event):
    global idx, exp_weights, vedo_text
    value = widget.GetRepresentation().GetValue()

    exp_weights[idx] = value
    face_model.set_expression(deepcopy(exp_weights * value))
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    points = vertices
    vedo_mesh.points(points)
    vedo_text.text(''.join([f'{_:.2f}, ' for _ in exp_weights]))    

def button_func():
    global exp_weights

    exp_weights = input('Please input expression idx to change:\n')
    exp_weights = np.fromstring(exp_weights, sep=',')
    face_model.set_expression(exp_weights)
    face_model.deform_mesh()
    points = face_model._deformed_vertices[:11248]
    vedo_mesh.points(points)
    vedo_text.text(''.join([f'{_:.2f}, ' for _ in exp_weights]))  

    
    # bu.switch() 

def button_idx():
    global idx
    idx = int(input('Please input index for change: '))


if __name__ == '__main__':

    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    exp_weights = np.zeros((53,))


    # exp_weights[51] = 0.5
    # exp_weights[18] = 0.47
    # exp_weights[26] = 0.42
    # exp_weights[27] = 0.30
    # exp_weights[26] = 0.275
    vedo_text = vedo.Text2D(''.join([f'{_:.2f}, ' for _ in exp_weights]), pos='bottom-left')
    mesh_ref = trimesh.load('../template_only_face.obj', process=False, maintain_order=True)
    faces = mesh_ref.faces
    face_model.set_expression(exp_weights)
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    vedo_mesh = vedo.Mesh([np.array(vertices), np.array(faces.reshape(-1, 3))])

    plt = Plotter(N=1)
    plt += vedo_mesh
    plt += vedo_text
    
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
    plt.show().close()