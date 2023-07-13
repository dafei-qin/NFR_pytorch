import numpy as np
import face_model_io
import os
from tqdm import tqdm
from copy import deepcopy
import vedo
from vedo import Plotter
import trimesh


def make_slider():
        def slider(widget, event):
            global idx, exp_weights, vedo_text
            value = widget.GetRepresentation().GetValue()

            exp_weights[idx] = value
            face_model.set_expression(deepcopy(exp_weights))
            face_model.deform_mesh()
            vertices = face_model._deformed_vertices[:11248]

            points = vertices
            vedo_mesh.points(points)
            vedo_text.text(''.join([f'{_:.2f}, ' for _ in exp_weights]))    
        return slider

def button_func():
    global idx, exp_weights

    idx = int(input('Please input expression idx to change:\n'))
    
    # bu.switch() 


if __name__ == '__main__':

    face_model = face_model_io.load_face_model('../FaceXModel')
    id_coeffs, exp_coeffs = face_model_io.read_coefficients('../sample_data/sample_identity_coeffs.json')
    print(face_model._expression_names)
    exp_weights = np.zeros((53,))
    # exp_weights = [-0.109, -0.050, 0.432, -0.048, -0.010, -0.002, -0.231, -0.077, 0.037, 0.025, -0.046, -0.042, 0.037, -0.093, -0.089, -0.072, 0.108, 0.096, 0.233, 0.070, -0.050, -0.019, 0.072, -0.173, -0.139, -0.041, 0.160, 0.155, -0.117, -0.048, 0.018, -0.086, -0.088, -0.359, -0.072, -0.064, 0.340, -0.060, 0.020, -0.381, 0.030, -0.037, -0.009, -0.323, -0.033, 0.059, 0.071, -0.035, 0.028, 0.007, -0.157, 0.194, 0.109]

    # exp_weights[51] = 0.5
    # exp_weights[18] = 0.47
    # exp_weights[26] = 0.42
    # exp_weights[27] = 0.30
    # exp_weights[26] = 0.275
    vedo_text = vedo.Text2D(''.join([f'{_:.2f}, ' for _ in exp_weights]), pos='bottom-left')
    mesh_ref = trimesh.load('../sample_data_out/random_expression00_onlyface.obj', process=False, maintain_order=True)
    faces = mesh_ref.faces
    face_model.set_expression(exp_weights)
    face_model.deform_mesh()
    vertices = face_model._deformed_vertices[:11248]

    vedo_mesh = vedo.Mesh([np.array(vertices), np.array(faces.reshape(-1, 3))])

    plt = Plotter(N=1)
    plt += vedo_mesh
    plt += vedo_text
    
    plt.at(0).addSlider2D(
            make_slider(),
            xmin=-2.00,
            xmax=2.00,
            value=0.00,
            pos=[(0.02, 0.05 + 0.05), (0.15, 0.05 + 0.05)],
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

    plt.show().close()