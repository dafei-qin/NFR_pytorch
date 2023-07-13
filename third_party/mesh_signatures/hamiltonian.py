import argparse
import logging

import glfw

import imgui
from imgui import extra
from imgui_datascience import imgui_fig

from pygl.context import WindowContext
from pygl.buffers import *
import pygl.shader as ShaderManager
from pygl.camera import Camera
from pygl.mesh import Mesh
from pygl import transform

from matplotlib import cm
import matplotlib.pyplot as plt
import scipy

import signature
import laplace

def compute_hamiltonian(extractor: signature.SignatureExtractor, 
                        v: np.ndarray):
    """Compute the spectrum of the hamiltonian operator
    See Hamiltonian Operator for Spectral Shape Analysis by Yoni Choukroun et al for detailed informations
    (http://arxiv.org/abs/1611.01990)

    Args:
        extractor (signature.SignatureExtractor): The signature extractor holding the mesh spectrum
        v (np.ndarray): A array containing a per vertex potential
    """
    W = extractor.W
    M = extractor.M
    sigma = -0.001
    thau  = extractor.evals[-1] * 5
    v = 0.5*thau*(np.tanh(v) + 1)
    try:
        from sksparse.cholmod import cholesky
        use_cholmod = True
    except ImportError:
        logging.warn(
            "Package scikit-sparse not found (Cholesky decomp). "
            "This leads to less efficient eigen decomposition.")
        use_cholmod = False

    A = W + M.dot(scipy.sparse.diags(v))
    if use_cholmod:
        chol = cholesky(A - sigma * M)
        op_inv = scipy.sparse.linalg.LinearOperator(
                        matvec=chol, 
                        shape=W.shape,
                        dtype=W.dtype)
    else:
        lu = scipy.sparse.linalg.splu(A - sigma * M)
        op_inv = scipy.sparse.linalg.LinearOperator(
                        matvec=lu.solve, 
                        shape=W.shape,
                        dtype=W.dtype)

    evals, evecs = scipy.sparse.linalg.eigsh(A, 
                                             extractor.n_basis, 
                                             M, 
                                             sigma=sigma,
                                             OPinv=op_inv)
    return evals, evecs

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('file', help='File to load')
parser.add_argument('--k', default='10', type=int, help='Number of eigenvalues and functions to compute')
parser.add_argument('--approx', default='fem', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')

args = parser.parse_args()

obj = Mesh.load(args.file, rescale=False)
# The potential should be encoded in the vertex colors 
# (Red -> maximum potential, white -> zero potential)
potential = np.where(obj.vertex_colors[:, 1] < 1, 1, 0)
print("potential shape", potential.shape)

window = WindowContext((640, 480), "Hamiltonian signature visualization")

method = args.laplace

r = np.linalg.norm(obj.mesh.vertices, axis=-1).max()
r *= 3

light_data = np.array([
    [[-r,  r, r, 1.0], [50.0 * r*r, 50.0 * r*r, 50.0 *r*r, 1.0]],
], dtype=np.float32)

window.set_active()

# Settings
rotate = True
show_potential = False
show_hameltoninan = True
active_eigen_value_index = 1
        
# Initialize signature extractor
if args.laplace is not None:
    extractor = signature.SignatureExtractor(path=args.laplace)
else:
    extractor = signature.SignatureExtractor(obj.mesh, args.k, args.approx)

evals, evecs = compute_hamiltonian(extractor, potential)

print(extractor.evecs.shape)

figure = plt.figure()
plt.xlabel("Eigen Index")
plt.ylabel("Eigen Value")
plt.plot(np.arange(args.k), extractor.evals)
plt.plot(np.arange(args.k), evals)

def update_viz():
    if show_potential:
        obj.vertex_colors = np.where(potential[:, None] == 1, [1, 0, 1], [1, 1, 1])
    else:
        used_evecs = evecs if show_hameltoninan else extractor.evecs
        cmap = cm.get_cmap('seismic')
        vals = used_evecs[:, active_eigen_value_index]
        range = np.abs(vals).max()
        vals = 0.5 * (vals / range + 1)
        obj.vertex_colors = cmap(vals)[:, :3]
    
    efunc_fig = plt.figure()
    plt.ylim((-1, 1))
    plt.xlabel("Vertex Index")
    plt.ylabel("Eigen Function")
    plt.plot(np.arange(obj.n_vertices), extractor.evecs[:, active_eigen_value_index])
    plt.plot(np.arange(obj.n_vertices), evecs[:, active_eigen_value_index])
    return efunc_fig

efunc_fig = update_viz()

# Visualization settings
ssbo = create_ssbo(light_data)
cam = Camera(window.size, far=max(100.0, 2 * r))
cam.look_at((0, 0, 0), (0, r*0.707, -r*0.707))
shader = ShaderManager.Shader("./shaders/pbr.vs", "./shaders/pbr.fs")

model = np.eye(4, dtype=np.float32)
last_time = glfw.get_time()

while window.start_frame():
    imgui.begin("Settings")
    rotate_changed, rotate = imgui.checkbox("Rotate", rotate)
    if rotate_changed:
        last_time = glfw.get_time()
    changed_0, show_potential = imgui.checkbox("Show Potential", show_potential)
    changed_1, show_hameltoninan = imgui.checkbox("Show Hameltonian", show_hameltoninan)
    changed_2, active_eigen_value_index = imgui.drag_int("Eigenvalue index", active_eigen_value_index, min_value=0, max_value=args.k-1)
    active_eigen_value_index = max(1, min(args.k - 1, active_eigen_value_index))
    if changed_0 or changed_1 or changed_2:
        efunc_fig = update_viz()
    imgui_fig.fig(efunc_fig, height=512, title="eigen function")
    imgui_fig.fig(figure, height=512, title="spectrum")
    imgui.end()

    shader.use(
        ao=1.0,
        metallic=0.0,
        tint=np.ones((3, ), dtype=np.float32),
        roughness=0.1,
        num_lights=len(light_data),
        camPos=cam.position[:3],
        projection=cam.P,
        view=cam.V)
    ssbo.bind(3)

    if rotate:
        now = glfw.get_time()
        delta_time = last_time - now
        last_time = now
        model = transform.rotate_y(delta_time * 30) @ model

    obj.render(shader, model=model)
    
    window.end_frame()
