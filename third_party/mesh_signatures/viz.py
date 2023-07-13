import argparse

import glfw

import imgui
from trimesh.util import euclidean

from pygl.context import WindowContext
from pygl.buffers import *
import pygl.shader as ShaderManager
from pygl.camera import Camera
from pygl.mesh import Mesh
from pygl import transform

from matplotlib import cm

import signature
import laplace

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('file', help='File to load')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

obj = Mesh.load(args.file, rescale=False)

window = WindowContext((640, 480), "Mesh signature visualization")

method = args.laplace

r = np.linalg.norm(obj.mesh.vertices, axis=-1).max()
r *= 3

light_data = np.array([
    [[-r,  r, r, 1.0], [50.0 * r*r, 50.0 * r*r, 50.0 *r*r, 1.0]],
], dtype=np.float32)

# Settings
methods = laplace.approx_methods()
signature_types = ['heat', 'wave']
current_vertex = 0
current_method_id = methods.index(args.approx)
current_signature_id = signature_types.index(args.kernel)
cutoff = 1.0

window.set_active()
        
# Initialize signature extractor
if args.laplace is not None:
    extractor = signature.SignatureExtractor(path=args.laplace)
else:
    extractor = signature.SignatureExtractor(obj.mesh, args.n_basis, args.approx)

def update_viz(vid):
    cmap = cm.get_cmap("viridis")
    ds = extractor.feature_distance(
        vid, 
        args.f_size, 
        signature_types[current_signature_id],
        cutoff=cutoff)
    obj.vertex_colors = cmap(ds / ds.max())[..., :3]
 

update_viz(current_vertex)

# Visualization settings
ssbo = create_ssbo(light_data)
cam = Camera(window.size, far=max(100.0, 2 * r))
cam.look_at((0, 0, 0), (0, r*0.707, -r*0.707))
shader = ShaderManager.Shader("./shaders/pbr.vs", "./shaders/pbr.fs")

r = np.max(np.linalg.norm(obj.vertices, axis=-1))
marker = Mesh.uv_sphere(0.02*r)


while window.start_frame():
    imgui.begin("Settings", True)
    changed, current_vertex = imgui.drag_int("Vertex", current_vertex, min_value=0, max_value=obj.n_vertices - 1)
    if changed:
        update_viz(current_vertex)
    imgui.separator()
    _, args.n_basis = imgui.input_int("Basis", args.n_basis)
    _, current_method_id = imgui.combo("Laplace Approximation", current_method_id, methods)
    if imgui.button("Recompute Laplace"):
        extractor.initialize(obj.mesh, args.n_basis, methods[current_signature_id])
        update_viz(current_vertex)
    imgui.separator()
    _, current_signature_id = imgui.combo("Signature Type", current_signature_id, signature_types)
    _, args.f_size = imgui.input_int("Feature size", args.f_size)
    _, cutoff = imgui.drag_float("Cutoff", cutoff, 0.01, 0.0, 1.0)
    if imgui.button("Compute Signature"):
        update_viz(current_vertex)
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

    model = transform.rotate_y(glfw.get_time() * 40)

    obj.render(shader, model=model)

    marker.render(shader, 
        model=model @ transform.translate(obj.vertex(current_vertex)),
        tint=[1.0, 0, 1.0])
    
    window.end_frame()
