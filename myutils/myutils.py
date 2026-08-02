import potpourri3d as pp3d
import trimesh
import open3d as o3d
from copy import deepcopy
import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
import torch
try:
    import pytorch3d
    from pytorch3d import transforms
    flag_pytorch3d = True
except ImportError:
    flag_pytorch3d = False
from torch_scatter import scatter_add
import transforms3d as t3d
import scipy
import scipy.sparse.linalg as sla
# import scipy.transform.Rotation as Rotation
import os
import warnings
import igl
# from mesh_signatures.signature import SignatureExtractor
import sys
import pickle
from tqdm import tqdm
sys.path.append('./third_party/diffusion-net/src')
sys.path.append('./third_party/')

import diffusion_net



class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.edges = None
        self.vertex_degree = None
        self.face_adjacency_edges = None
        self.face_adjacency_unshared = None
        # self.edges_unique_length = None
        self.edges_unique = None
        # self.area_faces = None

        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)



    def __getitem__(self, key):
        if key == 0:
            return self.vertices

        elif key == 1:
            return self.faces
        else:
            raise KeyError # print(f"Only allows Key 0: vertices, Key 1: faces, but receives Key {key} ")

    def update_area(self):
        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)


    @classmethod
    def load(self, path: str, read_face=True):
        """
        Load obj file
        load the .obj format mesh file with square or triangle faces
        return the vertices list and faces list
        """
        if path.endswith('.obj'):
            file = open(path, 'r')
            lines = file.readlines()
            vertices = []
            faces = []
            for line in lines:
                if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                    line_split = line.split(" ")
                    # ver = line_split[1:4]
                    ver = [each for each in line_split[1:] if each != '']
                    ver = [float(v) for v in ver]
                    vertices.append(ver)
                else:
                    if read_face:
                        if line.startswith('f'):
                            line_split = line.split(" ")
                            if '/' in line:
                                tmp_faces = line_split[1:]
                                f = []
                                if '\n' in tmp_faces:
                                    tmp_faces.pop(tmp_faces.index('\n'))
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face.split('/')[0]))
                                faces.append(f)
                            else:
                                tmp_faces = line_split[1:]
                                f = []
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face))
                                faces.append(f)
                    else:
                        pass

            if read_face:
                file.close()
                return Mesh(np.array(vertices), np.array(faces) - 1)
            else:
                file.close()
                return Mesh(np.array(vertices), None)
        # else:
        #     raise ValueError('Wrong file format, not a correct .obj mesh file!')
        #     ret

            
    @classmethod
    def from_trimesh(self, mesh: trimesh.Trimesh):
        
        new_mesh = Mesh(deepcopy(mesh.vertices), deepcopy(mesh.faces))

        new_mesh.edges = deepcopy(mesh.edges)
        new_mesh.vertex_degree = deepcopy(mesh.vertex_degree)
        new_mesh.face_adjacency_edges = deepcopy(mesh.face_adjacency_edges)
        new_mesh.face_adjacency_unshared = deepcopy(mesh.face_adjacency_unshared)
        # new_mesh.edges_unique_length = deepcopy(mesh.edges_unique_length)
        new_mesh.edges_unique = deepcopy(mesh.edges_unique)
        # new_mesh.area_faces = deepcopy(mesh.area_faces)
        
        return new_mesh


    
    def transfer(self, shift, scale):
        self.vertices = self.vertices * np.array(scale)[np.newaxis]
        self.vertices = self.vertices + np.array(shift)[np.newaxis]
        self.update_area()
        
    def write(self, file_name_path):
        faces = self.faces
        vertices = self.vertices
        faces = faces + 1
        with open(file_name_path, 'w') as f:
            for v in vertices:
                # print(v)
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for face in faces:
                if len(face) == 4:
                    f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])) 
                if len(face) == 3:
                    f.write("f {} {} {}\n".format(face[0], face[1], face[2])) 



def calc_norm(mesh):
    cross1 = lambda x,y:np.cross(x,y)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def calc_norm_torch(batch_v, face, at='face'):
    B_S = batch_v.shape[0]
    N_V = batch_v.shape[1]
    batch_v = batch_v.permute(1, 0, 2)
    vec1 = (batch_v[face[:, 1]] - batch_v[face[:, 0]]).permute(1, 0, 2)
    vec2 = (batch_v[face[:, 2]] - batch_v[face[:, 0]]).permute(1, 0, 2)

    face_norm = torch.nn.functional.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]
    if at == 'face':
        return face_norm
    else:
        idx = torch.cat([face[:, 0], face[:, 1], face[:, 2]], dim=0)
        face_norm = face_norm.repeat(1, 3, 1)

        norm = scatter_add(face_norm, idx, dim=1, dim_size=N_V)
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)  # [N, 3]
        return norm

def convert_per_face_to_per_vertex(inputs_f, face, n_vertex):

    # use a sparse matrix of which face contains each vertex to
    # figure out the summed normal at each vertex
    # allow cached sparse matrix to be passed

    indices = np.asanyarray(face)
    columns = n_vertex

    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape(
        (-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    data = np.ones(len(col), dtype=bool)

    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                    shape=shape,
                                    dtype=data.dtype)



    summed = torch.from_numpy(matrix.dot(inputs_f.transpose(0, 1).reshape(inputs_f.shape[1], -1)).reshape(matrix.shape[0], inputs_f.shape[0], inputs_f.shape[-1]).transpose(1, 0, 2))
    
    return summed
        
def face_to_vertex_torch(face):
    face = face.numpy()
    indices = face
    columns = np.max(face) + 1
    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape(
        (-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    data = np.ones(len(col), dtype=bool)

    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)))
    coo = matrix.multiply(1 / (matrix.sum(axis=1) + 1e-6))
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return matrix                            

def calc_cent(mesh):
    fv = mesh.vertices[mesh.faces]
    return fv.mean(axis=-2) 

def get_biggest_connected(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))

    triangle_clusters, cluster_n_triangles, cluster_area = (o3d_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    o3d_mesh_1 = deepcopy(o3d_mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    o3d_mesh_1.remove_triangles_by_mask(triangles_to_remove)

    return Mesh(np.asarray(o3d_mesh_1.vertices), np.asarray(o3d_mesh_1.triangles))

def remove_degenerated_triangles(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    o3d_mesh_removed = o3d_mesh.remove_degenerate_triangles()
    return Mesh(np.asarray(o3d_mesh_removed.vertices), np.asarray(o3d_mesh_removed.triangles))

def calc_jacobian_stat(jacobians):
    # jacobians: (N, 3, 3) N matrices of the jacobians
    # theta: Rotation angle
    # omega: Rotation axis
    # A: scaling factor
    # Using the Polar Decomposition J = UP
    # U: Unitary, rotation
    # P: Positive semi-definite Hermitian (Symmetric) Matrix
    W, S, V = np.linalg.svd(jacobians, full_matrices=True)
    P = (V.transpose(0, 2, 1) * S[:, np.newaxis]) @ V # Scaling matrix
    U = W @ V # Rotation matrix
    A = eigvals(P)
    rotvec = Rotation.from_matrix(U).as_rotvec()

    theta = np.linalg.norm(rotvec, axis=-1)
    omega = rotvec / (theta[:, np.newaxis] + 1e-5)


    return theta, omega, A

def apply_deformation(triangles, jacobians):
    # triangles: (N, 3, 3) N triangles where (i, :, j) is the jth vertex of the ith triangle
    # jacobians: (N, 3, 3) N deformation gradients
    fix_point = triangles.mean(axis=1, keepdims=True)
    # fix_point = 0
    
    triangles_shift = triangles - fix_point
    
    spans_t = jacobians @ triangles_shift
    return spans_t + fix_point
    
def quat2exp(quants):

    if len(quants.shape) ==3:
        sp = quants.shape[:-1]
        quants = quants.reshape(-1, 4)
    else:
        sp = None
    results = np.array([(t3d.euler.quat2axangle(q)) for q in quants])
    results_rot = Rotation.from_euler('xyz', np.array([t3d.euler.axangle2euler(v, the) for (v, the) in results])).as_matrix()

    results = np.array([a*b for (a, b) in results])
    if sp:
        return results.reshape((*sp, 3)), results_rot.reshape((*sp, 3, 3))
    else:
        return results, results_rot

def blender2transform(scale, shift, rot):
    # Input:
        # Scale: [X, Y, Z] in blender 3.5
        # Shift: [X, Y, Z] in blender 3.5
        # rot: [X, Y, Z] (in deg) in blender 3.5
    # Output:
        # Transformation object that contains the scale, shift and rotation, which can be applied to mesh vertices via transform_apply

    scale = [scale[0], scale[2], scale[1]]
    shift = [shift[0], shift[2], -shift[1]]
    rot = Rotation.from_euler('xyz', [rot[0], rot[2], -rot[1]], degrees=True)
    return {'scale':scale, 'shift':shift, 'rot':rot}

def transform_apply(vertices, transform):
    # vertices: either (V, 3) for a single mesh or (B, V, 3) for a batch/sequence.
    # Rotation only accepts (N, 3), so flatten before applying and restore the original shape.
    vertices = vertices * transform['scale'] + transform['shift']
    shape = vertices.shape
    vertices = transform['rot'].apply(np.asarray(vertices).reshape(-1, 3))
    return vertices.reshape(shape)

def decompose_jacobians(jacobians, repr='6dof'):
    # jacobians: (B, N, 9) B x N matrices of the jacobians
    # J = UP
    # U: Rotations, representated by the 6dof
    # P: Scaling, representated by a positive semi-definite symmetric matrix
    
    B = jacobians.shape[0]
    N = jacobians.shape[1]
    jacobians = jacobians.reshape(-1, 3, 3)
    jacobians = jacobians.numpy()
    W, S, V = np.linalg.svd(jacobians, full_matrices=True)
    P = (V.transpose(0, 2, 1) * S[:, np.newaxis]) @ V # Scaling matrix
    U = W @ V # Rotation matrix
    U = torch.from_numpy(U)
    P = torch.from_numpy(P)

    if repr == '6dof':
        rots = transforms.matrix_to_rotation_6d(U)
    elif repr == 'quat':
        rots = transforms.matrix_to_quaternion(U)
    elif repr == 'expmap':
        rots = transforms.matrix_to_axis_angle(U)


    scals = P.reshape(-1, 9)[:, [0, 1, 2, 4, 5, 8]]


    return torch.cat((rots.reshape(B, N, -1),  scals.reshape(B, N, -1)), dim=-1)

def reconstruct_jacobians(inputs, repr='matrix', eps=1e-9):
    # inputs: (BS, N, ?) inputs for reconstruct the jacobians
    # ? = 9, repr == 'matrix'
    # ? = 12, repr == '6dof'
    # ? = 10, repr == 'quat'
    # ? = 9,  repr == 'expmap'
    BS = inputs.shape[0]
    N = inputs.shape[1]
    if repr == 'matrix':
        return inputs.reshape(BS, N, 3, 3)
        
    rots = inputs[:, :, :-6]
    scals = inputs[:, :, -6:]
    # rots = rots.reshape(BS, N, 2, 3)
    rot_matrix = torch.empty(BS, N, 3, 3)

    if repr == '6dof':
        rot_matrix = transforms.rotation_6d_to_matrix(rots)
    elif repr == 'quat':
        rot_matrix = transforms.quaternion_to_matrix(rots)
    elif repr == 'expmap':
        rot_matrix = transforms.axis_angle_to_matrix(rots)

    scal_matrix = torch.empty((BS, N, 9)).to(inputs.device)
    scal_matrix[:, :, [0, 1, 2, 4, 5, 8]] = scals
    scal_matrix[:, :, [3, 6, 7]] = scals[:, :, [1, 2, 4]]
    scal_matrix = scal_matrix.reshape(BS, N, 3, 3)

    return torch.matmul(rot_matrix, scal_matrix)


def transformation(vertices, matrix):
    # Input:
    # Vertices: V x 3
    # matrix: 3 x 4
    
    return matrix[:, :3][np.newaxis] @ vertices[:, :, np.newaxis] + matrix[:, 3][np.newaxis, :, np.newaxis]


def positional_encoder(p, size, pe='normal', use_pos=False):
    if pe == 'gauss':
        p_transformed = np.pi * p @ size
        p_transformed = torch.cat(
            [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
    else:
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(size)], dim=-1)
    if use_pos:
        p_transformed = torch.cat([p_transformed, p], -1)
    return p_transformed


def calc_per_face_hks(mesh):
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators([torch.from_numpy(mesh.vertices)], [torch.from_numpy(mesh.faces)], k_eig=128, op_cache_dir='/media/qindafei/SSD/face/cache')
    neutral_hks = diffusion_net.geometry.compute_hks_autoscale(evals_list[0], evecs_list[0], 10)
    neutral_hks = neutral_hks
    neutral_hks_mean = torch.mean(neutral_hks, dim=0, keepdim=True)
    neutral_hks_std = torch.std(neutral_hks, dim=0, keepdim=True)
    neutral_hks = (neutral_hks - neutral_hks_mean) / (neutral_hks_std + 1e-7)
    neutral_hks_per_face = neutral_hks[mesh.faces]
    neutral_hks_per_face = neutral_hks_per_face.mean(dim=-2)
    return neutral_hks_per_face


class Normalizer(object):
    def __init__(self, std_path, device, zero_mean=True):

        if zero_mean:
            self.gradients_std = np.load(os.path.join(std_path, 'gradients_std.npy'))
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.eye(3).view(-1,).unsqueeze(0).unsqueeze(0).to(device).float()
        else:
            self.gradients_mean, self.gradients_std = np.load(os.path.join(std_path, 'wks_mean_std.npz'), allow_pickle=True)['arr_0'].item().values()
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.from_numpy(self.gradients_mean).to(device).float()

    def normalize(self, tensor):

        return (tensor - self.gradients_mean.to(tensor.device)) / self.gradients_std.to(tensor.device)

    def inv_normalize(self, tensor):

        return tensor * self.gradients_std.to(tensor.device) + self.gradients_mean.to(tensor.device)


class Normalizer_landmark(Normalizer):
    def __init__(self, std_path, device):
        with open(os.path.join(std_path, 'landmarks_stat.pkl'), 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()

class Normalizer_img(Normalizer):
    def __init__(self, std_path, device):
        import warnings
        warnings.filterwarnings('ignore')
        with open(os.path.join(std_path, 'img_stat.pkl'), 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()

class WaveKernelSignatureError(Exception):
    pass

class WaveKernelSignature:
    '''
    Computes wave kernel signature for a given mesh
    '''
    def __init__(self,
                 vertices,
                 faces,
                 top_k_eig=40,
                 timestamps=100):
        # vertices, faces are both numpy arrays.

        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.vertices = vertices
        self.faces = faces

        # self.vertices_gpu = torch.from_numpy(vertices).cuda()
        # self.faces_gpu = torch.from_numpy(faces).cuda()

        self.top_k_eig = top_k_eig
        self.timestamps = timestamps
        self.max_iter = 10000

    def compute(self, vis=False, obj_name=None, pos=[0, 0, 0]):
        '''
        compute the wks. Afterwards WKS stores in self.wks
        '''
        cp = igl.connected_components(igl.adjacency_matrix(self.faces))
        assert(cp[0]==1), f"{cp}"
        L = -igl.cotmatrix(self.vertices, self.faces) # this is fast 0.04 seconds
        M = igl.massmatrix(self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI)
        # assert(not numpy.any(numpy.isinf(L)))
        try:
            try:
                try:
                    self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
                        L, self.top_k_eig, M, sigma=0, which='LM', maxiter=self.max_iter)
                except:
                    self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
                        L, self.top_k_eig, M, sigma=1e-4, which='LM', maxiter=self.max_iter)
            except:
                np.nan_to_num(L.data, 0)
                self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
                        L, self.top_k_eig, M, sigma=0, which='LM', maxiter=self.max_iter)

        except:
            raise WaveKernelSignatureError("Error in computing WKS")


        self.eig_vecs /= 200 #np.linalg.norm(self.eig_vecs, axis=0, keepdims=True)


        delta = (np.log(self.eig_vals[-1]) - np.log(self.eig_vals[1])) / self.timestamps
        sigma = 7 * delta
        e_min = np.log(self.eig_vals[1]) + 2 * delta
        e_max = np.log(self.eig_vals[-1]) - 2 * delta
        es = np.linspace(e_min, e_max, self.timestamps)  # T
        self.delta = delta

        
        coef = np.expand_dims(es, 0) - np.expand_dims(np.log(self.eig_vals[1:]), 1)  # (K-1)xT
        coef = np.exp(-np.square(coef) / (2 * sigma * sigma))  # (K-1)xT #element wise square
        sum_coef = coef.sum(0)  # T
        K = np.matmul(np.square(self.eig_vecs[:, 1:]), coef)  # VxT. Scaling of the eigen vectors by coef. Coef depends only on the eigen values. Triangulation agnostic.
        self.wks = K / np.expand_dims(sum_coef, 0)  # VxT Scaling of the eigen vectors by sum_coef. Coef depends only on the eigen values. Triangulation agnostic.
        # print(np.linalg.norm(self.wks, axis=0))
        # print(np.linalg.norm(self.wks, axis=1))


def load_state_dict(model, state_dict):
    key = list(state_dict.keys())[0]
    if 'module' in key:
        if type(model) != torch.nn.DataParallel:
            state_dict = {name[7:]:value for name, value in state_dict.items()}
    else:
        if type(model) == torch.nn.DataParallel:
            state_dict = {'module.' + name:value for name, value in state_dict.items()}
    if 'module' in list(state_dict.keys())[0]:
        cnn_enc = {name[12 +7:]:value for name, value in state_dict.items() if 'img_encoder' in name}
        cnn_fc = {name[7+7:]:value for name, value in state_dict.items() if 'img_fc' in name}
        exp_enc = {name[8+7:]:value for name, value in state_dict.items() if 'encoder' in name and 'img' not in name}
        iden_enc = {name[10+7:]:value for name, value in state_dict.items() if 'global_pn' in name}
        mlp = {name[7 + 8:]:value for name, value in state_dict.items() if 'linears' in name}
        linear_out = {name[7 + 11:]:value for name, value in state_dict.items() if 'linear_out' in name}
        gns = {name[7 + 4:]:value for name, value in state_dict.items() if 'gns' in name}
        if model.module.img_encoder is not None:
            model.module.img_encoder.load_state_dict(cnn_enc)
            if model.module.img_enc_type == 'cnn':
                model.module.img_fc.load_state_dict(cnn_fc)
        try:
            model.module.encoder.load_state_dict(exp_enc)
        except RuntimeError: # This is a dfn
            exp_enc = {name[4:]:exp_enc[name] for name in list(exp_enc.keys())[10:]}
            model.module.encoder.dfn.load_state_dict(exp_enc)
        if len(iden_enc.keys()) > 0:
            try:
                model.module.global_pn.load_state_dict(iden_enc)
            except RuntimeError:
                iden_enc = {name[4:]:iden_enc[name] for name in list(iden_enc.keys())[10:]}
                model.module.global_pn.dfn.load_state_dict(iden_enc)
        model.module.linears.load_state_dict(mlp)
        model.module.linear_out.load_state_dict(linear_out)
        model.module.gns.load_state_dict(gns)
    else:
        cnn_enc = {name[12:]:value for name, value in state_dict.items() if 'img_encoder' in name}
        cnn_fc = {name[7:]:value for name, value in state_dict.items() if 'img_fc' in name}
        exp_enc = {name[8:]:value for name, value in state_dict.items() if 'encoder' in name and 'img' not in name}
        iden_enc = {name[10:]:value for name, value in state_dict.items() if 'global_pn' in name}
        mlp = {name[ 8:]:value for name, value in state_dict.items() if 'linears' in name}
        linear_out = {name[11:]:value for name, value in state_dict.items() if 'linear_out' in name}
        gns = {name[4:]:value for name, value in state_dict.items() if 'gns' in name}
        if model.img_encoder is not None:
            model.img_encoder.load_state_dict(cnn_enc)
            if model.img_enc_type == 'cnn':
                model.img_fc.load_state_dict(cnn_fc)
        try:
            model.encoder.load_state_dict(exp_enc)
        except RuntimeError: # This is a dfn
            exp_enc = {name[4:]:exp_enc[name] for name in list(exp_enc.keys())[10:]}
            model.encoder.dfn.load_state_dict(exp_enc)
        if len(iden_enc.keys()) > 0:
            try:
                model.global_pn.load_state_dict(iden_enc)
            except RuntimeError: # This is a dfn
                iden_enc = {name[4:]:iden_enc[name] for name in list(iden_enc.keys())[10:]}
                model.global_pn.dfn.load_state_dict(iden_enc)
        model.linears.load_state_dict(mlp)
        model.linear_out.load_state_dict(linear_out)
        model.gns.load_state_dict(gns)

    return model


def get_dfn_info(mesh, cache_dir=None, map_location='cuda'):
    verts_list = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.faces).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(mesh.faces)]
    dfn_info = [_.to(map_location).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def get_dfn_info_except_grad(verts, face, k_eig=128, dtype=torch.float32, device='cpu'):
    eps = 1e-8



    L = pp3d.cotan_laplacian(verts, face, denom_eps=1e-8)
    massvec_np = pp3d.vertex_areas(verts, face)
    massvec_np += eps * np.mean(massvec_np)
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col
    L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
    massvec_eigsh = massvec_np
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps
    failcount = 0
    while True:
        try:
            # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
            evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)
        
            # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
            evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

            break
        
        except Exception as e:
            print(e)
            if(failcount > 3):
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)
        
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    # print('Child return')
    to_return =  [massvec, L.indices(), L.values(), L.shape, evals, evecs]
    # to_return = []
    # to_return =  [massvec, evals, evecs]
    # to_return = L
    return to_return

def get_gradient_info(mesh, edges, verts_outgoing, rhs_mat_list, row, col):
    from diffusion_net.geometry import build_tangent_frames, edge_tangent_vectors, build_grad_custom, build_grad
    from diffusion_net.utils import sparse_np_to_torch
    verts_np = mesh.vertices
    faces_np = mesh.faces
    N = len(verts_np)
    verts = torch.from_numpy(verts_np)
    faces = torch.from_numpy(faces_np)
    # L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    frames = build_tangent_frames(torch.from_numpy(verts_np), torch.from_numpy(faces_np), normals=None)
    # # massvec_np = pp3d.vertex_areas(verts_np, faces_np)
    # # massvec_np += eps * np.mean(massvec_np)
    # L_coo = L.tocoo()
    # inds_row = L_coo.row
    # inds_col = L_coo.col
    # edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), dtype=torch.long)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    values = build_grad_custom(verts, edges, edge_vecs, verts_outgoing, rhs_mat_list)
    mat = scipy.sparse.coo_matrix(
        (values, (row, col)), shape=(
            N, N)).tocsc()
    # mat_ = build_grad(verts, edges, edge_vecs, row, col).tocsc()
    # assert np.linalg.norm(mat.data - mat_.data) / N < 1e-6
    gradX_np = np.real(mat)
    gradY_np = np.imag(mat)
    # assert (gradX_np.data == np.real(values)).all()
    # assert (gradY_np == np.imag(values)).all()
    
    gradX = sparse_np_to_torch(gradX_np).to(dtype=float)
    gradY = sparse_np_to_torch(gradY_np).to(dtype=float)
    ind = gradX._indices()
    gradX = gradX._values()
    gradY = gradY._values()
    
    return ind, gradX, gradY

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def pix_to_face_pytorch(neutral_obj, renderer):
    img_neutral, fragments = renderer.renderbatch([torch.from_numpy(neutral_obj.vertices).float().to('cuda')], [torch.from_numpy(neutral_obj.faces).to('cuda')], reverse=True)
    img_neutral = img_neutral[0].cpu().numpy().astype(np.float32)
    p2f = fragments.pix_to_face[0, ..., 0].cpu().float()
    zbuf = fragments.zbuf[0, ..., 0].cpu().float()
    mapping = torch.zeros((neutral_obj.faces.shape[0], p2f.shape[0]**2))
    for i in range(mapping.shape[0]):
        mapping[i][p2f.reshape(-1) == i] = 1
        if mapping[i].max() != 0:
            mapping[i] /= torch.sum(mapping[i])
    mapping = mapping.float().to_sparse()
    return mapping, zbuf

if flag_pytorch3d:
    class renderer:
        def __init__(self, view_d=6, img_size=1024, fragments=False):

            import warnings
            warnings.filterwarnings('ignore')
            from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras, 
            PointLights, 
            Materials, 
            RasterizationSettings, 
            MeshRenderer,
            MeshRendererWithFragments,
            MeshRasterizer)

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")
            R, T = look_at_view_transform(view_d, 0, 0) 
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
            raster_settings = RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
                cull_backfaces=True
            )
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            self.return_fragment = fragments
            if self.return_fragment:
                rd = MeshRendererWithFragments
            else:
                rd = MeshRenderer
            # renderer = rd(
            #     rasterizer=MeshRasterizer(
            #         cameras=cameras, 
            #         raster_settings=raster_settings
            #     ),
            #     shader=pytorch3d.renderer.HardFlatShader(
            #         device=device, 
            #         cameras=cameras,
            #         lights=lights
            #     )
            # )
            materials = Materials(
                device=device,
                specular_color=[[0.0, 0.0, 0.0]],
                shininess=100
            )
            # color = [172, 219, 255]
            color = torch.tensor([255, 255, 255]) / 2 / 255
            lights = PointLights(device=device, location=[[0.0, 0.0, 6]])
            renderer = rd(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=pytorch3d.renderer.HardFlatShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights
                )
            )

            self.renderer = renderer
            self.color = color
            self.device = device
            self.materials = materials

        def renderbatch(self, vertices, faces, reverse=False):
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
            meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).unsqueeze(0).expand(len(meshes), meshes.num_verts_per_mesh()[0], -1))
            if self.return_fragment:
                images, fragments = self.renderer(meshes,materials=self.materials)
            else:
                images = self.renderer(meshes,materials=self.materials)
                fragments = None
            if reverse:
                return 1 - images[..., :3], fragments
            return images[..., :3], fragments


        def mesh2img(self, mesh, reverse=False, noise=False):
            mesh = pytorch3d.structures.Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(self.device)], faces=[torch.from_numpy(mesh.faces).to(self.device)])
            mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).expand_as(mesh.verts_packed())[None])
            if self.return_fragment:
                images, fragments = self.renderer(mesh,materials=self.materials)
            else:
                images = self.renderer(mesh,materials=self.materials)
                fragments = None
            if noise:
                images += (torch.randn(images.shape[:3]).unsqueeze(-1) * 0.01).to(images.device)
            if reverse:
                return ((1 - images[0, ..., :3].detach().cpu().numpy()) * 255).astype(np.uint8), fragments
            return (images[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8), fragments

# class ldmk:

#     def __init__(self, view_d=6):

#         import warnings
#         warnings.filterwarnings('ignore')
#         sys.path.append('./third_party/pytorch_face_landmark')
#         from test_batch_detections import detect_landmark, load_model
#         from pytorch3d.renderer import (
#         look_at_view_transform,
#         FoVPerspectiveCameras, 
#         PointLights, 
#         Materials, 
#         RasterizationSettings, 
#         MeshRenderer, 
#         MeshRasterizer)

#         if torch.cuda.is_available():
#             device = torch.device("cuda:0")
#             torch.cuda.set_device(device)
#         else:
#             device = torch.device("cpu")
#         R, T = look_at_view_transform(view_d, 0, 0) 
#         cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
#         raster_settings = RasterizationSettings(
#             image_size=1024, 
#             blur_radius=0.0, 
#             faces_per_pixel=1, 
#         )
#         lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
#         renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(
#                 cameras=cameras, 
#                 raster_settings=raster_settings
#             ),
#             shader=pytorch3d.renderer.HardFlatShader(
#                 device=device, 
#                 cameras=cameras,
#                 lights=lights
#             )
#         )
#         materials = Materials(
#             device=device,
#             specular_color=[[0.0, 0.0, 0.0]],
#             shininess=100
#         )
#         # color = [172, 219, 255]
#         color = torch.tensor([255, 255, 255]) / 2
#         lights = PointLights(device=device, location=[[0.0, 0.0, 3]])
#         renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(
#                 cameras=cameras, 
#                 raster_settings=raster_settings
#             ),
#             shader=pytorch3d.renderer.HardPhongShader(
#                 device=device, 
#                 cameras=cameras,
#                 lights=lights
#             )
#         )
#         model, detector = load_model(checkpoint_dir='./third_party/pytorch_face_landmark/checkpoint/')

#         self.renderer = renderer
#         self.model = model
#         self.detector = detector
#         self.color = color
#         self.device = device
#         self.materials = materials
#         self.detect_landmark = detect_landmark

#     def mesh2ldmk(self, mesh):
#         mesh = pytorch3d.structures.Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(self.device)], faces=[torch.from_numpy(mesh.faces).to(self.device)])
#         mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).expand_as(mesh.verts_packed())[None]/255)
#         images = self.renderer(mesh,materials=self.materials)
#         test_fig = images[0, ..., :3].cpu().numpy() * 255
#         landmark, img = self.detect_landmark(self.model, self.detector, test_fig)
#         return landmark, img

#     def mesh2img(self, mesh, reverse=False, noise=False):
#         mesh = pytorch3d.structures.Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(self.device)], faces=[torch.from_numpy(mesh.faces).to(self.device)])
#         mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).expand_as(mesh.verts_packed())[None]/255)
#         images = self.renderer(mesh,materials=self.materials)
#         if noise:
#             images += (torch.randn(images.shape[:3]).unsqueeze(-1) * 0.01).to(images.device)
#         if reverse:
#             return ((1 - images[0, ..., :3].detach().cpu().numpy()) * 255).astype(np.uint8)
#         return (images[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)


if __name__ == '__main__':
    
    from glob import glob
    from vedo import trimesh2vedo, show, screenshot, Plotter, settings
    from tqdm import tqdm
    # mesh = trimesh.load('./data/ICT_live_100/000_neutral.obj', process=False, maintain_order=True)
    # WKS = WaveKernelSignature(mesh.vertices, mesh.faces)
    # WKS.compute(vis=True, obj_name='live_100_000')
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # objs_f = glob('./meshes/test-wks/*.obj')
    objs = glob('/media/qindafei/SSD/face/data/ICT_live_100/*.obj')
    verts = []
    for f in objs:
        mesh = Mesh.load(f)
        verts.append(mesh.vertices[np.newaxis])
        face = mesh.faces
    verts = torch.from_numpy(np.concatenate(verts, axis=0))
    face = torch.from_numpy(face)
    calc_norm_torch(verts, face)