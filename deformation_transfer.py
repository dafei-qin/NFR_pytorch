import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
from copy import deepcopy
import igl
import trimesh
import numpy as np
np.seterr(all='ignore')
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy.sparse.linalg import splu as sparse_lu
import scipy.sparse as sparse
import scipy
from matplotlib import pyplot as plt
from cupyx.scipy.sparse.linalg import SuperLU as cupy_SuperLU
from cupyx.scipy import sparse as cupy_sparse
import cupy
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from torch_sparse import spmm, transpose
# from cholespy import CholeskySolverF, MatrixType
import time
import logging
from myutils.myutils import Mesh
# from Mesh import Mesh


vecQR = np.vectorize(np.linalg.qr, signature='(m,n)->(m,p),(p,n)')


def calculate_jacobians(neutral_mesh, vertices):
    transfer = Transfer(neutral_mesh, deepcopy(neutral_mesh))
    jacobians = []
    for v in vertices:
        current_mesh = Mesh(v, neutral_mesh.faces)
        j = transfer.deformation_gradient(current_mesh)
        jacobians.append(deepcopy(j[np.newaxis]))

    return np.concatenate(jacobians, axis=0)


class Transfer:
    def __init__(self, source: Mesh, target: Mesh, project: bool=False, area: bool=True, device='cuda:0', use_chol=False):
        self.source = source
        self.target = target
        self.do_project = project
        # self.coeffs = self.target_coef.astype(np.single)
        self.coeffs = self.target_coef # IMPORTANT: change to single precision will lose ~11 order of reconstruct accuracy (1e-14 -> 1e-3)
        self.device = int(device[-1])
        # self.coeffs = cupy.array(self.coeffs)
        self.A, self.cupy_A, self.idxs, self.vals = self.target_A
        # self.A.tocsc()
        if area:
            self.area, self.cupy_area = self.target_area
        else:
            self.area = None
        # self.coeffs = cupy.array(self.coeffs)
        # self.A = cupy.array(self.A)
        # self.A = cupy_sparse.csc_matrix(self.A)
        # self.area = cupy.array(self.area)
        if area:
            if use_chol:
                self.A = self.A.astype(np.single)
                self.area = self.area.astype(np.single)
                self.ATareaA = (self.A.T @ self.area @ self.A).tocoo()
                rows = torch.from_numpy(self.ATareaA.row).to(self.device)
                cols = torch.from_numpy(self.ATareaA.col).to(self.device)
                data = torch.from_numpy(self.ATareaA.data).to(self.device)
                print(np.linalg.matrix_rank(self.ATareaA.todense()))
                try:
                    self.solver = CholeskySolverF(self.ATareaA.shape[0], rows, cols, data, MatrixType.COO)
                except ValueError:
                    logging.warning('WARNING! Add correction matrix for Cholesky Decomposition.')
                    self.ATareaA += (sparse.identity(self.ATareaA.shape[0]) * 1e-6)
                    self.ATareaA = self.ATareaA.tocoo()
                    rows = torch.from_numpy(self.ATareaA.row).to(self.device)
                    cols = torch.from_numpy(self.ATareaA.col).to(self.device)
                    data = torch.from_numpy(self.ATareaA.data).to(self.device)
                    self.solver = CholeskySolverF(self.ATareaA.shape[0], rows, cols, data, MatrixType.COO)
            else:
                self.ATareaA = (self.A.T @ self.area @ self.A).tocsc()
                self.lu = sparse_lu(self.ATareaA)
                self.solver = cupy_SuperLU(self.lu)

            # self.ATarea = (self.A.T.astype(np.single) @ self.area.astype(np.single)).T # Later transpose back
            self.ATarea = (self.A.T @ self.area).T # Later transpose back
            self.ATarea.sort_indices()
            self.ATarea = self.ATarea.tolil()
            self.lil_rows = self.ATarea.rows
            rows = [np.repeat(row_idx, len(self.lil_rows[row_idx])) for row_idx in range(len(self.lil_rows))]
            rows = [x for y in rows for x in y]
            cols = [x for y in self.lil_rows for x in y]
            with cupy.cuda.Device(self.device):
                self.idxs = cupy.concatenate((cupy.array(rows)[:, cupy.newaxis], cupy.array(cols)[:, cupy.newaxis]), axis=1).T
                # self.vals = (self.cupy_A.T @ self.cupy_area).T.data
                # self.vals = cupy.array([x for y in self.ATarea.data for x in y], dtype=np.single)
                self.vals = cupy.array([x for y in self.ATarea.data for x in y])
        else:
            raise NotImplementedError
            self.lu = sparse_lu((self.A.T @ self.A).tocsc())

    def __call__(self, pose: Mesh, project=False):
        def _get_mass_matrix(vertices,faces,is_sparse):

            d_area = igl.doublearea(vertices,faces)
            d_area = np.hstack((d_area, d_area, d_area))
            if is_sparse:
                return sparse.csc_matrix(sparse.diags(d_area))
            return sparse.diags(d_area)
        S = self.deformation_gradient(pose)
        if project: # Project into the tangent space of the target 
            target_fv = self.get_fv(self.target)
            target_W = (target_fv[:, :, 1:] - target_fv[:, :, 0][:, :, np.newaxis]) # The basis are not formed by perpendicular unit vectors. TODO: Use igl.local_basis
            b0, b1, b_o = igl.local_basis(pose.vertices, pose.faces.astype(int))
            igl_W = np.concatenate([b0[:, :, np.newaxis], b1[:, :, np.newaxis]], axis=-1)
            # S = S @ target_W @ self.coeffs # TODO: Why multiply self.coeffs here?
            # S = S @ igl_W # TODO: Why multiply self.coeffs here?
            # S = S.reshape(-1, 2)
            # vertices = self.lu.solve(self.A.T @ self.area @ S)
            igl_cot = igl.cotmatrix(pose.vertices, pose.faces)
            igl_grad = igl.grad(pose.vertices, pose.faces)
            igl_lap = igl_grad.T @ self.area @ igl_grad
            lu_igl = sparse_lu(igl_lap)
            S = S.reshape(-1, 3)
            v1 = self.lu.solve(self.A.T @ self.area @ S)
            v2 = lu_igl.solve(igl_grad.T @ self.area @ S)
            print()
        else:
            S = S.reshape(-1, 3)
        if self.area is not None:     
            vertices = self.lu.solve(self.A.T @ self.area @ S)
        else:
            vertices = self.lu.solve(self.A.T @ S)

        return Mesh(vertices, self.target.faces)
    

    @property
    def target_coef(self):
        target_fv = self.get_fv(self.target)
        target_W = target_fv[:, :, 1:] - target_fv[:, :, 0][:, :, np.newaxis]

        q_, r_ = vecQR(target_W)
        coeffs = np.linalg.inv(r_) @ q_.transpose(0, 2, 1)
        return coeffs

    @property
    def target_A(self):
        
        target_fv = self.get_fv(self.target).astype(np.single)

        # A = sparse.lil_matrix((3 * target_fv.shape[0], len(self.target.vertices)))
        # A = cupy_sparse.csc_matrix((3 * target_fv.shape[0], len(self.target.vertices)))
        coeffs = np.concatenate((-(self.coeffs[:, 0] + self.coeffs[:, 1])[:, np.newaxis], self.coeffs), axis=1).reshape(-1)
        row_idxs = np.arange(3 * target_fv.shape[0]).reshape(-1, 3).repeat(3, axis=0).reshape(-1)
        col_idxs = np.array(self.target.faces).repeat(3)
        A = sparse.csc_matrix((coeffs, (row_idxs, col_idxs)))
        with cupy.cuda.Device(self.device):
            cupy_coeffs = cupy.array(coeffs)
            cupy_row_idxs = cupy.array(row_idxs)
            cupy_col_idxs = cupy.array(col_idxs)
            cupy_A = cupy_sparse.csc_matrix((cupy_coeffs, (cupy_row_idxs, cupy_col_idxs)))


            return A, cupy_A, cupy.concatenate((cupy_row_idxs[:, cupy.newaxis], cupy_col_idxs[:, cupy.newaxis]), axis=1).T, cupy_coeffs

    @property
    def target_area(self):
        target_fv = self.get_fv(self.target)
        AB = target_fv[:, :, 1] - target_fv[:, :, 0]
        AC = target_fv[:, :, 2] - target_fv[:, :, 0]
        area = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)
        area_diag = np.repeat(area[:, np.newaxis], 3, axis=-1).reshape(-1)
        area = sparse.diags(area_diag)
        with cupy.cuda.Device(self.device):
            cupy_area = cupy_sparse.diags(area_diag)
        return area, cupy_area


        

    
    @classmethod
    def get_fv(cls, mesh: Mesh):
        return mesh.vertices[mesh.faces].transpose(0, 2, 1)

    @classmethod
    def span_matrix(cls, mesh: Mesh):
        fv = cls.get_fv(mesh)

        v1 = fv[:, :, 0]
        v2 = fv[:, :, 1]
        v3 = fv[:, :, 2]
        cross = np.cross(v2 - v1, v3 - v1)
        v4 = v1 +  cross / np.sqrt(np.linalg.norm(cross, axis=-1))[:, np.newaxis]

        return np.stack((v2 - v1, v3 - v1, v4 - v1), axis=-1)
    @classmethod
    def span_to_fv(cls, span):
        cross = np.cross(span[:, :, 0], span[:, :, 1])
        v1 = span[:, :, 2] - cross / np.sqrt(np.linalg.norm(cross, axis=-1))[:, np.newaxis]
        v2 = span[:, :, 0] + v1
        v3 = span[:, :, 1] + v1
        return np.stack((v1, v2, v3), axis=-1)
    
    
    def deformation_gradient(self, pose: Mesh) -> np.ndarray:
        source_span = self.span_matrix(self.source)
        pose_span = self.span_matrix(pose)
        Q = pose_span @ np.linalg.inv(source_span)

        return Q.transpose(0, 2, 1)


class deformation_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, solver, idxs, vals, shape):
        # input: (B, F, ?) The jacobian information, shape depends on the expression format:
        # repr == 'matrix': (B, F, 3, 3)
        # repr == '6dof':   (B, F, 12)
        # repr == 'quat':   (B, F, 10)
        # repr == 'expmap': (B, F, 9)

        # input = jacobians
        # lu_solver: The solver of the deformation transfer problem
        # idxs, vals: the sparse representation of the rhs matrix
        # shape: the shape of the rhs matrix
        batch_size = input.shape[0]
        input = input.reshape(input.shape[0], -1, 3)
        input = input.transpose(1, 0)
        input = input.reshape(input.shape[0], -1) # since lu.solve only support dim (3F, D), we concate all additional dimentions to the last dim

        ctx.solver = solver
        ctx.idxs = idxs
        ctx.vals = vals
        ctx.shape = shape
        ctx.set_materialize_grads(False)
        # ctx.grad = from_dlpack(ctx.solver.solve(ctx.rhs.toarray()).toDlpack())
        # cupy_input = cupy.from_dlpack(to_dlpack(input))
        b = spmm(idxs, vals, m=shape[0], n=shape[1], matrix=input)
        b = cupy.from_dlpack(to_dlpack(b))
        cupy_output = ctx.solver.solve(b)
        output = from_dlpack(cupy_output.toDlpack())
        output = output.reshape(-1, batch_size, 3)
        output = output.transpose(0, 1)
        # print(f'forward time: {time.time() - t:.4f}s')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t = time.time()
        if grad_output is None:
            return None, None, None, None, None

    
        grad_output = grad_output.permute(1, 0, 2).reshape(grad_output.shape[1], -1)
        grad = from_dlpack(ctx.solver.solve(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack())
        if grad.isnan().any():
            print(grad)
            raise ValueError('Nan found after solving for gradient!')
        # raise ValueError
        ctx.idxs, ctx.vals = transpose(ctx.idxs, ctx.vals, m=ctx.shape[0], n=ctx.shape[1])
        grad = spmm(ctx.idxs, ctx.vals, m=ctx.shape[1], n=ctx.shape[0], matrix=grad)
        if grad.isnan().any():
            print(grad)
            raise ValueError('Nan found after spmm with rhs!')
        grad = grad.reshape(grad.shape[0], -1, 3)
        grad = grad.transpose(0, 1)
        grad = grad.reshape(grad.shape[0], -1, 3, 3)
        # grad = torch.matmul(grad_output, grad)
        # grad = grad.permute(0, 2, 1).reshape(grad_output.shape[0],-1 , 3, 3)
        # print(f'Backward time: {time.time() - t:.4f}s')
        
        # Some clean up
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        del ctx.solver

        return grad, None, None, None, None


class deformation_gradient_cho(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, solver, idxs, vals, shape):
        raise NotImplementedError
        # input: (B, F, ?) The jacobian information, shape depends on the expression format:
        # repr == 'matrix': (B, F, 3, 3)
        # repr == '6dof':   (B, F, 12)
        # repr == 'quat':   (B, F, 10)
        # repr == 'expmap': (B, F, 9)

        # input = jacobians
        # lu_solver: The solver of the deformation transfer problem
        # idxs, vals: the sparse representation of the rhs matrix
        # shape: the shape of the rhs matrix
        batch_size = input.shape[0]
        input = input.reshape(input.shape[0], -1, 3)
        input = input.transpose(1, 0)
        input = input.reshape(input.shape[0], -1) # since lu.solve only support dim (3F, D), we concate all additional dimentions to the last dim

        ctx.solver = solver
        ctx.idxs = idxs
        ctx.vals = vals
        ctx.shape = shape
        ctx.set_materialize_grads(False)
        # ctx.grad = from_dlpack(ctx.solver.solve(ctx.rhs.toarray()).toDlpack())
        # cupy_input = cupy.from_dlpack(to_dlpack(input))
        b = spmm(idxs, vals, m=shape[0], n=shape[1], matrix=input)
        output = torch.zeros_like(b)
        ctx.solver.solve(b, output)
        output = output.reshape(-1, batch_size, 3)
        output = output.transpose(0, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        t = time.time()
        if grad_output is None:
            return None, None, None, None, None

        grad_output = grad_output.permute(1, 0, 2).reshape(grad_output.shape[1], -1)
        grad = torch.zeros_like(grad_output)
        ctx.solver.solve(grad_output, grad)
        # raise ValueError
        ctx.idxs, ctx.vals = transpose(ctx.idxs, ctx.vals, m=ctx.shape[0], n=ctx.shape[1])
        grad = spmm(ctx.idxs, ctx.vals, m=ctx.shape[1], n=ctx.shape[0], matrix=grad)
        grad = grad.reshape(grad.shape[0], -1, 3)
        grad = grad.transpose(0, 1)
        grad = grad.reshape(grad.shape[0], -1, 3, 3)

        return grad, None, None, None, None

def calc_transform(mesh_x, mesh_y):
    """
    Calculate the Rotation R, Scaling S and translation t from triangle meshs mesh_x to mesh_y.
    I.e. mesh_y = R @ mesh_x + t
    The vertices of a mesh should be separated by column.
    I.e. mesh_x[i, :, j] = the jth vertex of the ith mesh.
    --------------------
    Inputs:
    ---------------------
    mesh_x (N, 3, 3):   N meshs of x
    mesh_y (N, 3, 3):   N meshs of y

    Returns:
    ----------------------
    Q   (N, 3, 3):      Rotations and scaling
    t   (N, 3, 1):      Translations

    """
    def form_V(mesh):
        if len(mesh.shape) == 2:
            mesh = mesh[np.newaxis, :, :]
        v1 = mesh[:, :, 0]
        v2 = mesh[:, :, 1]
        v3 = mesh[:, :, 2]
        cross = np.cross(v2 - v1, v3 - v1)
        v4 = v1 +  cross / np.sqrt(np.linalg.norm(cross, axis=-1))[:, np.newaxis]

        return np.stack((v2 - v1, v3 - v1, v4 - v1), axis=-1)

    
    V_x = form_V(mesh_x)
    V_y = form_V(mesh_y)

    Q = V_y @ np.linalg.inv(V_x)

    t = mesh_y - Q @ mesh_x
    assert np.allclose(t, t[:, :, 0][:, :, np.newaxis]) , "The transformation t should be the same for all three vetrices, otherwise the data order is wrong. The scripts only support trianglated mesh."

    return Q, t[:, :, 0][:, :, np.newaxis]



# def vectorized_qr(M):

def calc_qr(face_vertices):
    '''
    Return:
    --------
    q (F, 3, 2)
    r (F, 2, 2)
    '''
    W_3x2 = (face_vertices[:, :, 1:] - face_vertices[:, :, 0][:, :, np.newaxis])
    # W_3x2 -= 
    vecQR = np.vectorize(np.linalg.qr, signature='(m,n)->(m,p),(p,n)')
    q, r = vecQR(W_3x2)
    return q, r
    
def calc_S(face_vertices, Q, q, r):
    '''
    Return:
    -------
    S (F, 3, 3)
    '''
    W_3x2 = face_vertices[:, :, 1:] - face_vertices[:, :, 0][:, :, np.newaxis]
    r_inv = np.linalg.inv(r)

    return Q @ W_3x2 @ r_inv @ q.transpose(0, 2, 1)

def calc_A(face_vertices, faces_idx, num_of_vertices):
    '''
    Return:
    --------
    A (3*F, V)
    '''
    q, r = calc_qr(face_vertices) # Target
    coeffs = np.linalg.inv(r) @ q.transpose(0, 2, 1)

    # Multiply by area.
    # A = np.zeros((3 * face_vertices.shape[0], num_of_vertices))
    A = scipy.sparse.lil_matrix((3 * face_vertices.shape[0], num_of_vertices))
    for f in range(face_vertices.shape[0]):
        A[3*f:3*f+3, faces_idx[f, 1]] = coeffs[f, 0]
        A[3*f:3*f+3, faces_idx[f, 2]] = coeffs[f, 1]
        A[3*f:3*f+3, faces_idx[f, 0]] = -(coeffs[f, 0] + coeffs[f, 1])

    return A

def eliminate_constrains(A, S, vertices_idx, vertices_gt):
    '''
    A (3*F, V)
    S (F, 3, 3)
    vertices_idx
    vertices_gt
    '''
    col_list = np.arange(A.shape[1])
    col_list = np.delete(col_list, vertices_idx)

    A = A.tocsc()
    S = S.reshape((-1, 3))
    S -= A[:, vertices_idx] @ vertices_gt
    A = scipy.sparse.lil_matrix(A[:, col_list])

    # A[:, vertices_idx] = 0
    return A, S

def calc_deform(A, S, tol, use_lu=True, area=None):
    if use_lu:
        A.tocsc()
        if area is not None:
            area = np.repeat(area[:, np.newaxis], 3, axis=-1).reshape(-1)
            area = sparse.diags(area)
            lu = sparse_lu((A.T @ area @ A).tocsc())
            return lu.solve(A.T @ area @ S)
        else:
            lu = sparse_lu(A.T @ A)
            return lu.solve(A.T @ S)

    # A = np.float32(A)
    # S = np.float32(A)
    A = A.tocsr()
    # AA = A.T @ A
    # AA = AA.tocsc()
    # plt.matshow(AA.todense())
    # plt.show()
    # inv = sparse_inv(AA)
    result = np.zeros((A.shape[1], 3))
    for i in range(3):
        result[:, i] = sparse_lsqr(A, S[:, i], atol=tol, btol=tol)[0]
    return result

def calc_area(face_vertices):
    AB = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    AC = face_vertices[:, :, 2] - face_vertices[:, :, 0]
    area = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)
    return area

def deformation_transfer(source_orig, source_deformed, target_orig, tol=1e-8, use_lu=True):
    vertices, faces = source_orig.vertices, source_orig.faces
    deform_vertices, _ = source_deformed.vertices, source_deformed.faces
    vertices_target, faces_target  = target_orig.vertices, target_orig.faces

    if np.max(faces) == vertices.shape[0]:
        faces -= 1
        faces_target -= 1
    
    orig_face_vertices = vertices[faces].transpose(0, 2, 1)
    deform_face_vertices = deform_vertices[faces].transpose(0, 2, 1)
    target_face_vertices = vertices_target[faces_target].transpose(0, 2, 1)

    Q, t = calc_transform(orig_face_vertices, deform_face_vertices)

    assert np.allclose(Q @ orig_face_vertices + t, deform_face_vertices) # Assert the transformation is correct

    area = calc_area(target_face_vertices)
    q, r = calc_qr(target_face_vertices) # Calculate the QR factorization of the W matrix
    # S = calc_S(orig_face_vertices, Q, q, r)
    S = calc_S(target_face_vertices, Q, q, r)
    # S = Q
    S = S.transpose(0, 2, 1)
    A = calc_A(target_face_vertices, faces, vertices.shape[0])
    S = S.reshape(-1, 3)
    # assert np.allclose(A @ deform_vertices, S)

    deformed_vertices = calc_deform(A, S, tol=tol, use_lu=use_lu, area=area)
    # if np.max(faces_target) == vertices_target.shape[0] - 1:
    #     faces_target += 1
    return (deformed_vertices, faces_target)
    
if __name__ == "__main__":
    
    pass
    # pass
    # vertices, faces = load_obj_file('./mh/emory_default_tri.obj')
    # vertices_target, faces_target = load_obj_file('./mh/hadley_default_tri.obj')
    # deform_vertices, _ = load_obj_file('./mh/emory_02_tri.obj')
    # face_vertices = vertices[faces].transpose(0, 2, 1)
    # deform_face_vertices = deform_vertices[faces].transpose(0, 2, 1)
    # face_vertices_target = vertices_target[faces_target].transpose(0, 2, 1)
    # Q, t = calc_transform(face_vertices, deform_face_vertices)
    
    # assert np.allclose(Q @ face_vertices + t, deform_face_vertices)
    # q, r = calc_qr(face_vertices)
    # S = calc_S(face_vertices, Q, q, r)
    # target_area = calc_area(face_vertices_target)[:, np.newaxis, np.newaxis]
    # S = S.transpose(0, 2, 1) * target_area
    # S = S.reshape(-1, 3)
    # A = calc_A(face_vertices_target, faces, vertices.shape[0]) # The face_vertices of the target mesh
    # # assert np.allclose(A @ deform_vertices, S)
    # vertices_constraint_idx = np.array([0, 1])
    # vertices_constraint_gt = deform_vertices[:2]
    # # A, S = eliminate_constrains(A, S, vertices_constraint_idx, vertices_constraint_gt)
    # deformed_vertices = calc_deform(A, S, tol=1e-8, area=target_area)
    # # for i, idx in enumerate(vertices_constraint_idx):
    # #     deformed_vertices = np.insert(deformed_vertices, idx, vertices_constraint_gt[i], axis=0)
    # # deform_vertices[vertices_constraint_idx] = vertices_constraint_gt
    # writeObj('./mh/deformed_02_area.obj', deformed_vertices, faces)


    # print()

