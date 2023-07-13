from copy import deepcopy
import igl

import numpy as np
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

import time
import logging
from myutils import Mesh
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

        if area:
           
            self.ATareaA = (self.A.T @ self.area @ self.A).tocsc()
            self.lu = sparse_lu(self.ATareaA)
            self.solver = cupy_SuperLU(self.lu)


            self.ATarea = (self.A.T @ self.area).T # Later transpose back
            self.ATarea.sort_indices()
            self.ATarea = self.ATarea.tolil()
            self.lil_rows = self.ATarea.rows
            rows = [np.repeat(row_idx, len(self.lil_rows[row_idx])) for row_idx in range(len(self.lil_rows))]
            rows = [x for y in rows for x in y]
            cols = [x for y in self.lil_rows for x in y]
            with cupy.cuda.Device(self.device):
                self.idxs = cupy.concatenate((cupy.array(rows)[:, cupy.newaxis], cupy.array(cols)[:, cupy.newaxis]), axis=1).T

                self.vals = cupy.array([x for y in self.ATarea.data for x in y])
        else:
            raise NotImplementedError
            self.lu = sparse_lu((self.A.T @ self.A).tocsc())

    def __call__(self, pose: Mesh, project=False):
        
        S = self.deformation_gradient(pose)
        if project: # Project into the tangent space of the target 
            target_fv = self.get_fv(self.target)
            target_W = (target_fv[:, :, 1:] - target_fv[:, :, 0][:, :, np.newaxis]) # The basis are not formed by perpendicular unit vectors. TODO: Use igl.local_basis
            b0, b1, b_o = igl.local_basis(pose.vertices, pose.faces.astype(int))
            igl_W = np.concatenate([b0[:, :, np.newaxis], b1[:, :, np.newaxis]], axis=-1)

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
        
        # Some clean up
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        del ctx.solver

        return grad, None, None, None, None
