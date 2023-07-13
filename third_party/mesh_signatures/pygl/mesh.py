import numpy as np
from OpenGL.GL import *

import trimesh

from .base import Context
from . import buffers

import math


class Mesh(object):

    @classmethod
    def load(cls, path, recenter=True, rescale=False):
        mesh = trimesh.load(path, force='mesh')
        if recenter:
            mesh.vertices -= mesh.vertices.mean(axis=-2, keepdims=True)
        if rescale:
            radius = np.linalg.norm(mesh.vertices, axis=-1).max()
            mesh.vertices = mesh.vertices / radius
        trimesh.repair.fix_normals(mesh)
        return cls(mesh)

    @classmethod
    def uv_sphere(cls, radius : float = 1.0):
        mesh = trimesh.creation.uv_sphere(radius)
        return cls(mesh)

    @classmethod
    def tetraeder(cls, edge_length : float = 1.0):
        s = edge_length * 0.5
        vertices = s * np.array([
            [ 1,  0, -1.0/math.sqrt(2)],
            [-1,  0, -1.0/math.sqrt(2)],
            [ 0,  1,  1.0/math.sqrt(2)],
            [ 0, -1,  1.0/math.sqrt(2)]
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]
        ])
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False)
        mesh.fix_normals()
        return cls(mesh)

    @classmethod
    def box(cls, size=1.0, inward_normals = False):
        mesh = trimesh.creation.box()
        mesh.vertices *= size
        mesh.face_normals = -1.0 * mesh.face_normals
        return cls(mesh)

    def __init__(self, mesh : trimesh.Trimesh):
        self.mesh = mesh
        if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'vertex_colors'):
            self._vertex_colors = self.mesh.visual.vertex_colors.astype(np.float32)[:, :3] / 255.0
        else:
            self._vertex_colors = np.ones_like(self.mesh.vertices)
        self._vaos = {}
        self._buffers = {}

    def render(self, shader, **kwargs):
        # check if we have a vao in the current contex
        ctx = Context.current()
        if not ctx in self._vaos:
            # Create the vao for this context
            compact_data = np.concatenate((self.mesh.vertices, self.mesh.vertex_normals, self._vertex_colors), axis=-1).astype(np.float32)
            indices = self.mesh.faces.astype(np.uint32)

            vbo = buffers.create_vbo(compact_data)
            ebo = buffers.create_index_buffer(indices)
            vao = buffers.VertexArrayObject()
            vao.setVertexAttributes(vbo, compact_data.shape[-1] * compact_data.itemsize, [
                (0, 3, GL_FLOAT, GL_FALSE, 0) ,
                (1, 3, GL_FLOAT, GL_FALSE, 3 * compact_data.itemsize),
                (2, 3, GL_FLOAT, GL_FALSE, 6 * compact_data.itemsize)
                ])
            vao.setIndexBuffer(ebo)
            self._buffers[ctx] = (vbo, ebo)
            self._vaos[ctx] = (vao, indices.size)

        vao, num_indices = self._vaos[ctx]
        vao.bind()
        shader.use(**kwargs)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        vao.unbind()

    @property
    def vertex_colors(self):
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, colors):
        colors = np.array(colors, dtype=np.float32)
        self._vertex_colors = colors
        
        assert colors.shape == self.mesh.vertices.shape, f"Invalid color shape, expected {self.mesh.vertices.shape} but got {colors.shape}"

        compact_data = np.concatenate((self.mesh.vertices, self.mesh.vertex_normals, self._vertex_colors), axis=-1).astype(np.float32)
        for _, (vbo, _) in self._buffers.items():
            vbo.update(compact_data)

    @property
    def n_vertices(self):
        return len(self.mesh.vertices)

    @property
    def vertices(self):
        return self.mesh.vertices

    def vertex(self, id : int):
        return self.mesh.vertices[id]