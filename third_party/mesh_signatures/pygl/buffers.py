import numpy as np
from OpenGL.GL import *

import ctypes
from .base import GLObject, Context

class BufferObject(GLObject):

    def __init__(self, data, target):
        super(BufferObject, self).__init__(glGenBuffers, glDeleteBuffers)
        self.__target = target
        if data is not None:
            self.resize(data.nbytes)
            self.update(data)

    def __enter__(self):
        glBindBuffer(self.__target, self.id)

    def __exit__(self, type, value, traceback):
        glBindBuffer(self.__target, 0)

    def bind(self, index=-1):
        glBindBuffer(self.__target, self.id)
        if index >= 0:
            glBindBufferBase(self.__target, index, self.id)

    def unbind(self):
        glBindBuffer(self.__target, 0)

    def update(self, data, offset=0, nbytes=None):
        nbytes = data.nbytes if nbytes is None else nbytes
        self.bind()
        glBufferSubData(self.__target, offset, data.nbytes, data)
        self.unbind()

    def resize(self, nbytes):
        self.bind()
        glBufferData(self.__target, nbytes, None, GL_STATIC_DRAW) 
        self.unbind()

    @property
    def target(self):
        return self.__target
    
def create_vbo(data=None):
    return BufferObject(data, GL_ARRAY_BUFFER)

def create_index_buffer(data=None):
    return BufferObject(data, GL_ELEMENT_ARRAY_BUFFER)

def create_ssbo(data=None):
    return BufferObject(data, GL_SHADER_STORAGE_BUFFER)

class VertexArrayObject(GLObject):
    def __init__(self):
        super(VertexArrayObject, self).__init__(glGenVertexArrays, glDeleteVertexArrays)

    def __enter__(self):
        glBindVertexArray(self.id)
        return self

    def __exit__(self, type, value, traceback):
        glBindVertexArray(0)

    def bind(self):
        glBindVertexArray(self.id)

    def unbind(Self):
        glBindVertexArray(0)

    def setIndexBuffer(self, ebo):
        if isinstance(ebo, BufferObject):
            assert ebo.target == GL_ELEMENT_ARRAY_BUFFER
            self.bind()
            ebo.bind()
            self.unbind()
            ebo.unbind
        else:
            raise('Invalid EBO type')

    def setVertexAttributes(self, vbo, stride, attribs):
        if not isinstance(vbo, BufferObject):
            raise('Ivalid VBO type')

        self.bind()
        vbo.bind()
        for (idx, dim, attr_type, normalized, rel_offset) in attribs:
            glEnableVertexAttribArray(idx)
            vertexAttrib = glVertexAttribIPointer if attr_type in [GL_INT, GL_UNSIGNED_INT] else glVertexAttribPointer
            vertexAttrib(idx, dim, attr_type, normalized, stride, ctypes.c_void_p(rel_offset))
            glEnableVertexAttribArray(idx)
        self.unbind()
        vbo.unbind()

    @classmethod
    def bind_dummy(cls):
        ctx = Context.current()
        assert ctx is not None, "No current context"
        glBindVertexArray(ctx.dummy_vao)

