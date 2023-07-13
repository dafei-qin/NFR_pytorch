import logging as log
import numpy as np
from OpenGL.GL import *
import os
import time

from .base import GLObject
from .texture import TextureBase

def type_from_path(path):
    ext = path[path.rfind('.')+1:]
    ext_map = {
        'vs': GL_VERTEX_SHADER,
        'tcs': GL_TESS_CONTROL_SHADER,
        'tes': GL_TESS_EVALUATION_SHADER,
        'gs': GL_GEOMETRY_SHADER,
        'fs': GL_FRAGMENT_SHADER,
        'cs': GL_COMPUTE_SHADER
    }

    return ext_map.get(ext, None)

class Shader(GLObject):
    def __init__(self, *sources):
        super(Shader, self).__init__()
        self.__shaders = [
            (s, type_from_path(s), None) for s in sources if type_from_path(s) is not None
        ]

    def free(self):
        log.debug("Shader.free")
        if self.id != 0:
            glDeleteProgram(self.id)
            self._id = 0

    def __compile(self):
        shader_ids = []
        
        now = time.time()

        for (path, shader_type, _) in self.__shaders:
            shader_ids.append(Shader.__create_shader__(path, shader_type))


        if self.id != 0:
            glDeleteProgram(self.id)
        self._id = glCreateProgram()
        for shader_id in shader_ids:
            glAttachShader(self.id, shader_id)

        glLinkProgram(self.id)
        if not glGetProgramiv(self.id, GL_LINK_STATUS):
            log.error(glGetProgramInfoLog(self.id))
            raise RuntimeError('Shader linking failed')
        else:
            log.debug('Shader linked.')

        for shader_id in shader_ids:
            glDeleteShader(shader_id)

        self.__shaders = [(s, t, now) for (s, t, _) in self.__shaders]

    @staticmethod
    def __read_file__(path):
        with open(path, 'r') as f:
            data = f.read()
        return data

    @staticmethod
    def __create_shader__(path, shader_type):
        shader_code = Shader.__read_file__(path)
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shader_code)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            print(f"{glGetShaderInfoLog(shader)}")
            raise RuntimeError(f"[{path}]: Shader failed to compile!")
        else:
            log.debug(f"[{path}]: Shader compiled!")
        return shader

    @property
    def up_to_date(self):
        if self.id == 0:
            return False
        else:
            now = time.time()
            shaders_updated = [ lu is None or lu < os.path.getmtime(s) for (s, _, lu) in self.__shaders ]
            return not True in shaders_updated

    def use(self, **kwargs):
        if not self.up_to_date:
            self.__compile()
        glUseProgram(self.id)
        self.set_uniforms(**kwargs)

    def set_float(self, name, value):
        glUniform1f(glGetUniformLocation(self.id, name), value)

    def set_int(self, name, value):
        glUniform1i(glGetUniformLocation(self.id, name), value)

    def set_vector(self, name, value):
        vector = np.array(value, dtype=np.float32).flatten()
        if vector.size == 1:
            glUniform1f(glGetUniformLocation(self.id, name), *vector)
        elif vector.size == 2:
            glUniform2f(glGetUniformLocation(self.id, name), *vector)
        elif vector.size == 3:
            glUniform3f(glGetUniformLocation(self.id, name), *vector)
        elif vector.size == 4:
            glUniform4f(glGetUniformLocation(self.id, name), *vector)
        else:
            raise RuntimeError("Invalid value")

    def set_matrix(self, name, value):
        assert isinstance(value, np.ndarray), 'Matrix must be a numpy array'
        if value.shape == (4, 4):
            glUniformMatrix4fv(glGetUniformLocation(self.id, name), 1, GL_TRUE, value)
        elif value.shape == (3, 3):
            glUniformMatrix3fv(glGetUniformLocation(self.id, name), 1, GL_TRUE, value)
        else:
            raise RuntimeError(f"Matrix shape {value.shape} not supported")

    def set_uniforms(self, **kwargs):
        """Set multiple uniforms for shader
        The types are guessed from data type of arguments
        """
        next_texture_slot = 0
        for name, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                value = np.array(value, dtype=np.float32)

            if isinstance(value, float):
                glUniform1f(glGetUniformLocation(self.id, name), value)
            elif isinstance(value, (int, bool)):
                glUniform1i(glGetUniformLocation(self.id, name), value)
            elif isinstance(value, np.ndarray):
                if value.dtype != np.float32:
                    print(f"Set uniform {name}: Unsupported format {value.dtype}. Going to convert")
                    value = value.astype(np.float32)
                if value.shape == (2, 2):
                    glUniformMatrix2fv(glGetUniformLocation(self.id, name), 1, GL_TRUE, value)
                elif value.shape == (3, 3):
                    glUniformMatrix3fv(glGetUniformLocation(self.id, name), 1, GL_TRUE, value)
                elif value.shape == (4, 4):
                    glUniformMatrix4fv(glGetUniformLocation(self.id, name), 1, GL_TRUE, value)
                else:
                    flat_value = value.flatten()
                    if value.size == 1:
                        glUniform1f(glGetUniformLocation(self.id, name), *value)
                    elif value.size == 2:
                        glUniform2f(glGetUniformLocation(self.id, name), *value)
                    elif value.size == 3:
                        glUniform3f(glGetUniformLocation(self.id, name), *value)
                    elif value.size == 4:
                        glUniform4f(glGetUniformLocation(self.id, name), *value)
                    else:
                        log.error(f"Unsupported matrix shape {value.shape}")
            elif isinstance(value, TextureBase):
                value.bind(next_texture_slot)
                glUniform1i(glGetUniformLocation(self.id, name), next_texture_slot)
                next_texture_slot += 1