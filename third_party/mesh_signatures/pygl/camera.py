import numpy as np
from OpenGL.GL import *
import math

from . import transform


def perspective(fov, aspect, near, far):
    num = 1.0 / np.tan(fov * 0.5)
    idiff = 1.0 / (near - far)
    return np.array([
        [num/aspect, 0, 0, 0],
        [0, num, 0, 0],
        [0, 0, (far + near) * idiff, 2*far*near*idiff],
        [0, 0, -1, 0]
    ], dtype=np.float32)

class Camera(object):

    def __init__(self, screen_size, near=0.01, far=100.0, fov=45.0):
        self.__T_view = np.eye(4, dtype=np.float32)
        self.__T_proj = np.eye(4, dtype=np.float32)
        self.__near = near
        self.__far  = far
        self.__dirty = True
        self.__size = screen_size
        self.__fov = fov

    def __recompute_matrices(self):
        self.__T_proj = perspective(math.radians(self.fov), self.aspect_ratio, self.near, self.far)
        self.__dirty = False

    def look_at(self, target, up=[0, 1, 0]):
        self.__T_view = transform.look_at(self.position, target, up)

    def look_at(self, target, eye, up=[0, 1, 0]):
        self.__T_view = transform.look_at(eye, target, up)

    @property
    def position(self):
        return self.__T_view[:, 3]

    @property
    def aspect_ratio(self):
        return self.__size[0] / self.__size[1]

    @property
    def VP(self):
        if self.__dirty:
            self.__recompute_matrices()
        return self.__T_proj @ self.__T_view

    @property
    def P(self):
        if self.__dirty:
            self.__recompute_matrices()
        return self.__T_proj

    @property
    def V(self):
        return self.__T_view

    @V.setter
    def V(self, T):
        self.__T_view = T

    @property
    def screen_size(self):
        return self.__size

    @screen_size.setter
    def screen_size(self, x):
        self.__dirty = True
        self.__size = x

    @property
    def far(self):
        return self.__far

    @far.setter
    def far(self, x):
        self.__dirty = True
        self.__far = x

    @property
    def near(self):
        return self.__near

    @near.setter
    def near(self, x):
        self.__dirty = True
        self.__near = x

    @property
    def fov(self):
        return self.__fov
    
    @fov.setter
    def fov(self, x):
        self.__dirty = True
        self.__fov = x