import os

import numpy as np
from OpenGL.GL import *

from enum import IntEnum, Enum

from .base import GLObject

import imageio

class twrap(IntEnum):
    clamp = GL_CLAMP_TO_EDGE
    border = GL_CLAMP_TO_BORDER
    repeat = GL_REPEAT
    mclamp = GL_MIRROR_CLAMP_TO_EDGE
    mrepeat = GL_MIRRORED_REPEAT
    none = 0

class tfilter(IntEnum):
    linear = GL_LINEAR
    nearest = GL_NEAREST
    linear_mip_nearest = GL_LINEAR_MIPMAP_NEAREST
    linear_mip_linear = GL_LINEAR_MIPMAP_LINEAR
    nearest_mip_linear = GL_NEAREST_MIPMAP_LINEAR
    none = 0

class tformat(IntEnum):
    red = GL_RED
    rg = GL_RG
    rgb = GL_RGB
    bgr = GL_BGR
    rgba = GL_RGBA
    bgra = GL_BGRA
    depth_stencil = GL_DEPTH_STENCIL
    depth = GL_DEPTH
    default = 0

    @staticmethod
    def from_array(a : np.ndarray):
        channels = a.shape[2]
        if channels == 1:
            return tformat.red
        elif channels == 2:
            return tformat.rg
        elif channels == 3:
            return tformat.rgb
        elif channels == 4:
            return tformat.rgba
        else:
            raise ValueError("Invalid input array, last dimension must be in [1, 2, 3, 4]")

class ttype(IntEnum):
    float32 = GL_FLOAT
    uint8 = GL_UNSIGNED_BYTE,
    uint24_8 = GL_UNSIGNED_INT_24_8

    @classmethod
    def from_array(cls, a : np.ndarray):
        if a.dtype == np.uint8:
            return cls.uint8
        elif a.dtype == np.float32:
            return cls.float32
        else:
            return None

def get_sized_format(fmt, tp):
    assert isinstance(fmt, tformat) and isinstance(tp, ttype)
    mapping = {
        ttype.float32: {
            tformat.rgb: GL_RGB32F,
            tformat.bgr: GL_RGB32F,
            tformat.red: GL_R32F,
            tformat.rg: GL_RG32F,
            tformat.rgba: GL_RGBA32F,
            tformat.bgra: GL_RGBA32F,
            tformat.depth: GL_DEPTH_COMPONENT32F
        },
        ttype.uint8: {
            tformat.red: GL_R8,
            tformat.rg: GL_RG8,
            tformat.rgb: GL_RGB8,
            tformat.bgr: GL_RGB8,
            tformat.rgba: GL_RGBA8,
            tformat.bgra: GL_RGBA8,
        },
        ttype.uint24_8: {
            tformat.depth: GL_DEPTH24_STENCIL8
        }
    }
    return mapping[tp][fmt]

def get_channels(fmt):
    mapping = {
        tformat.red: 1,
        tformat.rg: 2,
        tformat.rgb: 3,
        tformat.bgr: 3,
        tformat.rgba: 4,
        tformat.bgra: 4,
        tformat.depth_stencil: 1,
        tformat.depth: 1
    }
    return mapping[fmt]

class TextureBase(GLObject):
    def __init__(self, size, tex_type, fmt, tp, flt, wrap):
        super(TextureBase, self).__init__(glGenTextures, glDeleteTextures)
        self._size = size
        self._ttype = tp
        self._tformat = fmt
        self._filter = flt
        self._wrap = wrap
        self._type = tex_type
        
    @property
    def sized_format(self):
        return get_sized_format(self._tformat, self._ttype)

    @property
    def format(self):
        return self._tformat

    @property
    def ttype(self):
        return self._ttype

    @property
    def filter(self):
        return self._filter

    @property
    def size(self):
        return self._size
    
    @property
    def channels(self):
        return get_channels(self._tformat)

    @property
    def cols(self):
        return self._size[0]

    @property
    def rows(self):
        return self._size[1] if len(self.size) > 1 else None

    @property
    def depth(self):
        return self._size[2] if len(self.size) > 2 else None

    @property
    def wrapping(self):
        return self._wrap

    def bind(self, slot=-1):
        if slot >= 0:
            glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(self._type, self.id)

    def unbind(self):
        glBindTexture(self._type, 0)

class Texture2D(TextureBase):

    @classmethod
    def load(cls, path, tfilter=tfilter.linear, wrap=twrap.clamp, build_mipmaps=False):
        ext = os.path.splitext(path)[-1]
        if ext == '.hdr':
            imageio.plugins.freeimage.download() # TODO only once?
        data = imageio.imread(path)
        return cls.from_numpy(data, tfilter, wrap, build_mipmaps)

    @classmethod
    def from_numpy(cls, data, tfilter=tfilter.linear, wrap=twrap.clamp, build_mipmaps=False):
        assert 2 <= data.ndim <= 3, f"Invalid number of dimensionst must be in [2, 3] but got {data.ndim}"
        if data.ndim == 2:
            data = data[..., np.newaxis]
        
        tp = ttype.from_array(data)
        tex = cls(
            *data.shape[:2],
            tformat=tformat.from_array(data),
            tp=ttype.from_array(data),
            tfilter=tfilter,
            wrap=wrap,
            build_mipmaps=build_mipmaps)

        tex.update(data)
        return tex

    def __init__(self, 
        width, height,
        tformat = tformat.rgba,
        wrap = (twrap.clamp, twrap.clamp),
        tp = ttype.uint8,
        tfilter = tfilter.linear,
        build_mipmaps = False):

        super(Texture2D, self).__init__(
            size=(width, height),
            tex_type=GL_TEXTURE_2D,
            fmt=tformat,
            tp=tp,
            wrap=wrap,
            flt=tfilter)

        self.with_mipmaps = build_mipmaps

        if not isinstance(self._wrap, (list, tuple)):
            self._wrap = (self._wrap, self._wrap)

        # Create the texture
        self.resize(width, height)
        self.set_wrapping(*self._wrap)
        self.set_filter(tfilter)

    def update(self, data, fmt = tformat.default, resize=True):
        assert isinstance(data, np.ndarray)

        if resize and (data.shape[0] != self.rows or data.shape[1] != self.cols):
            self.resize(data.shape[1], data.shape[0])
        elif data.shape[:2] != self._size:
            data = data[0:self.rows, 0:self.cols, :]
        
        self.bind()
        fmt = self.format if fmt is tformat.default else fmt

        assert data.shape[-1] == get_channels(fmt)
        
        old_align = glGetIntegerv(GL_UNPACK_ALIGNMENT)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, self.cols)
        glTexSubImage2D(self._type, 0, 0, 0, self.cols, self.rows, fmt, self.ttype, data)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, old_align)
        
        if self.with_mipmaps:
            glGenerateMipmap(self._type)

        self.unbind()

    def resize(self, cols, rows):
        glBindTexture(self._type, self.id)
        glTexImage2D(self._type, 0, self.sized_format, cols, rows, 0, self.format, self.ttype, None)
        self._size = (cols, rows)
        glBindTexture(self._type, 0)

    def set_wrapping(self, s : twrap, t: twrap):
        glBindTexture(self._type, self.id)
        s0, t0 = self._wrap
        if s != twrap.none:
            glTexParameteri(self._type, GL_TEXTURE_WRAP_S, s)
            s0 = s
        if t != twrap.none:
            glTexParameteri(self._type, GL_TEXTURE_WRAP_T, t)
            t0 = t
        self._wrap = (s0, t0)
        glBindTexture(self._type, 0)

    def set_filter(self, min_filter : tfilter, mag_filter : tfilter = None):
        if mag_filter is None:
            mag_filter = min_filter
        glBindTexture(self._type, self.id)
        glTexParameteri(self._type, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(self._type, GL_TEXTURE_MAG_FILTER, mag_filter)
        glBindTexture(self._type, 0)
        self._filter = (min_filter, mag_filter)
