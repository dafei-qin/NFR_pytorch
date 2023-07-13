import numpy as np
import math

def normalize(v):
    m = np.linalg.norm(v)
    if m == 0:
        return v
    return v / m

def translate(xyz):
    x, y, z = xyz[:3]
    return np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]], dtype=np.float32)

def scale(xyz):
    x, y, z = xyz[:3]
    return np.matrix([[x,0,0,0],
                      [0,y,0,0],
                      [0,0,z,0],
                      [0,0,0,1]], dtype=np.float32)

def rotate(a, xyz):
    x, y, z = normalize(xyz)
    a = math.radians(a)
    s = math.sin(a)
    c = math.cos(a)
    nc = 1 - c
    return np.matrix([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]], dtype=np.float32)

def rotate_x(a):
    a = math.radians(a)
    s = math.sin(a)
    c = math.cos(a)
    return np.matrix([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]],  dtype=np.float32)

def rotate_y(a):
    a = math.radians(a)
    s = math.sin(a)
    c = math.cos(a)
    return np.matrix([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]], dtype=np.float32)

def rotate_z(a):
    a = math.radians(a)
    s = math.sin(a)
    c = math.cos(a)
    return np.matrix([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=np.float32)

def look_at(eye, center, up=[0, 1, 0]):
    eye = np.array(eye, dtype=np.float32).flatten()[:3]
    center = np.array(center, dtype=np.float32).flatten()[:3]
    up = np.array(up, dtype=np.float32).flatten()[:3]

    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.matrix(np.identity(4), dtype=np.float32)
    M[:3,:3] = np.vstack([s, u, -f])
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] =  np.dot(f, eye)
    return M