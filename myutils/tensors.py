import numpy as np

def tensor_3d_to_4d(v):
    return np.concatenate([v, np.ones((*v.shape[:-1], 1))], axis=-1)

def tensor_4d_to_3d(v):
    return (v / v[..., 3:4])[..., :3]

def transform(v, R):
    return v @ R[:3, :3] + R[:3, 3]
    # return tensor_4d_to_3d(tensor_3d_to_4d(v) @ R)

def construct_matrix(scale, shift):
    T = np.eye(4)
    T[:3, :3] = np.diag(scale)
    T[:3, 3] = shift
    return T
