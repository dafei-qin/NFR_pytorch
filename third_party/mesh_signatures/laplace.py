import numpy as np
import scipy
import logging
import trimesh

import math

def build_mass_matrix(mesh : trimesh.Trimesh):
    """Build the sparse diagonal mass matrix for a given mesh

    Args:
        mesh (trimesh.Trimesh): Mesh to use.

    Returns:
        A sparse diagonal matrix of size (#vertices, #vertices).
    """
    areas = np.zeros(shape=(len(mesh.vertices)))
    for face, area in zip(mesh.faces, mesh.area_faces):
        areas[face] += area / 3.0

    return scipy.sparse.diags(areas)

def approx_methods() -> list[str]:
    """Available laplace approximation types."""
    try:
        import lapy
        return [ 'beltrami', 'cotangens', 'mesh', 'fem' ]
    except ImportError:
        # logging.warn(
        #     "fem appxoimation only works if lapy is installed. "
        #     "You can find lapy on github: https://github.com/Deep-MI/LaPy.\n"
        #     "Install it with pip:\n"
        #      "pip3 install --user git+https://github.com/Deep-MI/LaPy.git#egg=lapy")
        return [ 'beltrami', 'contangens', 'mesh' ]        
    
def build_laplace_betrami_matrix(mesh : trimesh.Trimesh):
    """Build the sparse laplace beltrami matrix of the given mesh M=(V, E).
    This is a positive semidefinite matrix C:

           -1         if (i, j) in E
    C_ij =  deg(V(i)) if i == j
            0         otherwise
    
    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C
    """
    n = len(mesh.vertices)
    IJ = np.concatenate([
        mesh.edges,
        [[i, i] for i in range(n)]
    ], axis=0)
    V  = np.concatenate([
        [-1 for _ in range(len(mesh.edges))],
        mesh.vertex_degree
    ], axis= 0)


    A = scipy.sparse.coo_matrix((V, (IJ[..., 0], IJ[..., 1])), shape=(n, n), dtype=np.float64)
    return A

def build_cotangens_matrix(mesh : trimesh.Trimesh):
    """Build the sparse cotangens weight matrix of the given mesh M=(V, E).
    This is a positive semidefinite matrix C:

           -0.5 * (tan(a) + tan(b))  if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)   if i == j
            0                        otherwise
    
    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    ij = mesh.face_adjacency_edges
    ab = mesh.face_adjacency_unshared

    uv = mesh.vertices[ij]
    lr = mesh.vertices[ab]


    def cotan(v1, v2):
        return np.sum(v1*v2) / np.linalg.norm(np.cross(v1, v2), axis=-1)

    ca = cotan(lr[:, 0] - uv[:, 0], lr[:, 0] - uv[:, 1])
    cb = cotan(lr[:, 1] - uv[:, 0], lr[:, 1] - uv[:, 1])

    wij = np.maximum(0.5 * (ca + cb), 0.0)

    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(ij):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]
    
    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A

def build_mesh_laplace_matrix(mesh : trimesh.Trimesh):
    """Build the sparse mesh laplacian matrix of the given mesh M=(V, E).
    This is a positive semidefinite matrix C:

           -1/(4pi*h^2) * e^(-||vi-vj||^2/(4h)) if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)              if i == j
            0                                   otherwise
    here h is the average edge length

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    h = np.mean(mesh.edges_unique_length)
    a = 1.0 / (4 * math.pi * h*h)
    wij = a * np.exp(-mesh.edges_unique_length**2/(4.0*h))
    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(mesh.edges_unique):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]
    
    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A

def build_laplace_approximation_matrix(mesh : trimesh.Trimesh, approx = 'beltrami'):
    """Build the sparse mesh laplacian matrix of the given mesh M=(V, E).
    This is a positive semidefinite matrix C:

           w_ij                    if (i, j) in E
    C_ij = -sum_{j in N(i)} (w_ij) if i == j
            0                      otherwise
    here h is the average edge length

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C
        approx (str): Approximation type to use, must be in ['beltrami', 'cotangens', 'mesh']. Defaults to 'beltrami'.

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    
    assert approx in approx_methods(), f"Invalid method '{approx}', must be in {approx_methods()}"

    if approx == 'beltrami':
        return build_laplace_betrami_matrix(mesh)
    elif approx == 'cotangens':
        return build_cotangens_matrix(mesh)
    else:
        return build_mesh_laplace_matrix(mesh)

def get_laplace_operator_approximation(mesh : trimesh.Trimesh, 
                                       approx = 'cotangens') -> tuple[np.ndarray, np.ndarray]:
    """Computes a discrete approximation of the laplace-beltrami operator on
    a given mesh. The approximation is given by a Mass matrix A and a weight or stiffness matrix W

    Args:
        mesh (trimesh.Trimesh): Input mesh
        approx (str, optional): Laplace approximation to use See laplace.approx_methods() for possible values. Defaults to 'cotangens'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of sparse matrices (Stiffness, Mass)
    """
    if approx not in approx_methods():
        raise ValueError(
            f"Invalid approximation method must be one of {approx_methods}."
            f"Got {approx}")
    
    if approx == 'fem':
        import lapy
        T = lapy.TriaMesh(mesh.vertices, mesh.faces)
        solver = lapy.Solver(T)
        return solver.stiffness, solver.mass
    else:
        W = build_laplace_approximation_matrix(mesh, approx)
        M = build_mass_matrix(mesh)
        return W, M
