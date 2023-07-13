import numpy as np
import scipy
import trimesh
import logging

import os
import math
import warnings
warnings.filterwarnings("error")

import mesh_signatures.laplace as laplace

class SignatureExtractor(object):
    def __init__(self, mesh=None, n_basis=-1, approx='cotangens', path=None):
        self._initialized = False
        if path is not None:
            self.load(path)
        elif n_basis > 0 and mesh is not None:
            self.initialize(mesh, n_basis, approx)

    def initialize(self, mesh : trimesh.Trimesh, n : int, approx='cotangens'):
        """Initializes the SignatureExtractor by computing eigenvectors and eigenvalues
        of the discrete laplace operator. 

        Args:
            mesh (trimesh.Trimesh): Mesh to extract signatures from
            n (int): Number of eigenvalues and eigenvectors to compute.
            approx (str, optional): Laplace operator approximation to use. 
                                    Must be in ['beltrami', 'cotangens', 'mesh', 'fem']. Defaults to 'cotangens'.
        """
        self.W, self.M = laplace.get_laplace_operator_approximation(mesh, approx)
        self.n_basis = min(len(mesh.vertices) - 1, n)

        sigma = -0.01
        try:
            from sksparse.cholmod import cholesky
            use_cholmod = True
        except ImportError as e:
            # logging.warn(
            #     "Package scikit-sparse not found (Cholesky decomp). "
            #     "This leads to less efficient eigen decomposition.")
            use_cholmod = False

        if use_cholmod:
            chol = cholesky((self.W - sigma * self.M).tocsc())
            op_inv = scipy.sparse.linalg.LinearOperator(
                            matvec=chol, 
                            shape=self.W.shape,
                            dtype=self.W.dtype)
        else:
            lu = scipy.sparse.linalg.splu((self.W - sigma * self.M).tocsc())
            op_inv = scipy.sparse.linalg.LinearOperator(
                            matvec=lu.solve, 
                            shape=self.W.shape,
                            dtype=self.W.dtype)
        

        self.evals, self.evecs = scipy.sparse.linalg.eigsh(self.W, 
                                                           self.n_basis, 
                                                           self.M, 
                                                           sigma=sigma,
                                                           OPinv=op_inv)

        self._initialized = True

    def heat_signatures(self, dim : int, return_times=False, times=None):
        """Compute the heat signature for all vertices

        Args:
            dim (int): Dimensionality (timesteps) of the signature.
            return_times (bool, optional): If True the function returns a tuple (signature, timesteps) 
                                           otherwise only the signature is returned. Defaults to False.
            times (arraylike, optional): Timesteps used for signature computation.
                                         If None the times are spaced logarithmically. Defaults to None.

        Note:
            This signature is based on 'A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion'
            by Jian Sun et al (http://www.lix.polytechnique.fr/~maks/papers/hks.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """
        assert self._initialized, "Signature extractor was not initialized"

        if times is None:
            tmax_idx = np.argmax(self.evals > 0)
            tmin  = 4 * math.log(10) / self.evals[-1]
            tmax  = 4 * math.log(10) / self.evals[tmax_idx]
            times = np.geomspace(tmin, tmax, dim)
        else:
            times = np.array(times).flatten()
            assert len(times) == dim, f"Requested feature dimension and time steps array do not match: {dim} and {len(times)}"


        phi2       = np.square(self.evecs[:, tmax_idx:])
        exp        = np.exp(-self.evals[tmax_idx:, None]*times[None])
        s          = np.sum(phi2[..., None]*exp[None], axis=1)
        heat_trace = np.sum(exp, axis=0)
        try:
            s          = s / heat_trace[None] 
        except RuntimeWarning:
            raise ValueError


        if return_times:
            return s, times
        else:
            return s

    def wave_signatures(self, dim : int, return_energies=False, energies=None):
        """Compute the wave signature for all vertices

        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            return_energies (bool, optional): If True the function returns a tuple (signature, energies) 
                                              otherwise only the signature is returned. Defaults to False.
            energies (arraylike, optional): Energie spectra used for signature computation.
                                            If None the energy is linearly spaced. Defaults to None.

        Note:
            This signature is based on 'The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis'
            by Mathieu Aubry et al (https://vision.informatik.tu-muenchen.de/_media/spezial/bib/aubry-et-al-4dmod11.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """

        assert self._initialized, "Signature extractor was not initialized"
        emin_idx = np.argmax(self.evals > 0)

        if energies is None:
            emin = math.log(self.evals[emin_idx])
            emax = math.log(self.evals[-1]) / 1.02
            energies = np.linspace(emin, emax, dim)
        else:
            energies = np.array(energies).flatten()
            assert len(energies) == dim, f"Requested featrue dimension and energies array do not match: {dim} and {len(energies)}"

        sigma        = 7.0 * (energies[-1] - energies[0]) / dim
        # phi2         = np.square(self.evecs[:, 1:])
        phi2         = np.square(self.evecs[:, emin_idx:])
        exp          = np.exp(-np.square(energies[None] - np.log(self.evals[emin_idx:, None])) / (2.0 * sigma * sigma))
        s            = np.sum(phi2[..., None]*exp[None], axis=1)
        energy_trace = np.sum(exp, axis=0)
        s            = s / energy_trace[None] 
            
        if return_energies:
            return s, energies
        else:
            return s

    def signatures(self, dim : int, kernel : str, return_x_ticks=False, x_ticks=None):
        """Computes a signature for each vertex

        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            kernel (str): Feature kernel used must be in ['heat', 'wave'].
            return_x_ticks (bool, optional): If True the function returns a tuple (signature, x_ticks) 
                                            otherwise only the signature is returned. Defaults to False.
            x_ticks (arraylike, optional): Variable used for the feature dimension. Defaults to None.

        Returns:
            Returns an array of shape (#vertices, dim) containing the mesh signatures of every vertex.
            If return_x_ticks is True this function returns a tuple (signature, x_ticks).
        """
        assert kernel in ['heat', 'wave'], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"


        if kernel == 'heat':
            return self.heat_signatures(dim, return_x_ticks, x_ticks)
        else:
            return self.wave_signatures(dim, return_x_ticks, x_ticks)

    def heat_distances(self, query, dim : int, return_signature=False, times=None, cutoff=1.0):
        """Compute distances of all vertices to vertices in query based on heat signature

        Note:
            We use the L2 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool, optional): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike, optional): Time steps used for signature computation. Defaults to None 
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the heat signature distance 
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).  
        """

        e = max(0, int(cutoff*dim)) + 1
        a = self.heat_signatures(dim, times=times)[..., :e]
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        dist_arr = np.squeeze(dist_arr)
        if return_signature:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr

    def wave_distance(self, query, dim :int, return_signatures=False, energies=None, cutoff=1.0):
        """Compute distances of all vertices to vertices in query based on the wave signature

        Note:
            We use the L1 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike): Time steps used for signature computation. Defaults to None 
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the wave signature distance 
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).  
        """
        e = max(0, int(dim * cutoff)) + 1
        a = self.wave_signatures(dim, energies=energies)[..., :e]
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sum(np.abs(diff), axis=-1)
        dist_arr = np.squeeze(dist_arr)
        if return_signatures:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr
            
    def feature_distance(self, query, dim : int, kernel : str, return_signatures=False, x_ticks=None, cutoff=1.0):
        """Compute distances of all vertices to vertices in query based on a mesh signature

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            kernel (str): Signature type to use. Must be in ['heat', 'wave']
            return_signature (bool): If True the function returns a tuple (distances, signatures). Defaults to False
            x_ticks (None, arraylike): x_ticks used for signature computation. Defaults to None 
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the signature distance 
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).  
        """
        assert kernel in ['heat', 'wave'], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"

        if kernel == 'heat':
            return self.heat_distances(query, dim, return_signatures, x_ticks, cutoff)
        else:
            return self.wave_distance(query, dim, return_signatures, x_ticks, cutoff)

    def save(self, path : str):
        """Save the laplace spectrum (eigenvalues and eigenvectors) to a .npz file

        Args:
            path (str): Output filename.
        """
        assert self._initialized, "Signature extractor was not initialized"

        folder, _ = os.path.split(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(path, evals=self.evals, evecs=self.evecs)

    def load(self, path):
        """Initialize by loading the laplace spectrum (eigenvalues and eigenvectors) from a .npz file

        Args:
            path (str): Input filename.
        """
        assert os.path.exists(path), "File not found"

        data = np.load(path)
        assert 'evals' in data and 'evecs' in data, "Error loading file"
        
        self.evals = data['evals']
        self.evecs = data['evecs']
        self.n_basis = len(self.evals)
        self._initialized = True

    @property
    def stiffness(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.W
    
    @property
    def mass(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.M

    @property
    def spectrum(self):
        if not self._initialized:
            raise RuntimeError("Extractor not initialized")
        return self.evals
        