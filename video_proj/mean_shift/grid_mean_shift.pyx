# cython: profile=True
""" -*- python -*- file
"""
cimport cython
cimport numpy as np
from video_proj.indexing cimport idx_type, clamp, flat_idx
from ..indexing import py_idx_type as pidx_t
import numpy as np

def classify_point(nfvec, labels, density):
    # nfvec is the feature vector normalized by the grid edge length
    # and grid origin offset -- in other words, it is a "fractional"
    # index coordinate

    # call the multi-linear interpolation of the coordinates nfvec
    # the band-limited mean, and call the multi-linear interpolation
    # of the density manifold as the point density
    pass

@cython.boundscheck(False)
def multilinear_interpolation(np.ndarray[np.float64_t, ndim=1] p, grid):
    # p is a "fractional index" -- i.e., it has been normalized
    # by the grid edge lengths and offset by the grid origin
    
    grid_domain_d = len(p)
    cdef np.ndarray[idx_type, ndim=1] dims = \
        np.array(grid.shape[:grid_domain_d], dtype=pidx_t)
    # grid has at least "grid_domain_d" dimensions. If grid represents
    # a vector valued function, then grid.ndim = grid_domain_d + 1
    cdef int grid_range_d
    grid_range_d = 1 if grid.ndim == grid_domain_d else grid.shape[-1]
    cdef np.ndarray[np.float64_t, ndim=2] f_grid = \
        grid.reshape(-1, grid_range_d)
    cdef np.ndarray[np.float64_t, ndim=1] r = np.empty((grid_range_d,), 'd')

    # these are the Nx2 interpolation nodes offsets
    cdef np.ndarray[idx_type, ndim=2] nodes = \
        np.empty((grid_domain_d, 2), dtype=pidx_t)
    # these are (1-\alpha, \alpha) pairs for each interp node
    cdef np.ndarray[np.float64_t, ndim=2] interp_coefs = \
        np.empty((grid_domain_d, 2), 'd')

    cdef double alpha
    cdef idx_type n0, n1
    cdef int d, n

    for d in range(grid_domain_d):
        n0 = clamp(p[d], 0, dims[d]-1)
        n1 = clamp(p[d]+1, 0, dims[d]-1)
        alpha = 1.0 - (p[d] - <double>n0)
        nodes[d,0] = n0
        nodes[d,1] = n1
        interp_coefs[d,0] = alpha
        interp_coefs[d,1] = 1.0 - alpha

    # now interpolate along each edge in the hypercube
    cdef int n_nodes = 2**grid_domain_d
    cdef np.ndarray[np.uint8_t, ndim=2] cube_corners = \
        np.unpackbits(
            np.arange(n_nodes, dtype='B')
            ).reshape(n_nodes,-1)
    # each row is length 8, but we only need grid_domain_d bits
    cdef int col_off = 8 - grid_domain_d
    cdef np.ndarray[idx_type, ndim=1] corner = \
        np.empty((grid_domain_d,), dtype=pidx_t)
    cdef idx_type f_idx
    for n in range(n_nodes):

        alpha = 1.0
        for d in range(grid_domain_d):
            r[d] = 0
            # cube_corners[n,d] either 0 or 1
            corner[d] = nodes[d, cube_corners[n,d+col_off]]
            alpha *= interp_coefs[d, cube_corners[n,d+col_off]]

        f_idx = flat_idx(
            <idx_type*>corner.data, grid_domain_d, <idx_type*>dims.data
            )
        for d in range(grid_range_d):
            r[d] += f_grid[f_idx,d] * alpha
    if grid_range_d == 1:
        return r[0]
    return r
