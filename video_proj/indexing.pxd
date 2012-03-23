""" -*- python -*- file
"""
cimport cython
cimport numpy as np

ctypedef np.int64_t idx_type

cdef inline void multi_idx(
     idx_type flat_idx, idx_type *dims, int nd, idx_type *idx
     )

cdef inline idx_type flat_idx(idx_type *multi_idx, int nd, idx_type *dims)

cdef np.ndarray[idx_type, ndim=1] flatten_idc(
    np.ndarray[idx_type, ndim=2] idc, 
    np.ndarray[idx_type, ndim=1] dims
    )

cdef inline idx_type clamp(double n, idx_type mn, idx_type mx)

cdef inline int oob(idx_type i, idx_type N)
