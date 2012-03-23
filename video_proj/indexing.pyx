""" -*- python -*- file
"""
from __future__ import division
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
import numpy as np

py_idx_type = np.int64

cdef inline void multi_idx(
     idx_type flat_idx, idx_type *dims, int nd, idx_type *idx
     ):
    cdef int k
    for k in range(nd-1, -1, -1):
        idx[k] = flat_idx % dims[k]
        flat_idx -= idx[k]
        flat_idx = (flat_idx // dims[k])
    return

@cython.boundscheck(False)
cdef inline idx_type flat_idx(idx_type *multi_idx, int nd, idx_type *dims):
    cdef int k, stride = 1
    cdef idx_type f_idx = 0	
    for k in range(nd-1, -1, -1):
        f_idx += multi_idx[k] * stride
        stride *= dims[k]
    return f_idx

@cython.boundscheck(False)
cdef np.ndarray[idx_type, ndim=1] flatten_idc(
    np.ndarray[idx_type, ndim=2] idc, 
    np.ndarray[idx_type, ndim=1] dims
    ):
    """
    Flatten an array of indices -- indices may either be columns
    or rows of idc array
    """
    cdef int nd = len(dims)
    idx_dim = 0 if idc.shape[0] == nd else 1
    if not idx_dim:
        idc = idc.transpose()
    cdef int cstride = idc.strides[1]
    cdef int rstride = idc.strides[0]//sizeof(idx_type)
    cdef idx_type *idx_buf
    if cstride != sizeof(idx_type):
        free_later = True
        idx_buf = <idx_type *> malloc(nd*sizeof(idx_type))
    else:
        free_later = False
    cdef int j, k, n_idx = idc.shape[0]
    cdef np.ndarray[idx_type, ndim=1] f_idx = \
        np.empty((n_idx,), dtype=py_idx_type)
    for j in range(n_idx):
        if free_later:
            for k in range(nd):
                idx_buf[k] = idc[j,k]
        else:
            idx_buf = <idx_type*> idc.data
            idx_buf += (j*rstride)
        f_idx[j] = flat_idx(idx_buf, nd, <idx_type*> dims.data)
    if free_later:
        free(idx_buf)
    return f_idx

def flatten_idc_p(np.ndarray[idx_type, ndim=2] idx, tuple dims):
    cdef np.ndarray[idx_type, ndim=1] a_dims = \
        np.array(dims, dtype=py_idx_type)
    return flatten_idc(idx, a_dims)

# may make a iterable version of this for an array of flat indices
def multi_idx_p(int flat_idx, tuple dims):
    cdef int nd = len(dims)
    cdef np.ndarray[idx_type, ndim=1] idx = \
        np.empty((nd,), dtype=py_idx_type)
    cdef np.ndarray[idx_type, ndim=1] a_dims = \
        np.array(dims, dtype=py_idx_type)
    multi_idx(
        <idx_type> flat_idx, <idx_type*> a_dims.data, nd,
        <idx_type*> idx.data
        )
    return idx

cdef inline idx_type clamp(float n, idx_type mn, idx_type mx):
    cdef idx_type i = <idx_type> n
    if i < mn:
        return mn
    if i > mx:
        return mx
    return i

cdef inline int oob(idx_type i, idx_type N):
    if (i<0) or (i>=N):
        return 1
    else:
        return 0

