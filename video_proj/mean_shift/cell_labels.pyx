""" -*- python -*- file
"""
import numpy as np
cimport numpy as cnp
cimport cython

# D is double float -- just make an array of signed ints
@cython.boundscheck(False)
def assign_modes_by_density(
        cnp.ndarray D, cnp.int32_t boundary=-1
        ):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Df = D.ravel()
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] gs = np.argsort(Df)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.zeros(gs.shape, 'i')
    cdef int x
    cdef cnp.uint32_t g
    cdef cnp.int32_t test, next_label = 1
    for x in xrange(len(gs)-1, -1, -1):
        g = gs[x]
        nb_idx = cell_neighbors(g, dims)
        nbs = labels[nb_idx]
        pos_nbs = nbs[nbs>0]
        # if there are no modes assigned to this group, assign a new one
        if len(pos_nbs)==0:
            labels[g] = next_label
            next_label += 1
            continue
        # if neighborhood is consistent, then removing the bias will
        # leave nothing behind
        test = pos_nbs[0]
        if np.sum(pos_nbs - test) == 0:
            labels[g] = test
            continue
        # if the non-negative neighboring points are mixed modal, then
        # mark down as a boundary        
        labels[g] = boundary

    labels_nd = np.reshape(labels, dims)
    return labels_nd, next_label-1

def cell_neighbors(c_idx, dims):
    # if c_idx is a flattened coordinate (eg, z*Nz*Ny + y*Ny + x),
    # then unravel it and proceed
    if np.isscalar(c_idx):
        scl_idx = True
        c_idx = np.lib.index_tricks.unravel_index(c_idx, dims)
    else:
        scl_idx = False

    nd = len(dims)
    # pad blank dimensions into c_idx
    c_idx = np.array(c_idx)
    c_idx.shape = (nd,) + (1,)*nd
    nb_offsets = np.mgrid[ (slice(-1,2),) * nd ].astype('i')
    nb_idx = (nb_offsets + c_idx).reshape( nd, -1 )
    # reject coordinates outside frame
    keep_cols = (nb_idx >=0 ).all(axis=0)
    nb_idx = nb_idx[:, keep_cols]
    keep_cols = np.ones( (nb_idx.shape[1],), np.bool )
    for k in xrange(nd):
        keep_cols &= nb_idx[k] < dims[k]
    nb_idx = nb_idx[:,keep_cols]
    if scl_idx:
        # finally, transform back to 1D indices
        return np.lib.index_tricks.ravel_multi_index(nb_idx, dims)
    return nb_idx
