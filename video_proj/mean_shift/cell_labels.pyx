""" -*- python -*- file
"""
import numpy as np
cimport numpy as np
cimport cython

# D is double float -- just make an array of signed ints
@cython.boundscheck(False)
def assign_modes_by_density(
        D, np.int32_t boundary=-1
        ):
    dims = D.shape
    cdef np.ndarray[np.float64_t, ndim=1] Df = D.ravel()
    cdef np.ndarray[np.int64_t, ndim=1] gs = np.argsort(Df)
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros((len(gs),), 'i')
    cdef np.ndarray[np.int32_t, ndim=1] nbs, pos_nbs
    cdef np.ndarray[np.int64_t, ndim=1] nb_idx
    cdef int x, g
    cdef np.int32_t test, next_label = 1
    for x in xrange(len(gs)-1, -1, -1):
        if x%1000 == 0:
            print x, ' '
        g = gs[x]
        ## nb_idx = cell_neighbors(g, dims)
        nb_idx = np.ravel_multi_index(
            cell_neighbors_brute(
                np.array(np.unravel_index(g, dims)), dims
                ),
                dims
            )
                
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

nb_1x1_offsets_6D = \
  np.mgrid[ (slice(-1,2),) * 6 ].reshape(6, -1)
nb_1x1_offsets_5D = \
  np.mgrid[ (slice(-1,2),) * 5 ].reshape(5, -1)
nb_1x1_offsets_3D = \
  np.mgrid[ (slice(-1,2),) * 3 ].reshape(3, -1)

cdef np.ndarray cell_neighbors(c_idx, dims):
    # if c_idx is a flattened coordinate (eg, z*Nx*Ny + y*Nx + x),
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
    if nd==3:
        nb_offsets = nb_1x1_offsets_3D
    elif nd==5:
        nb_offsets = nb_1x1_offsets_5D
    else:
        nb_offsets = nb_1x1_offsets_6D
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

cdef inline int oob(np.int64_t i, int N):
    if (i < 0) or (i >= N):
        return 1
    return 0

def cell_neighbors_brute(np.ndarray[np.int64_t, ndim=1] idx, dims):
    cdef int k = 0        
    cdef np.ndarray[np.int64_t, ndim=2] nb_idx = np.empty( (5, 3**5), 'l' )
    cdef np.int64_t a0, a1, a2, a3, a4
    for a0 in range(-1,2):
        if oob(idx[0]+a0, dims[0]):
            continue
        for a1 in range(-1,2):
            if oob(idx[1]+a1, dims[1]):
                continue
            for a2 in range(-1,2):
                if oob(idx[2]+a2, dims[2]):
                    continue
                for a3 in range(-1,2):
                    if oob(idx[3]+a3, dims[3]):
                        continue
                    for a4 in range(-1,2):
                        if oob(idx[4]+a4, dims[4]):
                            continue
                        nb_idx[0,k] = idx[0]+a0
                        nb_idx[1,k] = idx[1]+a1
                        nb_idx[2,k] = idx[2]+a2
                        nb_idx[3,k] = idx[3]+a3
                        nb_idx[4,k] = idx[4]+a4
                        k += 1
    return nb_idx[:,:k]

