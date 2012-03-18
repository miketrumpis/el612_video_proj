# cython: profile=True
""" -*- python -*- file
"""
import numpy as np
cimport numpy as np
cimport cython

# a simple container
class Saddle(object):
    def __init__(self, idx, elevation, neighbors):
        self.idx = idx
        self.elevation = elevation
        self.neighbors = neighbors

    def __repr__(self):
        return 'index: %d, elevation: %1.2f, neighbors: %s'%(
            self.idx, self.elevation, self.neighbors
            )

@cython.boundscheck(False)
def assign_modes_by_density(
        D, np.int32_t boundary=-1
        ):
    clusters = dict()
    peak_by_mode = dict()
    saddles = list()
    cdef tuple dims = D.shape
    cdef int nd = len(dims)
    cdef np.ndarray[np.float64_t, ndim=1] Df = D.ravel()
    cdef np.ndarray[np.int64_t, ndim=1] gs = np.argsort(Df)
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros((len(gs),), 'i')
    cdef np.ndarray[np.int32_t, ndim=1] nbs, pos_nbs
    cdef np.ndarray[np.int64_t, ndim=1] nb_idx
    cdef np.ndarray[np.int64_t, ndim=2] nb_idx_arr = np.empty((nd, 3**nd), 'l')
    cdef np.ndarray[np.int64_t, ndim=1] g_nd_arr = np.empty((nd,), 'l')
    cdef int x, i, g
    cdef list g_nd = [0]*nd
    cdef np.int32_t test, next_label = 1
    for x in xrange(len(gs)-1, -1, -1):
        if x%1000 == 0:
            print x, ' '
        g = gs[x]
        multi_idx(g, dims, g_nd)
        for i in range(nd):
            g_nd_arr[i] = g_nd[i]
        num_nb = cell_neighbors_brute(g_nd_arr, dims, nb_idx_arr)
        nb_idx = flatten_idx(nb_idx_arr[:,:num_nb], dims)
        nbs = np.take(labels, nb_idx)
        pos_idx = np.where(nbs>0)[0]
        pos_nbs = np.take(nbs, pos_idx)
        # if there are no modes assigned to this group, assign a new one
        if len(pos_nbs)==0:
            labels[g] = next_label
            # start new cluster and note the peak
            clusters[<int>next_label] = [g]
            peak_by_mode[<int>next_label] = Df[g]
            next_label += 1
            continue
        # if neighborhood is consistent, then removing the bias will
        # leave nothing behind
        test = pos_nbs[0]
        if np.sum(pos_nbs - test) == 0:
            labels[g] = test
            clusters[<int>test].append(g)
            continue
        # if the non-negative neighboring points are mixed modal, then
        # mark down as a boundary        
        labels[g] = boundary
        saddles.append( Saddle(g, Df[g], np.unique(pos_nbs)) )

    saddles = sorted(saddles, key=lambda x: x.elevation, reverse=True)
    labels_nd = np.reshape(labels, dims)
    return labels_nd, clusters, peak_by_mode, saddles

def flatten_idx_passthru(np.ndarray[np.int64_t, ndim=2] idx, tuple dims):
    return flatten_idx(idx, dims)

def multi_idx_passthru(int flat_idx, tuple dims):
    idx = [0]*len(dims)
    multi_idx(flat_idx, dims, idx)
    return idx

@cython.boundscheck(False)
cdef np.ndarray[np.int64_t, ndim=1] flatten_idx(
    np.ndarray[np.int64_t, ndim=2] idx,
    tuple dims
    ):
    cdef int nd = len(dims)
    cdef int n_idx = idx.shape[1]
    cdef int j, k, stride=1
    cdef np.ndarray[np.int64_t, ndim=1] f_idx = np.zeros((n_idx,), 'l')
    for k in range(nd-1, -1, -1):
        for j in range(n_idx):
            f_idx[j] += idx[k,j] * stride
        stride *= dims[k]
    return f_idx

cdef inline void multi_idx(int flat_idx, tuple dims, list idx):
    cdef int k, nd = len(dims)
    for k in range(nd-1, -1, -1):
        idx[k] = flat_idx % dims[k]
        flat_idx -= idx[k]
        flat_idx = <int> (flat_idx / dims[k])
    return

cdef inline int oob(np.int64_t i, int N):
    if (i < 0) or (i >= N):
        return 1
    return 0

@cython.boundscheck(False)
cdef int cell_neighbors_brute(
    np.ndarray[np.int64_t, ndim=1] idx,
    tuple dims,
    np.ndarray[np.int64_t, ndim=2] nb_idx
    ):
    cdef int k = 0        
    cdef np.int64_t a0, a1, a2, a3, a4
    cdef int nd = len(dims)
    for a0 in range(-1,2):
        if oob(idx[0]+a0, dims[0]):
            continue
        for a1 in range(-1,2):
            if oob(idx[1]+a1, dims[1]):
                continue
            for a2 in range(-1,2):
                if oob(idx[2]+a2, dims[2]):
                    continue
                if nd == 3:
                    nb_idx[0,k] = idx[0]+a0
                    nb_idx[1,k] = idx[1]+a1
                    nb_idx[2,k] = idx[2]+a2
                    k += 1
                else:
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
    return k

