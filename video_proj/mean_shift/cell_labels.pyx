# cython: profile=True
""" -*- python -*- file
"""
import numpy as np
cimport numpy as np
cimport cython

from video_proj.indexing cimport multi_idx, flatten_idc, oob, idx_type
from ..indexing import py_idx_type as pidx_t

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

cdef class Mode:
    cdef public idx_type idx
    cdef public double elevation
    cdef public int label

    cdef public list neighbors
    
    def __init__(self, idx, elevation, label):
        self.idx = idx
        self.elevation = elevation
        self.label = label

    def __repr__(self):
        return 'index: %d, elevation: %1.2f, label %d, neighbors: %s'%(
            self.idx, self.elevation, self.label, self.neighbors
            )

# XXX: this will be merged into active code
cdef class NewSaddle:
    cdef public idx_type idx
    cdef public double elevation, persistence
    cdef public Mode sub, dom

    def __init__(self, idx, elevation, m1, m2):
        if m1.elevation > m2.elevation:
            self.sub = m2
            self.dom = m1
        else:
            self.sub = m1
            self.dom = m2
        self.persistence = self.sub.elevation - elevation
        self.elevation = elevation
        self.idx = idx

    def __repr__(self):
        s = 'index: %d, elevation: %1.2f, persistence %1.2f\n'%(
            self.idx, self.elevation, self.persistence
            )
        s = s+'Sub Mode: %s\nDom Mode: %s'%(self.sub, self.dom)
        return s

# XXX: should consider pre-masking out zero occupancy or density cells
@cython.boundscheck(False)
def assign_modes_by_density(
        D, np.int32_t boundary=-1
        ):
    clusters = dict()
    peak_by_mode = dict()
    saddles = list()
    ## cdef tuple dims = D.shape
    cdef np.ndarray[idx_type, ndim=1] dims = np.array(D.shape, dtype=pidx_t)
    cdef int nd = len(dims)
    cdef np.ndarray[np.float64_t, ndim=1] Df = D.ravel()
    cdef np.ndarray[idx_type, ndim=1] gs = np.argsort(Df)
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros((len(gs),), 'i')
    cdef np.ndarray[np.int32_t, ndim=1] nbs, pos_nbs
    cdef np.ndarray[idx_type, ndim=1] pm = np.array([-1, 1], dtype=pidx_t)
    cdef np.ndarray[idx_type, ndim=1] nb_idx
    cdef np.ndarray[idx_type, ndim=2] nb_idx_arr = \
        np.empty((nd, 3**nd), dtype=pidx_t)
    cdef np.ndarray[idx_type, ndim=1] g_nd = np.empty((nd,), dtype=pidx_t)
    cdef int x, i
    cdef idx_type g
    ## cdef list g_nd = [0]*nd
    cdef np.int32_t test, next_label = 1
    for x in xrange(len(gs)-1, -1, -1):
        if x%1000 == 0:
            print x, ' '
        g = gs[x]
        # if miraculously this is a one dimensional density, then
        # the cell neighbors are [g-1, g+1]
        if nd == 1:
            nbs = np.take(labels, pm+g, mode='clip')
        else:
            multi_idx(g, <idx_type*>dims.data, nd, <idx_type*>g_nd.data)
            num_nb = cell_neighbors_brute(g_nd, dims, nb_idx_arr)
            nb_idx = flatten_idc(nb_idx_arr[:,:num_nb], dims)
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

@cython.boundscheck(False)
cdef int cell_neighbors_brute(
    np.ndarray[idx_type, ndim=1] idx,
    np.ndarray[idx_type, ndim=1] dims,
    np.ndarray[idx_type, ndim=2] nb_idx
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
            if nd == 2:
                nb_idx[0,k] = idx[0]+a0
                nb_idx[1,k] = idx[1]+a1
                k += 1
            else:
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

