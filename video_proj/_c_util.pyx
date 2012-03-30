cimport cython
cimport numpy as np

ctypedef np.int32_t label_type # XXX: this should not be redefined here

@cython.boundscheck(False)
def cluster_sizes(np.ndarray[label_type, ndim=2] seg_map):
    cdef int c, m, n, Ny = seg_map.shape[0], Nx = seg_map.shape[1]
    cdef label_type ell
    cdef dict cluster_counts = dict()
    for m in range(Ny):
        for n in range(Nx):
            ell = seg_map[m,n]
            c = cluster_counts.get( ell, 0 )
            cluster_counts[ell] = c + 1
    return cluster_counts
        
        
@cython.boundscheck(False)
def cluster_size_filter(np.ndarray[label_type, ndim=2] seg_map, int rho):
    # relabel seg_map based on cluster size threshold
    cdef dict c_sizes = cluster_sizes(seg_map)
    cdef np.ndarray[label_type, ndim=1] flat_map = seg_map.ravel()
    cdef int k, m, n = seg_map.size
    cdef label_type ell
    cdef label_type init = 0
    # find first label of sufficient mass
    m = 0
    while not init:
        ell = flat_map[m]
        if c_sizes[ell] < rho:
            m += 1
        else:
            init = ell
    flat_map[:m] = init
    # now grow from here
    for k in range(m+1,n):
        ell = flat_map[k]
        if ell > 0 and c_sizes[ell] < rho:
            flat_map[k] = flat_map[k-1]
