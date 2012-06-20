import numpy as np

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
    cdef label_type init = -1
    # find first label of sufficient mass
    m = 0
    while init < 0 and m < n:
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

@cython.boundscheck(False)
def draw_boundaries(
        np.ndarray[label_type, ndim=2] seg_map, list saddles
        ):
    pd = dict()
    for s in saddles:
        sub = s.sub
        dom = s.dom
        if (sub,dom) in pd or (dom,sub) in pd:
            print "correspondence exists!"
        pd[(sub,dom)] = s.persistence
        pd[(dom,sub)] = s.persistence

    cdef int Nx = seg_map.shape[1], Ny = seg_map.shape[0]
    cdef int x, xp, xm, y, yp, ym
    cdef int l, lu, ld, ll, lr
    cdef double gray, pst
    cdef double mx_pst = max(saddles).persistence
    #cdef np.ndarray[np.uint8_t, ndim=2] bmap = np.zeros((Ny,Nx), 'B')
    cdef np.ndarray[np.float64_t, ndim=2] bmap = np.zeros((Ny,Nx), 'd')
    for y in xrange(Ny):
        yp = min(Ny-1, y+1)
        ym = max(0, y-1)
        for x in xrange(Nx):
            xp = min(Nx-1, x+1)
            xm = max(0, x-1)

            l = seg_map[y,x]
            lu = seg_map[yp,x]
            ld = seg_map[ym,x]
            ll = seg_map[y,xm]
            lr = seg_map[y,xp]
            
            gray = -1.0
            if l != lu:
                pst = pd.get((l,lu), -1)
                gray = max(gray, pst)
            if l != ld:
                pst = pd.get((l,ld), -1)
                gray = max(gray, pst)
            if l != ll:
                pst = pd.get((l,ll), -1)
                gray = max(gray, pst)
            if l != lr:
                pst = pd.get((l,lr), -1)
                gray = max(gray, pst)
            if gray > 0:
                gray = (gray/mx_pst)**(0.5)
                bmap[y,x] = 0.1 + 0.9*gray
    return bmap
