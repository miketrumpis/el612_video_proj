# cython: profile=True
""" -*- python -*- file
"""
import numpy as np
import scipy.ndimage as ndimage
import itertools
from copy import deepcopy

cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from cython.operator cimport dereference

from video_proj.indexing cimport multi_idx, flatten_idc, oob, idx_type, \
    array_oob, flat_idx
from ..indexing import py_idx_type as pidx_t
from video_proj.mean_shift.histogram import histogram
from video_proj.mean_shift.classification import PixelModeClassifier
from video_proj import colors, util

cdef double huge = 1e16
ctypedef np.int32_t label_type
py_label_type = np.int32

def smooth_density(p, sigma, accum=None, **gaussian_kws):
    p = ndimage.gaussian_filter(p, sigma, **gaussian_kws)
    if accum is not None:
        # filter vector valued function accum coordinate-wise
        n_coords = accum.shape[-1]
        for n in xrange(n_coords):
            accum[...,n] = ndimage.gaussian_filter(
                accum[...,n], sigma, **gaussian_kws
                )
        return p, accum
    return p

cdef class Mode:
    cdef public idx_type idx
    cdef public double elevation
    cdef public label_type label

    cdef public set neighbors
    
    def __init__(self, idx, elevation, label):
        self.idx = idx
        self.elevation = elevation
        self.label = label
        self.neighbors = set()

    def __repr__(self):
        return 'index: %d, elevation: %1.2f, label %d, neighbors: %s'%(
            self.idx, self.elevation, self.label, self.neighbors
            )

    def __richcmp__(self, m2, int op):
        if op == 0:
            return self.elevation < m2.elevation
        return False

def neighboring_modes(m1, m2):
    if len(m1.neighbors) < len(m2.neighbors):
        return m2.label in m1.neighbors
    else:
        return m1.label in m2.neighbors

cdef class Saddle:
    cdef public idx_type idx
    cdef public double elevation, persistence
    ## cdef public Mode sub, dom
    cdef public int sub, dom

    def __init__(self, idx, elevation, m1, m2):
        if m1 < m2:
            self.sub = m1.label
            self.dom = m2.label
            self.persistence = m1.elevation - elevation
        elif m2 < m1:
            self.sub = m2.label
            self.dom = m1.label
            self.persistence = m2.elevation - elevation
        else:
            self.sub = m2.label
            self.dom = m1.label
            self.persistence = huge
        self.elevation = elevation
        self.idx = idx

    def __repr__(self):
        s = 'index: %d, elevation: %1.2f, persistence %1.2f\n'%(
            self.idx, self.elevation, self.persistence
            )
        ## s = s+'Sub Mode: %s\nDom Mode: %s'%(self.sub, self.dom)
        s = s+'Sub Mode: %d\nDom Mode: %d'%(self.sub, self.dom)
        return s

    def __richcmp__(self, s2, int op):
        if op == 0:
            return self.persistence < s2.persistence
        return False

# This should be out of date
## def update_saddle(Saddle saddle, dict modes, update_dict=None):
##     dom = saddle.dom
##     sub = saddle.sub
##     if update_dict:
##         dom = update_dict[dom]
##         sub = update_dict[sub]
##     if modes[dom] < modes[sub]:
##         saddle.sub = dom
##         saddle.dom = sub
##     if dom == sub:
##         saddle.persistence = huge
##     else:
##         sm = modes[saddle.sub]
##         saddle.persistence = sm.elevation - saddle.elevation
##     return

cdef class Boundary:
    cdef public idx_type idx
    cdef public double elevation
    cdef public set neighbors

    def __init__(self, idx, elevation, neighbors):
        self.idx = idx
        self.elevation = elevation
        self.neighbors = neighbors

cdef class ModeSeeking:

    # a cyclical list of label lookups
    # -- when true_labels[i] == i, then i is a surviving label,
    # -- otherwise, keep cycling with the returned index
    cdef vector[label_type] *_true_labels

    # state booleans and modes
    cdef int _spatial, _accum_feats, _trained

    # Need to keep these around to produce new classifiers
    cdef np.ndarray grid, labels
    cdef list cell_edges
    cdef tuple domain_dims

    # state variables
    cdef public list saddles
    cdef public list boundaries
    cdef public dict clusters
    cdef public double spatial_bw, color_bw

    def __init__(
            self, spatial_bw=None, color_bw=None,
            accumulated_features=True
            ):

        if not (spatial_bw or color_bw):
            raise ValueError(
                'Spatial and Color bandwidths not specified. '+\
                'At least one of the feature bandwidths must be set.'
                )
        # XXX: need to check logic here
        if not spatial_bw:
            self._spatial = 0
            self.spatial_bw = 1
        else:
            self._spatial = 1
            self.spatial_bw = spatial_bw
        if not color_bw:
            self.color_bw = 1
        else:
            self.color_bw = color_bw

        if not accumulated_features:
            print 'Note: accumulated features reset to be True'
        self._accum_feats = 1

        self._true_labels = new vector[label_type]()
        self.saddles = list()
        self.boundaries = list()
        self.clusters = dict()
        self._trained = False

    cdef _get_label(self, label_type label):
        ## cdef vector[label_type] true_labels = self._true_labels[0]        
        if not self._true_labels[0].size():
            return None
        cdef label_type final = self._true_labels[0][label]
        while final != self._true_labels[0][final]:
            final = self._true_labels[0][final]
        return final
            
    def get_label(self, label_type label):
        return self._get_label(label)

    def reset_labels(self):
        if not self._trained:
            raise RuntimeError('This model is not yet trained')
        self._true_labels.clear()
        self._true_labels.push_back(0)
        for k in self.clusters.keys():
            self._true_labels.push_back(k)

    property true_labels:

        def __get__(self):
            ## cdef vector[label_type] tl = self._true_labels[0]
            cdef int n_labels = self._true_labels[0].size()
            return [self._true_labels[0][i] for i in xrange(n_labels)]

    cdef _update_saddle(self, Saddle saddle, copy=False):
        # don't modified the original labels of the saddle,
        # just the persistence
        dom = self._get_label(saddle.dom)
        sub = self._get_label(saddle.sub)
        if copy:
            return Saddle(
                saddle.idx, saddle.elevation,
                self.clusters[dom], self.clusters[sub]
                )
        if self.clusters[dom] < self.clusters[sub]:
            saddle.sub = dom
            saddle.dom = sub
        if dom == sub:
            saddle.persistence = huge
        else:
            sm = self.clusters[saddle.sub]
            saddle.persistence = sm.elevation - saddle.elevation
        return

    def update_all_saddles(self, copy=False):
        if not self._trained:
            raise RuntimeError('This model is not yet trained')
        if copy:
            return [self._update_saddle(s, True) for s in self.saddles]
        # otherwise make saddles current with the state of true_labels
        for s in self.saddles:
            self._update_saddle(s)

    def persistence_merge(self, thresh):
        if not self._trained:
            raise RuntimeError('This model is not yet trained')
        # this should do merging of persistent modes, and then
        # return a new PixelClassifier
        saddles = self.update_all_saddles(copy=True)
        cdef Saddle min_saddle
        merged = False
        while not merged:
            merged = True
            self.update_all_saddles()
            min_saddle = min(self.saddles)
            ## for min_saddle in saddles:
            ##     self._update_saddle(min_saddle)
            ## min_saddle = min(saddles)
            ## print min_saddle.persistence
            if min_saddle.persistence < thresh:
                self._true_labels[0][min_saddle.sub] = min_saddle.dom
                merged = False
        cls = self.classifier_from_state()
        ## self.reset_labels()
        ## self.saddles = saddles
        return cls

    cdef _merged_boundaries(self):
        cdef set nb_set
        cdef Boundary b
        cdef list ix = list()
        cdef list ell = list()
        for b in self.boundaries:
            nb_set = set( [self._get_label(f) for f in b.neighbors] )
            if len(nb_set) == 1:
                ix.append(b.idx)
                ell.append(nb_set.pop())
        return np.array(ix, dtype=pidx_t), np.array(ell, dtype=py_label_type)

    @cython.boundscheck(False)
    def classifier_from_state(self):
        if not self._trained:
            raise RuntimeError('This model is not yet trained')
        cdef np.ndarray[label_type, ndim=1] new_labels = \
            self.labels.ravel().copy()
        cdef int k, n
        cdef label_type x
        # new labels should be udpated to include every boundary
        # point that is no longer a true boundary (which has been
        # merged into a dominant mode through persistence)
        cdef np.ndarray[idx_type, ndim=1] ix
        cdef np.ndarray[label_type, ndim=1] ell
        ix, ell = self._merged_boundaries()
        ## print ix, ell
        n = len(ix)
        for k in range(n):
            new_labels[ix[k]] = ell[k]
        # now replace all labels with true labels
        n = new_labels.size
        for k in range(n):
            x = new_labels[k]
            if x > 0:
                new_labels[k] = self._get_label(x)
        return PixelModeClassifier(
            np.reshape(new_labels, self.domain_dims),
            self.grid, self.cell_edges, spatial=self._spatial
            )

    def train_on_image(self, image, density_xform=np.log):
        # This should return something like a PixelClassifier
        b, accum, edges = histogram(
            image, self.spatial_bw,
            spatial_features=self._spatial,
            color_spacing = self.color_bw
            )

        zmask = (b>0)
        p, mu = smooth_density(b, 1, accum=accum, mode='constant')
        del b
        del accum
        self.grid = np.concatenate( (mu, p[...,None]), axis=-1).copy()
        self.cell_edges = edges
        self.domain_dims = p.shape
        if density_xform is None:
            density_xform = lambda x: x
        self.labels = self.assign_modes(density_xform(p), zmask.astype('B'))
        self._trained = 1
        return PixelModeClassifier(
            self.labels, self.grid, self.cell_edges, spatial=self._spatial
            )

    @cython.boundscheck(False)
    def assign_modes(
            self, D, np.ndarray zmask, np.int32_t boundary=-1
            ):

        #
        self.clusters = dict()
        self.saddles = list()

        ## cdef vector[label_type] true_labels = self._true_labels[0]

        ## self._true_labels[0].clear()
        ## self._true_labels[0].push_back(0)
        self._true_labels.clear()
        self._true_labels.push_back(0)

        # dims, nd are for indexing conversions
        cdef np.ndarray[idx_type, ndim=1] dims = np.array(D.shape, dtype=pidx_t)
        cdef int nd = len(dims)

        # the flattened density -- conveniently agnostic to D's original shape
        cdef np.ndarray[np.float64_t, ndim=1] Df = D.ravel()

        # Set up a sorted list of occupied cells
        # zmask is a binary array which is 0 at the location of
        # zero-occupancy cells
        cdef np.ndarray[idx_type, ndim=1] gs1 = np.argsort(Df)
        cdef np.ndarray[np.uint8_t, ndim=1] zmask_f = \
            np.take(zmask, gs1)
        cdef int n_occup = zmask_f.sum()
        cdef np.ndarray[idx_type, ndim=1] gs = np.empty((n_occup,), dtype=pidx_t)

        cdef int x, i, j, n_cells = len(Df)
        i = 0
        for x in xrange(n_cells):
            if zmask_f[x]:
                gs[i] = gs1[x]
                i += 1

        # The array of labels
        cdef np.ndarray[label_type, ndim=1] labels = \
            np.zeros((n_cells,), dtype=py_label_type)

        # adjacency neighborhood offsets
        ## cdef np.ndarray[idx_type, ndim=2] nb_offsets = \
        ##     np.mgrid[ (slice(-1,2),) * nd ].astype(pidx_t).reshape(nd,-1).T
        cdef np.ndarray[idx_type, ndim=2] nb_idx_arr = \
            np.empty((nd, 3**nd), dtype=pidx_t)
        cdef int num_nb #= nb_offsets.shape[0]
        # revolving index and center cell index
        cdef np.ndarray[idx_type, ndim=1] nb_idx
        cdef np.ndarray[idx_type, ndim=1] g_nd = np.empty((nd,), dtype=pidx_t)
        # flat index
        cdef idx_type g, f_nb_idx

        cdef label_type test, next_label = 1
        cdef label_type tallest_neighbor_label
        cdef double tallest_neighbor_height

        # used within the loops below
        cdef list neighbors
        cdef set nb_clusters
        cdef Mode m1, m2

        for x in range(n_occup-1, -1, -1):
            if x%1000 == 0:
                print x, ' '
            g = gs[x]

            ## neighbors = list()
            nb_clusters = set()

            multi_idx(g, <idx_type*>dims.data, nd, <idx_type*>g_nd.data)
            num_nb = cell_neighbors_brute(g_nd, dims, nb_idx_arr)
            nb_idx = flatten_idc(nb_idx_arr[:,:num_nb], dims)
            nbs = np.take(labels, nb_idx)
            tallest_neighbor_height = -1e10
            for i in range(num_nb):
                test = nbs[i]
                if test > 0:
                    if Df[i] > tallest_neighbor_height:
                        tallest_neighbor_height = Df[i]
                        tallest_neighbor_label = test
                    nb_clusters.add(test)

            if len(nb_clusters) == 0:
                # easy, just add new cluster
                density = Df[g]
                self.clusters[next_label] = Mode(g, density, next_label)
                ## self._true_labels[0].push_back(next_label)
                self._true_labels.push_back(next_label)
                labels[g] = next_label
                next_label += 1
            elif len(nb_clusters) == 1:
                # also easy, just copy label
                labels[g] = tallest_neighbor_label
            else:
                for label_1, label_2 in itertools.combinations(nb_clusters, 2):
                    m1 = self.clusters[label_1]
                    m2 = self.clusters[label_2]
                    # if these modes were not already joined at a
                    # higher elevation saddle point, then join them here
                    if not neighboring_modes(m1, m2):
                        ## self._saddles[0].push_back(
                        ##     Saddle(g, Df[g], m1, m2)
                        ##     )
                        self.saddles.append( Saddle(g, Df[g], m1, m2) )
                        m1.neighbors.add(label_2)
                        m2.neighbors.add(label_1)
		# In any case, mark this boundary in case it needs to
		# be merged later
                self.boundaries.append( Boundary(g, density, nb_clusters) )
                labels[g] = -2 * tallest_neighbor_label #?
        return np.reshape(labels, D.shape)


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
