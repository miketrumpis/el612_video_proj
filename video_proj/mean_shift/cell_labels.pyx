# cython: profile=True
""" -*- python -*- file
"""
import numpy as np
import itertools

cimport numpy as np
cimport cython
from libcpp.vector cimport vector

from video_proj.indexing cimport multi_idx, flatten_idc, oob, idx_type, \
    array_oob, flat_idx
from ..indexing import py_idx_type as pidx_t
from video_proj.mean_shift import modes, histogram
from video_proj import colors, util

cdef double huge = 1e16
ctypedef np.int32_t label_type
py_label_type = np.int32

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

# XXX: this will be merged into active code
cdef class NewSaddle:
    cdef public idx_type idx
    cdef public double elevation, persistence
    ## cdef public Mode sub, dom
    cdef public int sub, dom

    def __init__(self, idx, elevation, m1, m2):
        if m1 < m2:
            self.sub = m1.label
            self.dom = m2.label
            self.persistence = m1.elevation - elevation
        else:
            self.sub = m2.label
            self.dom = m1.label
            self.persistence = m2.elevation - elevation
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

def update_saddle(NewSaddle saddle, dict modes, update_dict=None):
    dom = saddle.dom
    sub = saddle.sub
    if update_dict:
        dom = update_dict[dom]
        sub = update_dict[sub]
    if modes[dom] < modes[sub]:
        saddle.sub = dom
        saddle.dom = sub
    if dom == sub:
        saddle.persistence = huge
    else:
        sm = modes[saddle.sub]
        saddle.persistence = sm.elevation - saddle.elevation
    return

@cython.boundscheck(False)
def new_assign_modes_by_density(
        D, np.ndarray zmask, np.int32_t boundary=-1
        ):
    cdef dict clusters = dict()
    cdef dict peaks = dict()
    cdef list saddles = list()
    cdef list true_label = list([0])

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
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros((n_cells,), 'i')

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

    cdef np.int32_t test, next_label = 1
    cdef np.int32_t tallest_neighbor_label
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
            # possibly inner-loop this
            ## nb_idx = g_nd + nb_offsets[i]
            ## nb_idx = 
            ## if array_oob(<idx_type*> nb_idx.data, <idx_type*> dims.data, nd):
            ##     continue
            ## f_nb_idx = flat_idx(
            ##     <idx_type*> nb_idx.data, nd, <idx_type*> dims.data
            ##     )
            ## test = labels[f_nb_idx]
            test = nbs[i]
            if test > 0:
                if Df[i] > tallest_neighbor_height:
                    tallest_neighbor_height = Df[i]
                    tallest_neighbor_label = test
                ## neighbors.append( (test, Df[f_nb_idx]) )
                nb_clusters.add(test)

        if len(nb_clusters) == 0:
            # easy, just add new cluster
            density = Df[g]
            clusters[next_label] = Mode(g, density, next_label)
            true_label.append(next_label)
            labels[g] = next_label
            peaks[next_label] = density
            next_label += 1
        elif len(nb_clusters) == 1:
            # also easy, just copy label
            labels[g] = tallest_neighbor_label #nb_clusters.pop()
        else:
            ## s_nb = sorted(neighbors, key=lambda x: x[1], reverse=True)
            ## test = s_nb[0][0]
            for label_1, label_2 in itertools.combinations(nb_clusters, 2):
                m1 = clusters[label_1]
                m2 = clusters[label_2]
                ## print label_1, m1
                ## print label_2, m2
                # if these modes were not already joined at a
                # higher elevation saddle point, then join them here
                if not neighboring_modes(m1, m2):
                    saddles.append( NewSaddle(g, Df[g], m1, m2) )
                    m1.neighbors.add(label_2)
                    m2.neighbors.add(label_1)
            labels[g] = -2 * tallest_neighbor_label #?
    return labels, clusters, peaks, saddles #,  true_label
                                





# XXX: should consider pre-masking out zero occupancy or density cells
@cython.boundscheck(False)
def assign_modes_by_density(
        D, np.ndarray zmask, np.int32_t boundary=-1
        ):
    clusters = dict()
    peak_by_mode = dict()
    saddles = list()


    cdef np.ndarray[idx_type, ndim=1] dims = np.array(D.shape, dtype=pidx_t)
    cdef int nd = len(dims)
    cdef np.ndarray[np.float64_t, ndim=1] Df = D.ravel()

    # Set up a sorted list of occupied cells
    # zmask is a binary array which is 0 at the location of
    # zero-occupancy cells
    cdef np.ndarray[idx_type, ndim=1] gs1 = np.argsort(Df)
    cdef np.ndarray[np.uint8_t, ndim=1] zmask_f = \
        np.take(zmask, gs1)
    cdef int n_occup = zmask_f.sum()
    cdef np.ndarray[idx_type, ndim=1] gs = np.empty((n_occup,), dtype=pidx_t)

    cdef int x, i, n_cells = len(Df)
    i = 0
    for x in xrange(n_cells):
        if zmask_f[x]:
            gs[i] = gs1[x]
            i += 1
    
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros((n_cells,), 'i')
    cdef np.ndarray[np.int32_t, ndim=1] nbs, pos_nbs
    cdef np.ndarray[idx_type, ndim=1] pm = np.array([-1, 1], dtype=pidx_t)
    cdef np.ndarray[idx_type, ndim=1] nb_idx
    cdef np.ndarray[idx_type, ndim=2] nb_idx_arr = \
        np.empty((nd, 3**nd), dtype=pidx_t)
    cdef np.ndarray[idx_type, ndim=1] g_nd = np.empty((nd,), dtype=pidx_t)
    cdef idx_type g
    cdef np.int32_t test, next_label = 1
    for x in xrange(n_occup-1, -1, -1):
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

cdef class ModeSeeking:

    # a cyclical list of label lookups
    # -- when true_labels[i] == i, then i is a surviving label,
    # -- otherwise, keep cycling with the returned index
    cdef vector[label_type] *_true_labels
    ## cdef vector[NewSaddle] *_saddles
    cdef int _spatial
    cdef str _color_mode
    cdef int _accum_feats
    
    cdef public list saddles
    cdef public dict clusters
    cdef public dict peaks
    cdef public double spatial_bw, color_bw
    cdef public np.ndarray grid, labels

    def __init__(
            self, spatial_bw=None, color_bw=None,
            color_mode='LAB', accumulated_features=True
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

        self._color_mode = color_mode.upper()
        if not accumulated_features:
            print 'Note: accumulated features reset to be True'
        self._accum_feats = 1

        self._true_labels = new vector[label_type]()
        ## self._saddles = new vector[NewSaddle]()

    def _process_image(self, image):
        # returns color converted image and feature vectors
        if len(image.shape) > 2:
            if self._color_mode == 'LAB':
                image = colors.rgb2lab(image)
        else:
            image = image.astype('d')
        features = util.image_to_features(image)
        if not self._spatial:
            features = features[:,2:]
        return image, features

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

    property true_labels:

        def __get__(self):
            ## cdef vector[label_type] tl = self._true_labels[0]
            cdef int n_labels = self._true_labels[0].size()
            return [self._true_labels[0][i] for i in xrange(n_labels)]

    def train_on_image(self, image, density_xform=np.log):
        # This should return something like a PixelClassifier
        image, features = self._process_image(image)
        b, accum, edges = histogram.histogram(
            image, self.spatial_bw,
            spatial_features=self._spatial,
            color_spacing = self.color_bw
            )

        p, mu = modes.smooth_density(b, 1, accum=accum, mode='constant')
        self.grid = np.concatenate( (mu, p[...,None]), axis=-1).copy()
        zmask = (b>0)
        self.assign_modes(density_xform(p), zmask.astype('B'))

    def persistence_merge(self, thresh):
        # this should do merging of persistent modes, and then
        # return a new PixelClassifier
        pass

    @cython.boundscheck(False)
    def assign_modes(
            self, D, np.ndarray zmask, np.int32_t boundary=-1
            ):

        #
        self.clusters = dict()
        self.peaks = dict()
        self.saddles = list()

        ## cdef vector[label_type] true_labels = self._true_labels[0]

        self._true_labels[0].clear()
        self._true_labels[0].push_back(0)

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
                self._true_labels[0].push_back(next_label)
                labels[g] = next_label
                self.peaks[next_label] = density
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
                        ##     NewSaddle(g, Df[g], m1, m2)
                        ##     )
                        self.saddles.append( NewSaddle(g, Df[g], m1, m2) )
                        m1.neighbors.add(label_2)
                        m2.neighbors.add(label_1)
                labels[g] = -2 * tallest_neighbor_label #?
        print next_label-1
        self.labels = labels

