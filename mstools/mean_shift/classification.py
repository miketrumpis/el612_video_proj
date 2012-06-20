import numpy as np
import itertools

from histogram import nearest_cell_idx, normalized_feature
from grid_mean_shift import multilinear_interpolation
from ..util import image_to_features, cluster_size_filter

def classify_sequence(classifier, vid):
    seg_vid = np.zeros(vid.shape[:3], 'i')
    for n in xrange(len(vid)):
        seg_vid[n] = classifier.classify(
            vid[n], refined=True, cluster_size_threshold=50
            )
    return seg_vid

class PixelModeClassifier(object):
    """
    Classifies an image based on a labeling manifold calculated
    over a regularly spaced grid of the feature space. Where the
    labels are unknown, perform a Mean Shift approximation with
    the quasi-mean and density manifold calculated on the same grid.
    The Mean Shift sequence should follow a path to a labeled mode.
    """
    
    def __init__(
            self, labels, mud_grid, cell_edges,
            boundary = -1, spatial = True
            ):
        # mud = mean vector (\mu) and density (d) concatenated over
        # feature space grid, which is defined by cell edges
        self.labels = labels
        self.mud_grid = mud_grid
        self.cell_edges = cell_edges
        self.boundary = boundary
        self.spatial = spatial

    def refine_labels(self):
        resolve_label_boundaries(
            self.labels, self.mud_grid, self.cell_edges, max_iter = 20
            )

    def classify(self, image, refined=True, cluster_size_threshold=0):
        initial = segment_image_from_labeled_modes(
            image, self.labels, self.cell_edges,
            spatial_features = self.spatial
            )
        if not refined:
            if cluster_size_threshold:
                cluster_size_filter(initial, cluster_size_threshold)
            return initial
        features = image_to_features(image)
        if not self.spatial:
            features = features[:,2:]
        resolve_segmentation_boundaries(
            initial, features, self.mud_grid, self.labels,
            self.cell_edges, max_iter = 10
            )
        if cluster_size_threshold:
            cluster_size_filter(initial, cluster_size_threshold)
        return initial

def cell_neighbors(c_idx, dims):
    """
    Returns (valid) ND-indices of the edgewise neighbors of the cell at
    index c_idx. This Python/Numpy implementation is reasonably fast,
    but still too slow to be used within a significant inner loop.
    """
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

def segment_image_from_labeled_modes(
        image, labels, cell_edges, spatial_features=True
        ):
    # for each pixel in the image, look up the (x,y,c1,c2,c3) feature in
    # the labels function -- this is just a matter of downsampling 
    # (or dividing the indices)

    # features correspond to pixels in row-major order, 
    # so we can just zip the whole
    # course-grid feature indices into a flat list
    features = image_to_features(image)
    if not spatial_features:
        features = features[:,2:]
    cell_idc = nearest_cell_idx(features, cell_edges)
    
    flat_idc = np.lib.index_tricks.ravel_multi_index(
        cell_idc.T, labels.shape
        )
    seg_img = labels.flat[flat_idc]
    seg_img.shape = image.shape[:2]
    return seg_img

def resolve_segmentation_boundaries(
        seg_image, features, grid, labels, cell_edges, max_iter = 20
        ):
    # NOTE: grid is now a ND manifold whose
    # range concatenates grid means and density
    bpts = np.where(seg_image<=0)
    ishape = seg_image.shape

    k = 1
    n_pt = len(bpts[0])
    next_pct = 10

    idx = np.zeros((features.shape[-1],), 'l')

    ## max_idx = np.array(grid.shape[:len(idx)], 'l')
    ## max_idx -= 1

    gdims = grid.shape[:-1]
    prev_ms = np.zeros((len(gdims),), 'd')
    unit_edges = [np.arange(d+1, dtype='d') for d in gdims]
    for y, x in itertools.izip(*bpts):
        pct = int(100*(k/float(n_pt)))
        k += 1
        if pct==next_pct:
            next_pct += 10
            print pct, '%'
        fvec = normalized_feature(features[y*ishape[1] + x], cell_edges)
        assigned = False
        n_iter = 0
        prev_ms[:] = 0
        while not assigned and n_iter < max_iter:
            mean = multilinear_interpolation(fvec, grid)
            density = mean[-1]
            ## print n_iter, density
            mean = mean[:-1]
            if density > 1e-4:
                mean /= density
            else:
                assigned = True
            ms = mean - fvec
            ## fvec = mean
            cs = np.dot(ms, prev_ms)
            if cs < 0:
                ms -= prev_ms * cs
            norm_sq = np.dot(ms, ms)
            if norm_sq > 1e-8:
                if norm_sq < (0.2**2):
                    ms *= 0.2/np.sqrt(norm_sq)
                fvec += ms
                prev_ms = ms/np.sqrt(norm_sq)
            else:
                assigned = True
        
            idx = nearest_cell_idx(fvec[None,:], unit_edges)[0]
            label = labels[tuple(idx)]
            if label > 0:
                seg_image[y,x] = label
                assigned = True
            n_iter += 1
        ## print 'label', label, 'after', n_iter, 'iters'
    return

def resolve_label_boundaries(
        labels, grid, cell_edges, max_iter = 20
        ):
    """
    For each boundary point in labels, perform a gradient ascent using
    the mean shift sequence. Terminate the walk once the sequence enters
    a region already marked as belonging to a mode.

    Parameters
    ----------

    labels: ndarray
      label map of the histogram grid points
    p: ndarray
      density over grid
    cell_edges: sequence
      bin edges of the histogram (needed for grid index conversion)
    features: ndarray
      the feature vectors of the original image
    sigma: float
      kernel bandwidth

    """
    # the boundary points should be ordered by decreasing density
    ## boundary_pts = np.where(labels.ravel()==boundary)[0]
    boundary_pts = np.where(labels.ravel()<0)[0]
    p = grid[...,-1]
    order = np.argsort(p.flat[boundary_pts])[::-1]
    ## order = np.argsort(p.flat[boundary_pts])[::-1]
    boundary_pts = np.unravel_index(boundary_pts[order], labels.shape)
    ## grid = np.concatenate( (mu, p[...,None]), axis=-1).copy()
    ## boundary_pts = np.unravel_index(boundary_pts, labels.shape)
    gdims = grid.shape[:-1]
    prev_ms = np.zeros((len(gdims),), 'd')
    unit_edges = [np.arange(d+1, dtype='d') for d in gdims]
    for x in itertools.izip(*boundary_pts):
        assigned = False
        fvec = np.array(x, 'd') + 0.5
        n_iter = 0
        prev_ms[:] = 0
        while not assigned and n_iter < max_iter:
            mean = multilinear_interpolation(fvec, grid)
            density = mean[-1]
            mean = mean[:-1]
            if density > 1e-4:
                mean /= density
            else:
                assigned = True
            ms = mean - fvec
            cs = np.dot(ms, prev_ms)
            if cs < 0:
                ms -= prev_ms * cs
            norm_sq = np.dot(ms, ms)
            if norm_sq > 1e-8:
                if norm_sq < (0.2**2):
                    ms *= 0.2/np.sqrt(norm_sq)
                fvec += ms
                prev_ms = ms/np.sqrt(norm_sq)
            else:
                assigned = True
            
            idx = nearest_cell_idx(fvec[None,:], unit_edges)[0]
            label = labels[tuple(idx)]
            if label > 0:
                labels[x] = label
                assigned = True
            n_iter += 1
        ## print 'label', label, 'after', n_iter, 'iters'
