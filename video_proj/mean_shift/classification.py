import numpy as np
import itertools

from histogram import nearest_cell_idx, normalized_feature
from grid_mean_shift import multilinear_interpolation
from ..util import image_to_features

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
        # this would refine the labeling array
        pass

    def classify(self, image, refined=True):
        initial = segment_image_from_labeled_modes(
            image, self.labels, self.cell_edges,
            spatial_features = self.spatial
            )
        if not refined:
            return initial
        features = image_to_features(image)
        if not self.spatial:
            features = features[:,2:]
        resolve_segmentation_boundaries(
            initial, features, self.mud_grid, self.labels,
            self.cell_edges, max_iter = 10
            )
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
        labels, p, mu, cell_edges,
        boundary = -1, max_iter = 20
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
    order = np.argsort(p.flat[boundary_pts])[::-1]
    boundary_pts = np.unravel_index(boundary_pts[order], labels.shape)
    grid = np.concatenate( (mu, p[...,None]), axis=-1).copy()
    gdims = grid.shape[:-1]
    prev_ms = np.zeros((len(gdims),), 'd')
    unit_edges = [np.arange(d+1, dtype='d') for d in gdims]
    for x in itertools.izip(*boundary_pts):
        walking = True
        fvec = np.array(x, 'd') + 0.5
        n_iter = 0
        # -- replace following with classifier built on
        #    multi-linear interpolation over the index grid
        #    and density manifold --
        while walking and n_iter < max_iter:
            ## if use_ellipse:
            ##     ball = quasi_elliptical_ball(btree, fvec, sigma, c_sigma)
            ## else:
            ##     sigma_ball_idc = btree.query_radius(fvec, sigma)[0]
            ##     ball = features[sigma_ball_idc]
            ## ## fvec = gaussian_mean_shift_update(fvec, sigma_ball, sigma)
            ## if not len(ball):
            ##     labels[x] = boundary
            ##     walking = False
            ##     continue
            ## fvec = np.mean(ball, axis=0)

            mean = multilinear_interpolation(fvec, grid)
            density = mean[-1]
            mean = mean[:-1]
            if density > 1e-8:
                mean /= density
            ms = mean - fvec
            cs = np.dot(ms, prev_ms)
            if cs < 0:
                ms -= prev_ms * cs
            norm_sq = np.dot(ms, ms)
            if norm_sq < (0.2**2):
                ms *= 0.2/np.sqrt(norm_sq)
            fvec += ms
            prev_ms = ms/np.sqrt(norm_sq)

            
            cx = nearest_cell_idx(fvec[None,:], unit_edges)[0]
            cx_label = labels[tuple(cx)]
            ## if cx_label != boundary:
            if cx_label > 0:
                labels[x] = cx_label
                walking = False
            n_iter += 1
        ## print 'label', cx_label, 'after', n_iter, 'iters'
