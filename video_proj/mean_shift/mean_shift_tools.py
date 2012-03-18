import numpy as np
import itertools
import scipy.ndimage as ndimage
from sklearn.ball_tree import BallTree

from histogram import nearest_cell_idx

def gaussian_mean_shift_update(y, x, sigma):
    """
    Generate the next value in the mean-shift sequence

    Parameters
    ----------

    y: ndarray
      previous mean shift point
    x: ndarray
      sample of feature vectors from which density estimate is found
    
    """
    # x is (N, d) where d is the dimensionality of the feature space
    # y is (d,)
    dx = np.power(y - x, 2).sum(axis=1)
    dx /= -(2*sigma**2)
    weights = np.exp(dx) # length-N
    return (weights[:,None]*x).sum(axis=0) / weights.sum()

def resolve_segmentation_boundaries(
        seg_image, features, labels, cell_edges, sigma,
        boundary = -1, max_iter = 300
        ):
    bpts = np.where(seg_image==boundary)
    ishape = seg_image.shape
    btree = BallTree(features, p=2)
    for y, x in zip(*bpts):
        fvec = features[y*ishape[1] + x]
        assigned = False
        n_iter = 0
        while not assigned and n_iter < max_iter:
            sigma_ball_idc = btree.query_radius(fvec, sigma)[0]
            sigma_ball = features[sigma_ball_idc]
            ## fvec = gaussian_mean_shift_update(fvec, sigma_ball, sigma)
            # alternative mean shift kernel is flat (uniform)
            fvec = np.mean(sigma_ball, axis=0)
            nn_idx = nearest_cell_idx(fvec[None,:], cell_edges)[0]
            label = labels[tuple(nn_idx)]
            if label != boundary:
                seg_image[y,x] = label
                assigned = True
            n_iter += 1
        ## print 'label', label, 'after', n_iter, 'iters'
    return

def resolve_label_boundaries(
        labels, p, cell_edges, features, sigma, boundary = -1, max_iter = 300
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
    boundary_pts = np.where(labels.ravel()==boundary)[0]
    order = np.argsort(p.flat[boundary_pts])[::-1]
    boundary_pts = np.unravel_index(boundary_pts[order], labels.shape)
    btree = BallTree(features, p=2)
    for x in itertools.izip(*boundary_pts):
        walking = True
        # what is the correspondence between (i,j,k) in labels index
        # and the approximate feature vector? guess we need lookup
        fvec = np.array([ (d[k] + d[k+1])/2.0 for d,k in zip(cell_edges, x) ])
        n_iter = 0
        while walking and n_iter < max_iter:
            sigma_ball_idc = btree.query_radius(fvec, sigma)[0]
            sigma_ball = features[sigma_ball_idc]
            ## fvec = gaussian_mean_shift_update(fvec, sigma_ball, sigma)
            fvec = np.mean(sigma_ball, axis=0)
            cx = nearest_cell_idx(fvec[None,:], cell_edges)[0]
            cx_label = labels[tuple(cx)]
            if cx_label != boundary:
                labels[x] = cx_label
                walking = False
            n_iter += 1
        print 'label', cx_label, 'after', n_iter, 'iters'

        
        
        
