# NOTE! Need to make sigma arguments work for mixed spatial/color bandwidths

from __future__ import division
import numpy as np
import itertools
import scipy.ndimage as ndimage
from sklearn.neighbors import BallTree

from histogram import nearest_cell_idx
from cell_labels import Saddle

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
    if np.iterable(sigma):
        sigma = np.asarray(sigma)
    # x is (N, d) where d is the dimensionality of the feature space
    # y is (d,)
    dx = np.power(y - x, 2).sum(axis=1)
    dx /= -(2*sigma**2)
    weights = np.exp(dx) # length-N
    return (weights[:,None]*x).sum(axis=0) / weights.sum()

def quasi_elliptical_ball(ball_tree, x, s_sigma, c_sigma):
    if s_sigma > c_sigma:
        b_sigma = s_sigma
        e_sigma = c_sigma
        e_space = ( slice(None), slice(2,None) )
    else:
        b_sigma = c_sigma
        e_sigma = s_sigma
        e_space = ( slice(None), slice(0,2) )
    sigma_ball_idc = ball_tree.query_radius(x, b_sigma)[0]
    sigma_ball = ball_tree.data[sigma_ball_idc]
    # now within this smaller set, look for points where the
    # e-dim subspace distance is less than e_sigma
    subspace = sigma_ball[e_space]
    diffs = (subspace - x[e_space[1:]])**2
    idc = np.where( diffs.sum(axis=1) <= e_sigma**2 )[0]
    if not len(idc):
        return idc
    return sigma_ball[idc]

def resolve_segmentation_boundaries(
        seg_image, features, labels, cell_edges, sigma,
        boundary = -1, c_sigma = None, max_iter = 50
        ):
    bpts = np.where(seg_image==boundary)
    ishape = seg_image.shape
    btree = BallTree(features, p=2)
    use_ellipse = c_sigma is not None
    k = 1
    n_pt = len(bpts[0])
    next_pct = 10
    nr = 2
    ones = np.ones(nr).reshape(nr,1)
    for y, x in itertools.izip(*bpts):
        pct = int(100*(k/float(n_pt)))
        k += 1
        if pct==next_pct:
            next_pct += 10
            print pct, '%'
        fvec = features[y*ishape[1] + x]
        assigned = False
        n_iter = 0
        while not assigned and n_iter < max_iter:
            if use_ellipse:
                ball = quasi_elliptical_ball(btree, fvec, sigma, c_sigma)
            else:
                sigma_ball_idc = btree.query_radius(fvec, sigma)[0]
                ball = features[sigma_ball_idc]
            
            ## ball = fvec*ones #np.tile(fvec, 100).reshape(100,features.shape[1])
            if len(ball)==1:
                seg_image[y,x] = boundary
                assigned = True
                continue
            ## fvec = gaussian_mean_shift_update(fvec, ball, sigma)
            # alternative mean shift kernel is flat (uniform)
            fvec = np.mean(ball, axis=0)
            nn_idx = nearest_cell_idx(fvec[None,:], cell_edges)[0]
            label = labels[tuple(nn_idx)]
            if label != boundary:
                seg_image[y,x] = label
                assigned = True
            n_iter += 1
        ## print 'label', label, 'after', n_iter, 'iters'
    return

def resolve_label_boundaries(
        labels, p, cell_edges, features, sigma,
        boundary = -1, c_sigma = None, max_iter = 50
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
    use_ellipse = c_sigma is not None
    for x in itertools.izip(*boundary_pts):
        walking = True
        # what is the correspondence between (i,j,k) in labels index
        # and the approximate feature vector? guess we need lookup
        fvec = np.array([ (d[k] + d[k+1])/2.0 for d,k in zip(cell_edges, x) ])
        n_iter = 0
        while walking and n_iter < max_iter:
            if use_ellipse:
                ball = quasi_elliptical_ball(btree, fvec, sigma, c_sigma)
            else:
                sigma_ball_idc = btree.query_radius(fvec, sigma)[0]
                ball = features[sigma_ball_idc]
            ## fvec = gaussian_mean_shift_update(fvec, sigma_ball, sigma)
            if not len(ball):
                labels[x] = boundary
                walking = False
                continue
            fvec = np.mean(ball, axis=0)
            cx = nearest_cell_idx(fvec[None,:], cell_edges)[0]
            cx_label = labels[tuple(cx)]
            if cx_label != boundary:
                labels[x] = cx_label
                walking = False
            n_iter += 1
        ## print 'label', cx_label, 'after', n_iter, 'iters'

        
        
        
