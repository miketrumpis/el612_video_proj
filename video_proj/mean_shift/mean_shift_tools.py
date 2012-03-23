# NOTE! Need to make sigma arguments work for mixed spatial/color bandwidths

from __future__ import division
import numpy as np
import itertools
import scipy.ndimage as ndimage

from histogram import nearest_cell_idx, normalized_feature
from cell_labels import Saddle
from grid_mean_shift import multilinear_interpolation

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

def resolve_segmentation_boundaries(
        seg_image, features, grid, labels, cell_edges, 
        boundary = -1, max_iter = 20
        ):
    # NOTE: grid is now a ND manifold whose
    # range concatenates grid means and density
    bpts = np.where(seg_image==boundary)
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
            
            ## np.clip(fvec.astype('l'), 0, max_idx, idx)
            idx = nearest_cell_idx(fvec[None,:], unit_edges)[0]
            label = labels[tuple(idx)]
            if label != boundary:
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
    boundary_pts = np.where(labels.ravel()==boundary)[0]
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
            mean = mean[:-1]/density
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
            if cx_label != boundary:
                labels[x] = cx_label
                walking = False
            n_iter += 1
        ## print 'label', cx_label, 'after', n_iter, 'iters'

        
        
        
