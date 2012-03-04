import numpy as np
import itertools
import scipy.ndimage as ndimage

import density_est

def mean_shift_update(y, x, sigma):
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
    ## keep = dx<(3*sigma)**2
    ## dx = dx[keep]
    ## x = x[keep]
    dx /= -(2*sigma**2)
    weights = np.exp(dx) # length-N
    return (weights[:,None]*x).sum(axis=0) / weights.sum()

def mean_shift_update2(y, xgrid, density):
    """

    Parameters
    ----------

    y: ndarray
      d-dimensional point in feature space
    xgrid: ndarray
      a regular d-dimensional grid of points over which a density is estimated
    density: ndarray
      the unnormalized 'density' (bin count) associated with each point
      in xgrid
    """
      
    # Note -- instead of using all of the original feature vector samples,
    # the weights can be approximated using the histogram count over a
    # regular grid. In this form, the weights are the values of Gaussians
    # at the grid points, scaled by the bin counts at those points.
    #
    # xgrid is (N, d) and y is (d,)
    # the bandwidth of the kernel is built into the grid edge length
    dx = np.power( y - xgrid, 2).sum(axis=1)
    dx /= -2
    weights = density * np.exp(dx)
    return (weights[:,None]*xgrid).sum(axis=0) / weights.sum()
    
    
    
def walk_uphill(labels, p, cell_edges, features, sigma, boundary=-1):
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
    max_iter = 20
    for x in itertools.izip(*boundary_pts):
        walking = True
        f_idx = np.lib.index_tricks.ravel_multi_index(x, labels.shape)
        # what is the correspondence between (i,j,k) in labels index
        # and the approximate feature vector? guess we need lookup
        y = np.array([ (d[k] + d[k+1])/2.0 for d,k in zip(cell_edges, x) ])
        n = 0
        while walking:
            # can this be scaled to unit variance per grid length?
            # yes, but it would not correspond to a valid grid index
            # unless the cell edges begin at zero
            y = mean_shift_update(y, features, 2*sigma)
            cx = density_est.nearest_cell_idx(y, cell_edges)
            cx_label = labels[tuple(cx)]
            if cx_label != boundary:
                labels[x] = cx_label
                walking = False
            n += 1
            if n >= max_iter:
                walking = False
        ## print 'label', cx_label, 'after', n, 'iters'

## def walk_uphill2(labels, density, bin_count, boundary=-1):
##     """
##     For each boundary point in labels, perform a gradient ascent using
##     the mean shift sequence. Terminate the walk once the sequence enters
##     a region already marked as belonging to a mode.

##     Parameters
##     ----------

##     labels: ndarray
##       label map of the histogram grid points
##     density: ndarray
##       density map of the histogram grid points
##     bin_count: ndarray
##       bin count map of the histogram grid points

##     Notes
##     -----

##     If the grid edges are uniform, it should be possible to perform this
##     walk agnostic of the cell locations
##     """
##     dims = density.shape
##     if dims != labels.shape:
##         raise ValueError('Shape of labels map does not match density map')

##     xgrid = np.indices(dims, dtype='d')
##     xgrid += 0.5
##     # form x*b(x) function and smooth each coordinate separately
##     x_weights = xgrid*bin_count
##     for k in xrange(len(dims)):
##         x_weights[k] = ndimage.gaussian_filter(
##             x_weights[k], 1.0, mode='constant'
##             )
##     xgrid = xgrid.reshape(len(dims), -1).T
##     idx_max = np.array(dims) - 1
##     boundary_pts = np.where(labels==boundary)
##     max_iter = 20
##     for x in itertools.izip(*boundary_pts):
##         walking = True
##         f_idx = np.lib.index_tricks.ravel_multi_index(x, labels.shape)
##         y = xgrid[f_idx]
##         n = 0
##         while walking:
##             ## y = mean_shift_update2(y, xgrid, density)
##             y = x_weights[(slice(None),)+tuple(y)] / density[tuple(y)]
##             cx = np.clip(np.round(y).astype('i'), 0, idx_max)
##             cx_label = labels[tuple(cx)]
##             if cx_label != boundary:
##                 labels[x] = cx_label
##                 walking = False
##             n += 1
##             if n >= max_iter:
##                 walking = False
##         print 'label', cx_label, 'after', n, 'iters'
        
        
        
