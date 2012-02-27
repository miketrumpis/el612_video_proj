import numpy as np
import itertools

import density_est

def mean_shift_update(y, x, sigma):
    # x is (N, d) where d is the dimensionality of the feature space
    # y is (d,)
    dx = np.power(y - x, 2).sum(axis=1)
    keep = dx<(3*sigma)**2
    dx = dx[keep]
    x = x[keep]
    dx /= -(2*sigma**2)
    weights = np.exp(dx) # length-N
    return (weights[:,None]*x).sum(axis=0) / weights.sum()
    
def walk_uphill(labels, cell_edges, gx, sigma, boundary=-1):
    dims = labels.shape
    ## labels = labels.ravel()
    boundary_pts = np.where(labels==boundary)
    max_iter = 20
    for x in itertools.izip(*boundary_pts):
        walking = True
        f_idx = np.lib.index_tricks.ravel_multi_index(x, labels.shape)
        y = gx[f_idx]
        n = 0
        while walking:
            y = mean_shift_update(y, gx, sigma)
            cx = density_est.nearest_cell_idx(y, cell_edges)
            cx_label = labels[tuple(cx)]
            if cx_label != boundary:
                labels[x] = cx_label
                walking = False
            n += 1
            if n >= max_iter:
                walking = False
        print 'label', cx_label, 'after', n, 'iters'
        
        
        
