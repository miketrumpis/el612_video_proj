import numpy as np
from mean_shift.histogram import nearest_cell_idx, normalized_feature
from mean_shift.grid_mean_shift import multilinear_interpolation
from indexing import flatten_idc_p

class Function(object):

    def __init__(self, fn_grid, grid_edges, interp_order=0):
        self.fn_grid = fn_grid
        self.edges = grid_edges
        self.nd = len(grid_edges)
        self._order = interp_order

    def __call__(self, x):
        if x.shape[0] == self.nd:
            # x is either (nd,) or (nd, nx), but should be (nx, nd)
            if len(x.shape) > 1:
                ## tpose = True
                x = x.T
            else:
                x = x[None,:]
        if self._order == 0:
            nn_idx = nearest_cell_idx(x, self.edges)
            fx = flatten_idc_p(nn_idx, self.fn_grid.shape)
            return np.take(self.fn_grid, fx)
        elif self._order == 1:
            norm_idx = normalized_feature(x, self.edges)
            return multilinear_interpolation(norm_idx, self.fn_grid)
        else:
            raise NotImplementedError(
                'Only nearest neighbor and bilinear interp available'
                )

        
        
