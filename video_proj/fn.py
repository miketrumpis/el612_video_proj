import numpy as np
from mean_shift.histogram import nearest_cell_idx
from mean_shift.cell_labels import flatten_idx_passthru

class Function(object):

    def __init__(self, fn_grid, grid_edges):
        self.fn_grid = fn_grid
        self.edges = grid_edges
        self.nd = len(grid_edges)

    def __call__(self, x):
        if x.shape[0] == self.nd:
            # x is either (nd,) or (nd, nx), but should be (nx, nd)
            if len(x.shape) > 1:
                tpose = True
                x = x.T
            else:
                x = x[None,:]
        nn_idx = nearest_cell_idx(x, self.edges)
        fx = flatten_idx_passthru(nn_idx.T, self.fn_grid.shape)
        return np.take(self.fn_grid, fx)

        
        
