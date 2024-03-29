# cython: profile=True
""" -*- python -*- file
"""
import numpy as np
cimport numpy as np
cimport cython
from mstools.indexing cimport idx_type

from ..util import image_to_features
from ..indexing import py_idx_type as pidx_t

def normalized_feature(f, list cell_edges):
    if len(f.shape) < 2:
        return nearest_cell_idx(f[None,:], cell_edges, real_idx=True)[0]
    return nearest_cell_idx(f, cell_edges, real_idx=True)

# XXX: why isn't this completely vectorized in Numpy? This is just
# a 1D affine transformation of each coordinate
@cython.boundscheck(False)
def nearest_cell_idx(
        np.ndarray[np.float64_t, ndim=2] x,
        list cell_edges,
        safe_idx=True,
        real_idx=False
        ):
    """
    Find the nearest neighbor index according to the cell edges,
    for each vectors in x

    Parameters
    ----------

    x: ndarray (nvec, ndim)
      Matrix of nvec vectors, each of ndim dimensions

    cell_edges: list
      ndim-length list of cell edges for some grid space

    safe_idx: bool
      For any coordinates that do not fall within a cell interval, use
      the nearest cell. If False, mark any out of bounds coordinates as -1

    real_idx: bool
      If true, return the real valued "fractional" index corresponding
      to each feature. In other words, this is the normalized feature
      vector.

    Returns
    -------

    g_idx: ndarray (nvec, ndim)
      Nearest-neighbor grid indices
    """
    # cell_edges is a d-length list of axis coordinates corresponding
    # to d-dimensional cell edges
    cdef int d, r, g_max
    cdef double f_idx
    cdef int nd = x.shape[1]
    cdef int nr = x.shape[0]
    cdef np.ndarray[idx_type, ndim=2] g_idx
    cdef np.ndarray[np.float64_t, ndim=2] r_idx
    cdef np.ndarray[np.float64_t, ndim=1] g_axis
    cdef np.float64_t g_edge, g_spacing
    if real_idx:
        r_idx = np.empty((nr, nd), dtype='d')
    else:
        g_idx = np.empty((nr, nd), dtype=pidx_t)
    for d in range(nd):
        g_axis = cell_edges[d]
        g_edge = g_axis[0]
        g_max = len(g_axis)-2
        g_spacing = g_axis[1] - g_axis[0]
        for r in range(nr):
            f_idx = ( (x[r,d]-g_edge)/g_spacing )
            if real_idx:
                r_idx[r,d] = f_idx
            else:
                if f_idx < 0:
                    g_idx[r,d] = 0 if safe_idx else -1
                elif f_idx >= g_max+1:
                    g_idx[r,d] = g_max if safe_idx else -1
                else:
                    g_idx[r,d] = <idx_type> f_idx
    if real_idx:
        return r_idx
    else:
        return g_idx

@cython.boundscheck(False)
def histogram(
        image, grid_spacing, spatial_features=True, color_spacing=None
        ):
    """
    Estimate the ND probability density of an image's features by
    histogram over a regularly spaced grid.

    Parameters
    ----------

    img: ndarray 2- or 5-D
      A quantized image shaped (Y,X) (grayscale), or (Y,X) + (<color dims>)

    grid_spacing: float
      The grid cell edge length. Color quantization range and image pixel
      range are treated equally in partitioning the density space. For
      alternative behavior, see color_spacing.

    spatial_features: bool
      Whether or not to treat the (x,y) coordinates of the image as
      coordinates in the feature space (default True).

    color_spacing: float
      If not None, then use this grid spacing on the color feature
      dimensions.

    Returns
    -------

    b: ndarray, 1-, 3- or 5-D
      The feature vector count in a regularly gridded field of cells (bins).
      The dimensionality may be: intensity only (1D); colors only (3D);
      spatio-intensity (3D); spatio-color (5D).

    accum: ndarray, 1-, 3- or 5-D
      The summed feature vectors discovered at each bin.

    edges: sequence
      The list of cell edges that defines the field of cells

    """
    if not color_spacing:
        color_spacing = grid_spacing
    grid_spacing, color_spacing = map(float, (grid_spacing, color_spacing))
    xy_shape = image.shape[:2][::-1] if spatial_features else ()
    n_color_dim = 1 if len(image.shape) < 3 else 3
    im_vec = image_to_features(image)
    color_range = tuple(im_vec[:,2:].ptp(axis=0))
    sigma = (grid_spacing,)*len(xy_shape) + (color_spacing,)*n_color_dim
    # remember im_vec's feature vectors are (x, y, <colors>)
    bins = map(
        lambda x: int(np.floor(x[0]/x[1])), zip(xy_shape+color_range, sigma)
        )
    edges = []
    color_minima = tuple(im_vec[:,2:].min(axis=0))
    minima = (0, 0) + color_minima if spatial_features else color_minima
    for n, d, mn, s in zip(bins, xy_shape + color_range, minima, sigma):
        r = d - n*s
        edges.append( np.linspace(mn, mn+n*s, n+1) + r/2. )
    # do cell center locations
    cell_centers = []
    for edge in edges:
        cell_centers.append( (edge[:-1] + edge[1:])/2 )

    cdef np.ndarray[np.float64_t, ndim=2] features = \
        im_vec if spatial_features else im_vec[:,2:]
    cdef np.ndarray[np.float64_t, ndim=2] nf = \
        normalized_feature(features, edges)
    cdef np.ndarray[np.int64_t, ndim=2] bin_idc = \
        nearest_cell_idx(features, edges) #, safe_idx=False)
    # Out of bounds indices are marked by -1 -- multiply coordinates
    # to find bad feature vectors
    in_bounds_features = (np.prod(bin_idc, axis=1) >= 0)
    bin_idc = bin_idc[in_bounds_features]
    features = features[in_bounds_features]

    cdef int n_feat = features.shape[0]
    cdef int f_dims = features.shape[1]
    cdef int k, m, bin

    # accumulate hits in a flattened array with row-major ordering
    cdef np.ndarray[np.float64_t, ndim=1] b = \
        np.zeros( (np.prod(bins),), 'd' )
    # accumulate summed feature vectors similarly
    cdef np.ndarray[np.float64_t, ndim=2] accum = \
        np.zeros( (np.prod(bins), f_dims), 'd' )
    # define the strides for flat indexing
    cdef np.ndarray[np.int64_t, ndim=1] strides = \
        np.cumprod(np.array(bins[::-1]))

    for k in range(n_feat):
        bin = bin_idc[k,f_dims-1]
        for m in range(f_dims-1):
            bin += bin_idc[k,f_dims-2-m]*strides[m]
        b[bin] += 1
        for m in range(f_dims):
            accum[bin,m] += nf[k,m]

    return (b.reshape(tuple(bins)),
            accum.reshape(tuple(bins)+(f_dims,)),
            edges)
