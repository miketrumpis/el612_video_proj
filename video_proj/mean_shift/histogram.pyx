""" -*- python -*- file
"""
import numpy as np
cimport numpy as np
cimport cython

import density_est

@cython.boundscheck(False)
def nearest_cell_idx(
        np.ndarray[np.float64_t, ndim=2] x,
        cell_edges
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

    Returns
    -------

    g_idx: ndarray (nvec, ndim)
      Nearest-neighbor grid indices
    """
    # cell_edges is a d-length list of axis coordinates corresponding
    # to d-dimensional cell edges
    cdef int d, r, idx, g_max
    cdef int nd = x.shape[1]
    cdef int nr = x.shape[0]
    cdef np.ndarray[np.int64_t, ndim=2] g_idx = np.empty((nr, nd), 'l')
    cdef np.float64_t g_edge, g_spacing
    for d in range(nd):
        g_axis = cell_edges[d]
        g_edge = g_axis[0]
        g_max = len(g_axis)-2
        g_spacing = g_axis[1] - g_axis[0]
        for r in range(nr):
            idx = <int> ( (x[r,d]-g_edge)/g_spacing )
            if idx < 0:
                g_idx[r,d] = 0
            elif idx > g_max:
                g_idx[r,d] = g_max
            else:
                g_idx[r,d] = idx
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

    b: ndarray, 3- or 5-D
      The induced (unnormalized) density function over spatio-intensity
      space (X,Y,I), or spatio-color space (X,Y,<color dims>)

    """
    if not color_spacing:
        color_spacing = grid_spacing
    grid_spacing, color_spacing = map(float, (grid_spacing, color_spacing))
    xy_shape = image.shape[:2][::-1] if spatial_features else ()
    n_color_dim = 1 if len(image.shape) < 3 else 3
    im_vec = density_est.image_to_features(image)
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

    features = im_vec if spatial_features else im_vec[:,2:]
    cdef np.ndarray[np.int64_t, ndim=2] bin_idc = \
        nearest_cell_idx(features, edges)

    # accumulate hits in a flattened array with row-major ordering
    cdef np.ndarray[np.float64_t, ndim=1] b = \
        np.zeros( (np.prod(bins),), 'd' )
    cdef int k, bin, dim
    cdef int n_feat = bin_idc.shape[0]
    cdef int f_dims = bin_idc.shape[1]
    cdef np.ndarray[np.int64_t, ndim=1] strides = \
        np.cumprod(np.array(bins[::-1]))
    for k in range(n_feat):
        bin = bin_idc[k,f_dims-1]
        for dim in range(f_dims-1):
            bin += bin_idc[k,f_dims-2-dim]*strides[dim]
        b[bin] += 1
    
    return b.reshape(tuple(bins)), cell_centers, edges
