import numpy as np
import scipy.ndimage as ndimage
## from cell_labels import assign_modes_by_density

def image_to_features(image):
    Ny, Nx = image.shape[:-1]
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    n_color_dim = 1 if len(image.shape) < 3 else 3
    features = np.c_[xx.ravel(), yy.ravel(), image.reshape(Ny*Nx, n_color_dim)]
    return features

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def histogram(img, grid_spacing):
    """
    Estimate the ND probability density of an image by histogram over an
    regularly spaced grid.

    Parameters
    ----------

    img: ndarray 2- or 5-D
      A [0,255] quantized image shaped (Y,X) (grayscale),
      or (Y,X) + (<color dims>)

    grid_spacing: int
      The grid cell edge length. Color quantization range and image pixel
      range are treated equally in partitioning the density space.

    Returns
    -------

    p: ndarray, 3- or 5-D
      The induced density function over spatio-intensity space (X,Y,I),
      or spatio-color space (X,Y,<color dims>)

    """
    xy_shape = img.shape[:2][::-1]
    n_color_dim = 1 if len(img.shape) < 3 else 3
    im_vec = image_to_features(img)
    color_range = tuple(im_vec[:,2:].ptp(axis=0) + 1)
    # remember im_vec's feature vectors are (x, y, <colors>)
    bins = map(
        lambda x: int(np.floor(float(x)/grid_spacing)),
        xy_shape + color_range
        )
    edges = []
    minima = (0, 0) + tuple(im_vec[:,2:].min(axis=0))
    for n, d, mn in zip(bins, xy_shape + color_range, minima):
        r = d - n*grid_spacing
        edges.append( np.arange(mn, (n+1)*grid_spacing, grid_spacing) + r/2. )
    p, x = np.histogramdd(
        im_vec, bins=edges
        )
    cell_centers = []
    for edge in x:
        cell_centers.append( (edge[:-1] + edge[1:])/2 )
    return p, cell_centers, edges

def smooth_density(p, sigma):
    return ndimage.gaussian_filter(p, sigma, mode='constant')

def cell_neighbors(c_idx, dims):
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

def assign_modes_by_density(D, boundary=-1, cluster_tol=1e-1):
    dims = D.shape
    D = D.ravel()
    gs = np.argsort(D)[::-1]
    labels = np.zeros(gs.shape, 'i')
    next_label = 1
    modes = list()
    for g in gs:
        nb_idx = cell_neighbors(g, dims)
        nbs = labels[nb_idx]
        pos_nbs = nbs[nbs>0]
        # if there are no modes assigned to this group
        if len(pos_nbs)==0:
            # assign a new mode
            labels[g] = next_label
            next_label += 1
            modes.append( (g, D[g]) )
            continue
        # if neighborhood is consistent, then removing the bias will
        # leave nothing behind
        test = pos_nbs[0]
        if np.sum(pos_nbs - test) == 0:
            labels[g] = test
            continue
        # if the non-negative neighboring points are mixed modal, then
        # mark down as a boundary        
        labels[g] = boundary
    return labels.reshape(dims), next_label-1, np.array(modes)

# should probably Cython this
def nearest_cell_idx(x, cell_edges):
    # cells is a d-length list of axis coordinates corresponding
    # to d-dimensional cell edges
    nd = len(cells)
    g_idx = np.empty(x.shape, 'i')
    for d in xrange(nd):
        g_axis = cells[d]
        g_spacing = g_axis[1] - g_axis[0]
        idx = (x[d]-g_axis[0])/g_spacing
        g_idx[d] = np.clip(idx.astype('i'), 0, len(g_axis)-2)
    return g_idx

def segment_image_from_labels(image, labels, cell_locs, boundary=-1):
    # for each pixel in the image, look up the (x,y,r,g,b) feature in
    # the labels function -- this is just a matter of downsampling 
    # (or dividing the indices)

    # features correspond to pixels in row-major order, 
    # so we can just zip the whole
    # course-grid feature indices into a flat list
    features = image_to_features(image)
    cell_idc = nearest_cell_idx(features.T, cell_locs)
    
    flat_idc = np.lib.index_tricks.ravel_multi_index(
        cell_idc, labels.shape
        )
    seg_img = labels.flat[flat_idc]
    seg_img.shape = image.shape[:2]
    return seg_img

if __name__=='__main__':
    import PIL.Image as PImage
    img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/100007.jpg')
    a_img = np.array(img)
    p = histogram(a_img, 40)
    p = smooth_density(p, 1)