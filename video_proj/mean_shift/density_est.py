import numpy as np
import scipy.ndimage as ndimage
import PIL.Image as PImage

def image_to_features(image):
    Ny, Nx = image.shape[:-1]
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    n_color_dim = 1 if len(image.shape) < 3 else 3
    features = np.c_[xx.ravel(), yy.ravel(), image.reshape(Ny*Nx, n_color_dim)]
    return features

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
    Ny, Nx = img.shape[:2]
    n_color_dim = 1 if len(img.shape) < 3 else 3
    im_vec = image_to_features(img)
    # make the bins accomodate a quick pidgeon-holing later on -- round up
    ybins, xbins, cbins = map(
        lambda x: int(np.ceil(float(x)/grid_spacing)),
        (Ny, Nx, 256)
        )    
    p, _ = np.histogramdd(
        im_vec, bins=(xbins, ybins) + (cbins,)*n_color_dim
        )
    return p

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

def assign_modes_by_density(D, boundary=-1):
    dims = D.shape
    D = D.ravel()
    gs = np.argsort(D)[::-1]
    labels = np.zeros(gs.shape, 'i')
    next_label = 1
    for g in gs:
        nb_idx = cell_neighbors(g, dims)
        nbs = labels[nb_idx]
        pos_nbs = nbs[nbs>0]
        # if there are no modes assigned to this group, assign a new one
        if len(pos_nbs)==0:
            labels[g] = next_label
            next_label += 1
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
    return labels.reshape(dims), next_label-1

def segment_image_from_labels(image, labels, grid_size, boundary=-1):
    # for each pixel in the image, look up the (x,y,r,g,b) feature in
    # the labels function -- this is just a matter of downsampling 
    # (or dividing the indices)

    # features correspond to pixels in row-major order, 
    # so we can just zip the whole
    # course-grid feature indices into a flat list
    course_features = image_to_features(image)/grid_size
    
    flat_idc = np.lib.index_tricks.ravel_multi_index(
        course_features.T, labels.shape
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
