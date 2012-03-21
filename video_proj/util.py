import numpy as np
import matplotlib.pyplot as pp
import matplotlib.animation as animation

import colors
from glob import glob

def image_to_features(image):
    """
    Convert image array into matrix of feature vectors, where each row
    contains a vector (x, y, c1, [c2, c3]), where ck are color values.
    """
    Ny, Nx = image.shape[:2]
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    n_color_dim = 1 if len(image.shape) < 3 else 3
    features = np.c_[xx.ravel(), yy.ravel(), image.reshape(Ny*Nx, n_color_dim)]
    return features

def contiguous_labels(labels, clusters):
    old_labels = clusters.keys()
    new_labels = dict(zip(range(1,len(old_labels)+1), old_labels))
    relabel = labels.copy()
    for nk, ok in new_labels.items():
        if nk == ok:
            continue
        old_cluster = clusters[ok]
        np.put(relabel, old_cluster, nk)
    return relabel

def yuv_image(b_arr, Nx, Ny, yuv_mode='420'):
    """
    Convert an array of bytes containing YUV information into an
    RGB frame with dimensions in shape.
    """
    if isinstance(b_arr, str):
        b_arr = open(b_arr, 'rb')
    if isinstance(b_arr, file):
        b_arr = np.fromfile(b_arr, dtype=np.uint8)
    if yuv_mode == '444':
        c_sub = 1
        raise NotImplementedError
    if yuv_mode in ('420', '422'):
        c_sub = 2
    else:
        raise NotImplementedError
    cbytes = Nx*Ny/c_sub**2
    y = b_arr[:Nx*Ny]
    u = b_arr[Nx*Ny:Nx*Ny+cbytes]
    v = b_arr[Nx*Ny+cbytes:]
    y.shape = (Ny, Nx); u.shape = (Ny/2, Nx/2); v.shape = (Ny/2, Nx/2)
    if c_sub > 1:
        u = np.repeat(u, c_sub, axis=0); u = np.repeat(u, c_sub, axis=1)
        v = np.repeat(v, c_sub, axis=0); v = np.repeat(v, c_sub, axis=1)
    image = np.c_[y.ravel(), u.ravel(), v.ravel()]
    rgb_image = colors.yuv2rgb(image).reshape(Ny,Nx,3)
    return rgb_image
    
def yuv_sequence(b_root, Nx, Ny, yuv_mode='420', t_range=()):
    b_files = glob(b_root)
    if t_range:
        b_files = b_files[t_range[0]:t_range[1]]
    vid = np.empty((len(b_files), Ny, Nx, 3), np.uint8)
    for t,f in enumerate(b_files):
        vid[t] = yuv_image(f, Nx, Ny)
    return vid

def animate_frames(frames, movie_name='', image_fov=None, fps=5):
    fig = pp.figure()
    ims = []
    for n, f in enumerate(frames):
        i = pp.imshow(f, extent=image_fov)
        x,_ = i.axes.get_xlim()
        y,_ = i.axes.get_ylim()
##         i.axes.text(x+1,y+1,'Frame %d'%(n+1,))
##         i.axes.set_title('Frame %d'%(n+1,))
        ims.append([i])
    ani = animation.ArtistAnimation(fig, ims)
    if movie_name:
        ani.save(movie_name+'.mp4', fps=fps)
    return ani

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
