import numpy as np
import matplotlib.pyplot as pp
import matplotlib.animation as animation

from _c_util import *

def image_to_features(image):
    """
    Convert image array into matrix of feature vectors, where each row
    contains a vector (x, y, c1, [c2, c3]), where ck are color values.
    """
    Ny, Nx = image.shape[:2]
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    n_color_dim = 1 if len(image.shape) < 3 else image.shape[2]
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

def animate_frames(frames, movie_name='', fps=5, **imshow_kw):
    fig = pp.figure()
    ims = []
    for n, f in enumerate(frames):
        i = pp.imshow(f, **imshow_kw)
        x,_ = i.axes.get_xlim()
        y,_ = i.axes.get_ylim()
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

class Bucket(object):
    """
    Ye Olde Bucket class
    """

    def __init__(self, *args, **kwargs):
        object.__init__(self)
        self.__dict__.update(kwargs)
        # in case this is instantiated by the call Bucket(somedict)
        # instead of Bucket(**somedict)
        if len(args)==1 and isinstance(args[0], dict):
            self.__dict__.update(args[0])

    
