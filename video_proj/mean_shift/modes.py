import numpy as np
import scipy.ndimage as ndimage

from cell_labels import Saddle
from histogram import nearest_cell_idx
import ..util as ut

def smooth_density(p, sigma):
    ## return ndimage.gaussian_filter(p, sigma, mode='constant')
    return ndimage.gaussian_filter(p, sigma)

def cell_neighbors(c_idx, dims):
    """
    Returns (valid) ND-indices of the edgewise neighbors of the cell at
    index c_idx. This Python/Numpy implementation is reasonably fast,
    but still too slow to be used within a significant inner loop.
    """
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

def segment_image_from_labeled_modes(
        image, labels, cell_locs, boundary=-1, spatial_features=True
        ):
    # for each pixel in the image, look up the (x,y,r,g,b) feature in
    # the labels function -- this is just a matter of downsampling 
    # (or dividing the indices)

    # features correspond to pixels in row-major order, 
    # so we can just zip the whole
    # course-grid feature indices into a flat list
    features = ut.image_to_features(image)
    if not spatial_features:
        features = features[:,2:]
    cell_idc = nearest_cell_idx(features, cell_locs)
    
    flat_idc = np.lib.index_tricks.ravel_multi_index(
        cell_idc.T, labels.shape
        )
    seg_img = labels.flat[flat_idc]
    seg_img.shape = image.shape[:2]
    return seg_img

def merge_persistent_modes(labels, saddles, clusters, peaks, thresh):
    n_labels = labels.copy()
    n_saddles = list()
    n_clusters = dict()
    n_peaks = dict()
    assignments = dict()
    for saddle in saddles:
        h = saddle.elevation
        nb_modes = sorted(saddle.neighbors, key=peaks.get)
        # if any of the n-1 peaks lower than the highest neighbor
        # are dominated by the highest neighbor, then merge clusters
        nb_sub_peaks = np.array([peaks[m] for m in nb_modes[:-1]])
        sm_idc = np.where((nb_sub_peaks - h) < thresh)[0]
        sub_modes = [nb_modes[m] for m in sm_idc]
        if not len(sub_modes):
            # still a saddle
            n_saddles.append(saddle)
            continue
        dom = nb_modes[-1]
        # if the dominant neighbor already has a re-assignment,
        # then use that label instead
        if dom in assignments:
            dom = assignments[dom]
        for m in sub_modes:
            assignments[m] = dom
        # if the number of sub_modes is the number of original neighbors
        # minus one, then this is no longer a saddle (all neighboring
        # clusters were merged)
        # NOTE -- if any peaks survive persistence merging at this
        # elevation, then they will never be merged at lower elevations,
        # so the neighboring mode labeling is safe and consistent
        if len(sub_modes) < len(nb_modes)-1:
            new_nbs = [m for m in nb_modes if m not in sub_modes]
            new_saddle = Saddle(
                saddle.idx, h, new_nbs + [dom]
                )
            n_saddles.append(new_saddle)
    # create reverse assignment correspondence
    survivors = set(assignments.values())
    sub_modes = dict( ((s, []) for s in survivors) )
    for k, v in assignments.items():
        sub_modes[v].append(k)
    for s in survivors:
        sub_idc = reduce(lambda x, y: x+y,
                         (clusters[sub] for sub in sub_modes[s]))
        np.put(n_labels, sub_idc, s)
        n_clusters[s] = clusters[s]+sub_idc
        n_peaks[s] = peaks[s]

    # NOTE: probably don't need to return n_peaks -- this is easily
    # found from the old peaks dictionary and the keys in n_clusters
    return n_labels, n_clusters, n_peaks, n_saddles
            
