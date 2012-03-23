import numpy as np
import scipy.ndimage as ndimage

from cell_labels import Saddle
from histogram import nearest_cell_idx
from ..util import image_to_features

def smooth_density(p, sigma, accum=None, **gaussian_kws):
    p = ndimage.gaussian_filter(p, sigma, **gaussian_kws)
    if accum is not None:
        # filter vector valued function accum coordinate-wise
        n_coords = accum.shape[-1]
        for n in xrange(n_coords):
            accum[...,n] = ndimage.gaussian_filter(
                accum[...,n], sigma, **gaussian_kws
                )
        return p, accum
    return p

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
    features = image_to_features(image)
    if not spatial_features:
        features = features[:,2:]
    cell_idc = nearest_cell_idx(features, cell_locs)
    
    flat_idc = np.lib.index_tricks.ravel_multi_index(
        cell_idc.T, labels.shape
        )
    seg_img = labels.flat[flat_idc]
    seg_img.shape = image.shape[:2]
    return seg_img

# XXX! Still a problem here!
def merge_persistent_modes(labels, saddles, clusters, peaks, thresh):
    # saddles must be sorted by descending height! as in the output
    # of assign_modes_...
    n_labels = labels.copy()
    n_labels = np.zeros_like(labels)
    n_saddles = list()
    n_clusters = dict()
    n_peaks = dict()
    assignments = dict()
    # make a separate mapping for saddle points that disappear
    accumulated_saddles = dict( ((m,[]) for m in clusters.keys()) )
    bordering_saddles = dict( ((m,[]) for m in clusters.keys()) )
    sub_clusters = dict( ((m,[]) for m in clusters.keys()) )
    for saddle in saddles:
        h = saddle.elevation
        # filter the saddle neighbors for any reassignments
        nbs = [assignments.get(m, m) for m in saddle.neighbors]
        unique_nbs = np.unique(nbs)
        if len(unique_nbs)==1:
            # this saddle point has been accumulated and it doesn't
            # even know it yet
            accumulated_saddles[unique_nbs[0]].append(saddle.idx)
            continue
        # the mode labels sorted by increasing height
        nb_modes = sorted(unique_nbs, key=peaks.get)
        # the peaks of the sorted neighboring modes
        nb_sub_peaks = np.array([peaks[m] for m in nb_modes[:-1]])
        # if any of the n-1 peaks lower than the highest neighbor
        # are dominated by the highest neighbor, then merge clusters
        sm_idc = np.where((nb_sub_peaks - h) < thresh)[0]
        sub_modes = [nb_modes[m] for m in sm_idc]
        if not len(sub_modes):
            # still a saddle, but possibly with new neighbors
            new_saddle = Saddle(saddle.idx, h, list(unique_nbs))
            n_saddles.append(new_saddle)
            continue
        dom = nb_modes[-1]
        for m in sub_modes:
            assignments[m] = dom
            # reassign sub clusters of mode m
            for subsub in sub_clusters[m]:
                assignments[subsub] = dom
                sub_clusters[dom].append(subsub)
            # release m from the dominant set
            sub_clusters.pop(m)
            # update the bordering saddles
            bordering_saddles[dom].extend(bordering_saddles[m])
            bordering_saddles.pop(m)
            # update the consumed saddles
            accumulated_saddles[dom].extend(accumulated_saddles[m])
            accumulated_saddles.pop(m)
            # finally claim the sub mode
            sub_clusters[dom].append(m)
        # if the number of sub_modes is the number of original neighbors
        # minus one, then this is no longer a saddle (all neighboring
        # clusters were merged) -- In this case, then add the saddle
        # index to the dominant mode cluster
        # NOTE -- if any peaks survive persistence merging at this
        # elevation, then they will never be merged at lower elevations,
        # so the neighboring mode labeling is safe and consistent.
        # However, the currently dominant mode may disappear
        if len(sub_modes) < len(nb_modes)-1:
            new_nbs = [m for m in nb_modes if m not in sub_modes]
            new_saddle = Saddle(
                saddle.idx, h, new_nbs
                )
            n_saddles.append(new_saddle)
            ## print dom, bordering_saddles[dom]
            bordering_saddles[dom].append(new_saddle)
            ## print dom, bordering_saddles[dom]
        else:
            accumulated_saddles[dom].append(saddle.idx)
    ## print assignments
    survivors = list( set(clusters.keys()).difference(assignments.keys()) )
    ## print survivors
    ## return survivors, sub_clusters, bordering_saddles, accumulated_saddles, n_saddles
    for s in survivors:
        new_cluster = clusters[s][:]
        if sub_clusters.get(s,[]):
            sub_idc = reduce(lambda x, y: x+y,
                             (clusters[sub] for sub in sub_clusters[s]))
            new_cluster.extend(sub_idc)
        new_cluster.extend(accumulated_saddles.get(s,[]))
        new_cluster = sorted(new_cluster)
        n_clusters[s] = new_cluster
        np.put(n_labels, new_cluster, s)
        n_peaks[s] = peaks[s]
    for mode, saddles in bordering_saddles.items():
        for s in saddles:
            s.neighbors.append(mode)
    # finally, mark -1 for all the remaining saddles
    s_cluster = [s.idx for s in n_saddles]
    np.put(n_labels, s_cluster, -1)
    # NOTE: probably don't need to return n_peaks -- this is easily
    # found from the old peaks dictionary and the keys in n_clusters
    return n_labels, n_clusters, n_peaks, n_saddles
            
