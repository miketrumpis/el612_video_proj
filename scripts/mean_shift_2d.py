import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.modes as modes
import video_proj.mean_shift.cell_labels as cell_labels
import video_proj.mean_shift.mean_shift_tools as ms_tools
import video_proj.colors as colors
import video_proj.util as ut

import PIL.Image as PImage
## ## # swan
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
# starfish
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## # monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')


img = np.array(img)

features = ut.image_to_features(img)
sigma = 3
gb_features = features[:,2:4]
color_range = tuple(gb_features.ptp(axis=0) + 1)
bins = map(lambda x: int(np.floor(float(x)/sigma)), color_range)
minima = tuple(gb_features.min(axis=0))
rg = color_range[0] - bins[0]*sigma; rb = color_range[1] - bins[1]*sigma
edges = (
    np.arange(minima[0], (bins[0]+1)*sigma, sigma) + rg/2.,
    np.arange(minima[1], (bins[1]+1)*sigma, sigma) + rb/2.
    )
b, x = np.histogramdd(gb_features, bins=edges)    
p = modes.smooth_density(b, 1, mode='constant')
p /= b.sum()
zmask = (p==0)
min_p = p[~zmask].min()
p[zmask] = 0.9*min_p

f = pp.figure()
pp.imshow(np.log(p), origin='lower', interpolation='nearest')
pp.xlabel('blue'); pp.ylabel('green')
pp.colorbar()
pp.contour(np.log(p), origin='lower')
f.axes[0].set_title('Log-Density')
f.savefig('2D_ex_logdense.pdf')

(labels,
 clusters,
 peaks,
 saddles) = cell_labels.assign_modes_by_density(np.log(p))
f = pp.figure()
pp.contour(np.log(p), 15, origin='lower')
pp.imshow(
    np.ma.masked_where( labels<0, labels ),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Initial Modes')
f.savefig('2D_ex_initial_labels.pdf')

(nlabels,
 nclusters,
 npeak_by_mode,
 nsaddles) = modes.merge_persistent_modes(
        labels, saddles, clusters, peaks, 0.5
     )
f = pp.figure()
pp.contour(np.log(p), 15, origin='lower')
pp.imshow(
    np.ma.masked_where( nlabels<0, nlabels ),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Persistent Modes')
f.savefig('2D_ex_persistent_labels.pdf')

# pick 1000 points with replacement
n_features = gb_features.shape[0]
r_idx = np.random.randint(0, high=n_features, size=min(n_features, 5000))
rand_features = gb_features[r_idx]
## rand_features = gb_features
ms_tools.resolve_label_boundaries(
    nlabels, p, edges, gb_features, sigma
    )
f = pp.figure()
pp.contour(np.log(p), 15, origin='lower')
pp.imshow(
    np.ma.masked_where((nlabels<0), nlabels),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Partially Resolved Boundaries')
f.savefig('2D_ex_resolved_labels.pdf')
pp.show()
