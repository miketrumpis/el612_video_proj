from __future__ import division
import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.modes as modes
import video_proj.mean_shift.mean_shift_tools as ms_tools
import video_proj.colors as colors
import video_proj.util as ut
import video_proj.mean_shift.histogram as histogram
import video_proj.mean_shift.cell_labels as cell_labels

import PIL.Image as PImage
## # swan
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
# llama (HARD!)
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## # starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## # monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')


img = np.array(img)
pp.figure()
pp.imshow(img)
img = colors.rgb2lab(img)
sigma = 20.0
spatial = True

## b, cell_locs, edges = histogram.histogram(img, sigma, spatial_features=spatial)
## p = modes.smooth_density(b, 0.5)
## ## p = b
## ## p /= b.sum()
## zmask = (p==0)
## min_p = p[~zmask].min()
## p[zmask] = 0.9*min_p

## ## persistence_factor = p.ptp() * .05
## (labels,
##  clusters,
##  peaks,
##  saddles) = cell_labels.assign_modes_by_density(np.log(p))
 
## s_img1 = modes.segment_image_from_labeled_modes(
##     img, labels, edges, spatial_features=spatial
##     )
## print 'partial segmentation : %d / %d pixels labeled'%((s_img1>0).sum(),
##                                                        s_img1.size)
## mx_label = max(clusters.keys())
## cm = colors.npt_colormap(mx_label)
## pp.figure()
## pp.imshow(np.ma.masked_where(s_img1<0, s_img1), cmap=cm)
## pp.colorbar()
## # now to refine the labels
## features = ut.image_to_features(img)
## if not spatial:
##     features = features[:,2:]
## s_img2 = s_img1.copy()
## ms_tools.resolve_segmentation_boundaries(
##     s_img2, features, labels, edges, sigma
##     )
## pp.figure()
## pp.imshow(np.ma.masked_where(s_img2<0, s_img2), cmap=cm)
## pp.colorbar()
## pp.show()

