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
# swan
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
iname = 'swan'
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## # starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## iname = 'starfish'
# monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')
## iname = 'monkey'


img = np.array(img)
f = pp.figure()
pp.imshow(img)
f.savefig(iname+'_raw.pdf')
f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
img = colors.rgb2lab(img)
## img = img[...,0].squeeze()
c_sigma = 20.0
spatial = True
s_sigma = 20.0

b, accum, edges = histogram.histogram(
    img, s_sigma, spatial_features=spatial, color_spacing=c_sigma
    )
p, mu = modes.smooth_density(b, 1, accum=accum, mode='constant')
del b
del accum
## p = b
## p /= b.sum()
zmask = (p==0)
min_p = p[~zmask].min()
p[zmask] = 0.9*min_p

## persistence_factor = p.ptp() * .05
(labels,
 clusters,
 peaks,
 saddles) = cell_labels.assign_modes_by_density(np.log(p))

## s_img0 = modes.segment_image_from_labeled_modes(
##     img, labels, edges, spatial_features=spatial
##     )
## print 'partial segmentation : %d / %d pixels labeled'%((s_img0>0).sum(),
##                                                        s_img0.size)
## mx_label = len(clusters.keys())
## cm = colors.npt_colormap(mx_label)
## f = pp.figure()
## pp.imshow(np.ma.masked_where(s_img0<0, s_img0), cmap=cm)
## pp.colorbar()
## f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
## f.savefig(iname+'_initial.pdf')

## # perform persistence based merging
## (nlabels,
##  nclusters,
##  npeak_by_mode,
##  nsaddles) = modes.merge_persistent_modes(
##         labels, saddles, clusters, peaks, 0.05
##      )
## clabels = ut.contiguous_labels(nlabels, nclusters)
 
## s_img1 = modes.segment_image_from_labeled_modes(
##     img, clabels, edges, spatial_features=spatial
##     )
## print 'partial segmentation : %d / %d pixels labeled'%((s_img1>0).sum(),
##                                                        s_img1.size)
## mx_label = len(nclusters.keys())
## cm = colors.npt_colormap(mx_label)
## f = pp.figure()
## pp.imshow(np.ma.masked_where(s_img1<0, s_img1), cmap=cm)
## pp.colorbar()
## f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
## f.savefig(iname+'_persistence.pdf')

## # now to refine the labels
## features = ut.image_to_features(img)
## # XXX: these gymnastics over spatial = {True,False} are cumbersome
## if spatial:
##     sigma = s_sigma
## else:
##     sigma = c_sigma
##     c_sigma = None
##     features = features[:,2:]
## if 0:
##     ms_tools.resolve_label_boundaries(
##         clabels, p, edges, features, sigma, c_sigma=c_sigma, max_iter=10
##         )
##     s_img2 = modes.segment_image_from_labeled_modes(
##         img, clabels, edges, spatial_features=spatial
##         )
##     print 'partial segmentation : %d / %d pixels labeled'%((s_img2>0).sum(),
##                                                            s_img2.size)

## else:
##     s_img2 = s_img1.copy()
##     ms_tools.resolve_segmentation_boundaries(
##         s_img2, features, clabels, edges, s_sigma, max_iter=5, c_sigma=c_sigma
##         )

## f = pp.figure()
## pp.imshow(np.ma.masked_where(s_img2<0, s_img2), cmap=cm)
## pp.colorbar()
## f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
## f.savefig(iname+'_refined.pdf')
pp.show()
