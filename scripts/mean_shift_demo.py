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
## iname = 'swan'
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## # starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## iname = 'starfish'
# monkey
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')
iname = 'monkey'


img = np.array(img)
f = pp.figure()
pp.imshow(img)
f.savefig(iname+'_raw.pdf')
f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
img = colors.rgb2lab(img)
## img = img[...,0].squeeze()
c_sigma = 15
spatial = True
s_sigma = 25.0

features = ut.image_to_features(img)
if not spatial:
    features = features[:,2:]

b, accum, edges = histogram.histogram(
    img, s_sigma, spatial_features=spatial, color_spacing=c_sigma
    )
p, mu = modes.smooth_density(b, 1, accum=accum, mode='constant')
del b
del accum
sr_grid = np.concatenate( (mu, p[...,None]), axis=-1).copy()

zmask = (p==0)
min_p = p[~zmask].min()
p[zmask] = 0.9*min_p

persistence_factor = np.log(p).ptp() * .01

def plot_masked_segmap(s_img):
    f = pp.figure()
    mx_label = s_img.max()
    cm = colors.npt_colormap(mx_label)
    pp.imshow(np.ma.masked_where(s_img<0, s_img), cmap=cm)
    pp.colorbar()
    f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
    return f

# Find non-iterative labeling via topographical analysis of log(P)
0/0
(labels,
 clusters,
 peaks,
 saddles) = cell_labels.assign_modes_by_density(np.log(p))
cluster_cutoff = int(0.005 * clusters.size + 0.5)
false_labels = modes.threshold_clusters(clusters, cluster_cutoff)

s_img0 = modes.segment_image_from_labeled_modes(
    img, labels, edges, spatial_features=spatial
    )
print 'partial segmentation : %d / %d pixels labeled'%((s_img0>0).sum(),
                                                       s_img0.size)
f = plot_masked_segmap(s_img0)
f.savefig(iname+'_initial.pdf')

# Resolve boundaries in the image with mean shift
s_img1 = s_img0.copy()
ms_tools.resolve_segmentation_boundaries(
    s_img1, features, sr_grid, labels, edges
    )
f = plot_masked_segmap(s_img1)
f.savefig(iname+'_pixel_resolved_from_initial.pdf')

# perform persistence based merging
(nlabels,
 nclusters,
 npeak_by_mode,
 nsaddles) = modes.merge_persistent_modes(
        labels, saddles, clusters, peaks, 0.05
     )
clabels = ut.contiguous_labels(nlabels, nclusters)
 
s_img2 = modes.segment_image_from_labeled_modes(
    img, clabels, edges, spatial_features=spatial
    )
print 'partial segmentation (merged) : %d / %d pixels labeled'%(
    (s_img2>0).sum(), s_img2.size
    )
f = plot_masked_segmap(s_img2)
f.savefig(iname+'_persistence.pdf')

# refine these boundaries in the image with mean shift
s_img3 = s_img2.copy()
ms_tools.resolve_segmentation_boundaries(
    s_img3, features, sr_grid, clabels, edges
    )
f = plot_masked_segmap(s_img3)
f.savefig(iname+'_pixel_resolved_from_merged.pdf')


# now to refine the labels and compare that to the pixel-refinement

ms_tools.resolve_label_boundaries(
    clabels, p, mu, edges
    )
s_img4 = modes.segment_image_from_labeled_modes(
    img, clabels, edges, spatial_features=spatial
    )
print 'partial segmentation (refined): %d / %d pixels labeled'%(
    (s_img4>0).sum(), s_img4.size
    )
f = plot_masked_segmap(s_img4)
f.savefig(iname+'_refined.pdf')
pp.show()
