import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.density_est as d_est
import video_proj.mean_shift.ms_seq as ms_seq
import video_proj.colors as colors
import video_proj.mean_shift.histogram as histogram

import PIL.Image as PImage
# swan
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## # starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## # monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')


img = np.array(img)
pp.figure()
pp.imshow(img)
img = colors.rgb2lab(img)
sigma = 10.0
spatial = True

## b, cell_locs, edges = d_est.histogram(img, sigma, spatial_coords=spatial)
## p = d_est.smooth_density(b, 1)
## p = b
## p /= b.sum()
## zmask = (p==0)
## min_p = p[~zmask].min()
## p[zmask] = 0.9*min_p
## ## persistence_factor = p.ptp() * .05
## labels, mx_label = d_est.assign_modes_by_density(
##     np.log(p), cluster_tol=2
##     )
## s_img1 = d_est.segment_image_from_labels(
##     img, labels, edges, spatial_features=spatial
##     )
## print 'partial segmentation : %d / %d pixels labeled'%((s_img1>0).sum(),
##                                                        s_img1.size)
## cm = colors.npt_colormap(mx_label)
## pp.figure()
## pp.imshow(np.ma.masked_where(s_img1<0, s_img1), cmap=cm)
## pp.colorbar()
## # now to refine the labels
## features = d_est.image_to_features(img)
## if not spatial:
##     features = features[:,2:]
## # pick 1000 points with replacement
## n_features = features.shape[0]
## r_idx = np.random.randint(0, high=n_features, size=5000)
## rand_features = features[r_idx]
## ms_seq.walk_uphill(labels, p, edges, rand_features, sigma)
## s_img2 = d_est.segment_image_from_labels(
##     img, labels, edges, spatial_features=spatial
##     )
## pp.figure()
## pp.imshow(np.ma.masked_where(s_img2<0, s_img2), cmap=cm)
## pp.colorbar()
## pp.show()
