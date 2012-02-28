import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.density_est as d_est
import video_proj.mean_shift.colors as colors

import PIL.Image as PImage
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
img = np.array(img)
b, cell_locs, edges = d_est.histogram(img, 30)
p = d_est.smooth_density(b, 1)
labels, mx_label = d_est.assign_modes_by_density(p)
s_img = d_est.segment_image_from_labels(img, labels, edges)
cm = colors.npt_colormap(mx_label)
pp.imshow(np.ma.masked_where(s_img<0, s_img), cmap=cm)

# now to refine the labels
