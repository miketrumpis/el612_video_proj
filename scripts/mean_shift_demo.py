import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.density_est as d_est

import PIL.Image as PImage
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/100007.jpg')
a_img = np.array(img)
p, cell_locs, edges = d_est.histogram(a_img, 20)
p = d_est.smooth_density(p, 1)
## labels, mx_label = d_est.assign_modes_by_density(p)
# now to refine the labels
