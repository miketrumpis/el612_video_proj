import numpy as np
import matplotlib.pyplot as pp
import PIL.Image as PImage

import video_proj.colors as colors
from video_proj.mean_shift.topological import ModeSeeking

# swan
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
iname = 'swan'

image = colors.rgb2lab(np.array(img))
gray_image = image[...,0]

sigma = 4.0

ms = ModeSeeking(color_bw=sigma)
c = ms.train_on_image(gray_image, density_xform=None)
x = c.cell_edges[0]
xc = (x[:-1] + x[1:])/2.0
p = c.mud_grid[...,-1]

peak_pts = [m.idx for (_, m) in ms.clusters.items()]
peak_hts = [m.elevation for (_, m) in ms.clusters.items()]
saddle_pts = [s.idx for s in ms.saddles]
saddle_hts = [s.elevation for s in ms.saddles]

f = pp.figure()
pp.plot(xc, p)
pp.plot(xc[peak_pts], peak_hts, 'go', label='Modes')
pp.plot(xc[saddle_pts], saddle_hts, 'ro', label='Saddles')
pp.title('Estimated Density of Image Lightness')
pp.legend()
f.savefig('1D_modes.pdf')
pp.show()
