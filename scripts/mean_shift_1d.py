import numpy as np
import matplotlib.pyplot as pp
import PIL.Image as PImage

import video_proj.colors as colors
import video_proj.mean_shift.modes as modes
import video_proj.mean_shift.cell_labels as cell_labels

# swan
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
iname = 'swan'

img = colors.rgb2lab(np.array(img))

# lightness features, in [0,100)
l_features = img[...,0].ravel()
sigma = 4.0

b, x = np.histogram(l_features, bins=np.ceil(100/sigma))
p = modes.smooth_density(b.astype('d'), .25, mode='constant')
xc = (x[:-1] + x[1:])/2.0

(labels,
 clusters,
 peaks,
 saddles) = cell_labels.assign_modes_by_density(p)

peak_pts = dict()
for label in clusters:
    argmx = np.argmax(np.where(labels==label, p, 0))
    peak_pts[label] = argmx

f = pp.figure()
pp.plot(xc, p)
pp.plot([xc[peak_pts[c]] for c in clusters.keys()],
        [peaks[c] for c in clusters.keys()], 'go', label='Modes')
pp.plot([xc[s.idx] for s in saddles],
        [s.elevation for s in saddles], 'ro', label='Saddles')
pp.title('Estimated Density of Image Lightness')
pp.legend()
f.savefig('1D_modes.pdf')
pp.show()
