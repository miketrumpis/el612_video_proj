import numpy as np
import matplotlib.pyplot as pp
import mstools.colors as colors
import mstools.util as ut
from mstools.mean_shift.topological import ModeSeeking

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

gb_image = img[...,0:2].astype('d')

sigma = 3
ms = ModeSeeking(color_bw=sigma)
c = ms.train_on_image(gb_image, bin_sigma=1.0)

p = c.mud_grid[...,-1]
zmask = (p==0)
min_p = p[~zmask].min()
p[zmask] = 0.9*min_p
np.log(p, out=p)

f = pp.figure()
pp.imshow(p, origin='lower', interpolation='nearest', cmap=pp.cm.hot)
pp.xlabel('blue'); pp.ylabel('green')
pp.colorbar()
pp.contour(p, levels=np.linspace(-10, p.max(), 15), origin='lower')
f.axes[0].set_title('Log-Density')
f.savefig('2D_ex_logdense.pdf')

f = pp.figure()
pp.contour(p, levels=np.linspace(-10, p.max(), 15), origin='lower')
pp.imshow(
    np.ma.masked_where( c.labels<=0, c.labels ),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Initial Modes')
f.savefig('2D_ex_initial_labels.pdf')

f = pp.figure()
c = ms.persistence_merge(0.5)
pp.contour(p, levels=np.linspace(-10, p.max(), 15), origin='lower')
pp.imshow(
    np.ma.masked_where( c.labels<=0, c.labels ),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Persistent Modes')
f.savefig('2D_ex_persistent_labels.pdf')

f = pp.figure()
c.refine_labels()
pp.contour(p, levels=np.linspace(-10, p.max(), 15), origin='lower')
pp.imshow(
    np.ma.masked_where((c.labels<=0), c.labels),
    origin='lower', alpha=.4, interpolation='nearest'
    )
pp.xlabel('blue'); pp.ylabel('green')
f.axes[0].set_title('Partially Resolved Boundaries')
f.savefig('2D_ex_resolved_labels.pdf')

pp.show()
