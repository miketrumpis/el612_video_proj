from __future__ import division
import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.classification as cls
import video_proj.colors as colors
import video_proj.util as ut
import video_proj.mean_shift.histogram as histogram
import video_proj.mean_shift.topological as topo

import PIL.Image as PImage

def plot_masked_segmap(s_img):
    f = pp.figure()
    mx_label = s_img.max()
    cm = colors.npt_colormap(mx_label)
    pp.imshow(np.ma.masked_where(s_img<0, s_img), cmap=cm)
    pp.colorbar()
    f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
    return f

def plot_masked_centroid_image(image, segmap):
    c_image = colors.quantize_from_centroid(image, segmap)
    f = pp.figure()
    mask = np.ones(c_image.shape, 'bool')
    mask.shape = (-1, 3)
    mask[segmap.ravel() <= 0] = 0
    pp.imshow(np.ma.MaskedArray(c_image, mask=mask))
    return f

## # swan
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
## iname = 'swan'
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## iname = 'llama'
# starfish
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
iname = 'starfish'
## # monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')
## iname = 'monkey'


rgb_img = np.array(img)
f = pp.figure()
pp.imshow(rgb_img)
f.savefig(iname+'_raw.pdf')
f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
img = colors.rgb2lab(rgb_img)
## img = img[...,0].squeeze()
c_sigma = 5
spatial = True
s_sigma = 20.0

m_seek = topo.ModeSeeking(
    spatial_bw = s_sigma, color_bw = c_sigma
    )
classifier = m_seek.train_on_image(img)

s_img0 = classifier.classify(img, refined=False)
print 'partial segmentation : %d / %d pixels labeled'%((s_img0>=0).sum(),
                                                       s_img0.size)
f = plot_masked_centroid_image(rgb_img, s_img0)
f.savefig(iname+'_initial.pdf')
pp.show()

classifier = m_seek.persistence_merge(2)
s_img1 = classifier.classify(img, refined=False)
print 'partial segmentation : %d / %d pixels labeled'%((s_img1>=0).sum(),
                                                       s_img1.size)
f = plot_masked_centroid_image(rgb_img, s_img1)
f.savefig(iname+'_persistence_no_refine.pdf')
pp.show()
