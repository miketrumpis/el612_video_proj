from __future__ import division
import numpy as np
import matplotlib.pyplot as pp
import video_proj.mean_shift.classification as cls
import video_proj.colors as colors
import video_proj.util as ut
import video_proj.importing as importing
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
    f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
    return f

## # swan
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
## iname = 'swan'
## c_sigma = 10; s_sigma = 30; p_thresh = 0.05
## # llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## iname = 'llama'
## c_sigma = 5.0; s_sigma = 40.0; p_thresh = 0.3
## # starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## iname = 'starfish'
## c_sigma = 10.0; s_sigma = 50.0; p_thresh = 0.0
# monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')
## iname = 'monkey'
## c_sigma = 10.0; s_sigma = 5.0; p_thresh = 0.17

## # Crew sequence
## vid, _ = importing.y4m_sequence(
##     '/Users/mike/docs/classes/el612/vids/crew_cif.y4m', t_range=(40,100)
##     )
## img = vid[0]
## iname = 'crew'
## c_sigma = 10; s_sigma = None; p_thresh = 0

# Football sequence
vid, _ = importing.y4m_sequence(
    '/Users/mike/docs/classes/el612/vids/football_422_cif.y4m',
    t_range=(40,100)
    )
img = vid[0]
iname = 'football'
c_sigma = 4; s_sigma = None; p_thresh = 0.03

rgb_img = np.array(img)
f = pp.figure()
pp.imshow(rgb_img)
f.savefig(iname+'_raw.pdf')
f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
img = colors.rgb2lab(rgb_img)
## img = img[...,0].squeeze()

m_seek = topo.ModeSeeking(
    spatial_bw = s_sigma, color_bw = c_sigma
    )
classifier = m_seek.train_on_image(img, bin_sigma = 1.25)

s_img0 = classifier.classify(img, refined=False, cluster_size_threshold=50)
print 'partial segmentation : %d / %d pixels labeled'%((s_img0>=0).sum(),
                                                       s_img0.size)
f = plot_masked_centroid_image(rgb_img, s_img0)
f.savefig(iname+'_initial.pdf')
f = plot_masked_segmap(s_img0)
f.savefig(iname+'_initial_labels.pdf')
pp.show()

classifier = m_seek.persistence_merge(p_thresh)
classifier.refine_labels()
s_img1 = classifier.classify(img, refined=False, cluster_size_threshold=50)
print 'partial segmentation : %d / %d pixels labeled'%((s_img1>=0).sum(),
                                                       s_img1.size)
f = plot_masked_centroid_image(rgb_img, s_img1)
f.savefig(iname+'_persistence_refined_model.pdf')
f = plot_masked_segmap(s_img1)
f.savefig(iname+'_persistence_refined_model_labels.pdf')


lab_vid = np.array(
    [colors.rgb2lab(vf) for vf in vid]
    )
svid = cls.classify_sequence(classifier, lab_vid)
anim = ut.animate_frames(
    np.ma.masked_where(svid < 0, svid),
    movie_name=iname+'_naive_classifier',
    fps=10,
    cmap=colors.npt_colormap(svid.max())
    )
anim.repeat = False
anim = ut.animate_frames(
    vid,
    movie_name=iname,
    fps=10,
    )
anim.repeat = False

pp.show()


