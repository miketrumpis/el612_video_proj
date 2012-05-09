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

video = False
# swan
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/8068.jpg')
## iname = 'swan'
## c_sigma = 10; s_sigma = 40; p_thresh = 0.05; n_parts = 3

# llama (HARD!)
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/test/6046.jpg')
## iname = 'llama'
## c_sigma = 5.0; s_sigma = 40.0; p_thresh = 0.3; n_parts = 4

# dharma
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/56028.jpg')
## iname = 'dharma'
## c_sigma = 3.0; s_sigma = 0; p_thresh = 0; n_parts = 10

# bear
img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/100080.jpg')
iname = 'bear'
c_sigma = 5.0; s_sigma = 70; p_thresh = 0.01; n_parts = 5

# starfish
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/12003.jpg')
## iname = 'starfish'
## c_sigma = 10.0; s_sigma = 50.0; p_thresh = 0; n_parts = 2

# monkey
## img = PImage.open('/Users/mike/docs/classes/el612/proj/berk_data/BSR/BSDS500/data/images/train/16052.jpg')
## iname = 'monkey'
## c_sigma = 10.0; s_sigma = 5.0; p_thresh = 0.17; 

## # Crew sequence
## vid, _ = importing.y4m_sequence(
##     '/Users/mike/docs/classes/el612/vids/crew_cif.y4m', t_range=(40,140)
##     )
## img = vid[0]
## iname = 'crew'
## c_sigma = 10; s_sigma = None; p_thresh = 0; n_parts = 2; video = True

# Football sequence
## vid, _ = importing.y4m_sequence(
##     '/Users/mike/docs/classes/el612/vids/football_422_cif.y4m',
##     t_range=(1,100)
##     )
## img = vid[0]
## iname = 'football'
## c_sigma = 3; s_sigma = None; p_thresh = 0.03; n_parts = 14; video = True

## vid, _ = importing.y4m_sequence(
##     '/Users/mike/docs/classes/el612/vids/stefan_sif.y4m',
##     t_range=()
##     )
## img = vid[0]
## iname = 'stefan'
## c_sigma = 5; s_sigma = None; p_thresh = 0; n_parts = 20; #video = True


rgb_img = np.array(img)
f = pp.figure()
pp.imshow(rgb_img)
f.axes[0].xaxis.set_visible(False); f.axes[0].yaxis.set_visible(False)
f.savefig(iname+'_raw.pdf')
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

classifier = m_seek.merge_until_n(n_parts)
classifier.refine_labels()
s_img1 = classifier.classify(img, refined=False, cluster_size_threshold=50)
print 'partial segmentation : %d / %d pixels labeled'%((s_img1>=0).sum(),
                                                       s_img1.size)
f = plot_masked_centroid_image(rgb_img, s_img1)
f.savefig(iname+'_persistence_%d_parts_model.pdf'%n_parts)
f = plot_masked_segmap(s_img1)
f.savefig(iname+'_persistence_%d_parts_model_labels.pdf'%n_parts)

bmap = ut.draw_boundaries(s_img1, m_seek.saddles)
f = pp.figure()
pp.imshow(bmap, cmap=pp.cm.gray)
f.savefig(iname+'_boundaries.pdf')

if video:
    lab_vid = np.array(
        [colors.rgb2lab(vf) for vf in vid]
        )
    svid = cls.classify_sequence(classifier, lab_vid)
    segcmap = colors.npt_colormap(svid.max())
    n = pp.normalize()
    svid_masked = np.where(svid < 0, 0, svid)
    svid_mapped = segcmap(n(svid_masked).ravel()).reshape(svid.shape+(4,))
    svid_mapped = (svid_mapped[...,:3]*255).astype('B')
    both_vids = np.concatenate((vid, svid_mapped), axis=2)

    anim = ut.animate_frames(
        both_vids,
        movie_name=iname+'_joint',
        fps=25,
        )
    anim.repeat = False

    ## anim = ut.animate_frames(
    ##     np.ma.masked_where(svid < 0, svid),
    ##     movie_name=iname+'_naive_classifier',
    ##     fps=25,
    ##     cmap=colors.npt_colormap(svid.max())
    ##     )
    ## anim.repeat = False
    ## anim = ut.animate_frames(
    ##     vid,
    ##     movie_name=iname,
    ##     fps=25,
    ##     )
    ## anim.repeat = False

    ## x_mn, x_mx = (240, 320)
    ## y_mn, y_mx = (46, 100)
    ## hx = int(x_mx - x_mn)
    ## hy = int(y_mx - y_mn)
    ## x0 = (x_mx + x_mn)//2
    ## y0 = (y_mx + y_mn)//2
    ## x = np.array([x0, y0], 'i')
    ## import video_proj.tracking.object_model as om
    ## seed = om.RectangleTrackingObject(x, hx, hy)

pp.show()


