"""
This module implements various mappings from the pixel domain
"""

import numpy as np

import video_proj.colors as colors
import video_proj.util as util

from video_proj.fn import Function

def mean_color_image(image, segmap):
    c_image = colors.quantize_from_centroid(image, segmap)
    return c_image

def density_image(image, p, edges, interpolation=1):
    pdf = Function(p, edges, interp_order=interpolation)
    # try to infer whether the spatial coordinates are part
    # of the feature space
    cdim = image.shape[-1] if len(image.shape) > 2 else 1
    if len(edges) > cdim:
        f_vec = util.image_to_features(image)
    else:
        f_vec = np.reshape(image, (-1, cdim))
    return pdf(f_vec).reshape(image.shape[:2])
