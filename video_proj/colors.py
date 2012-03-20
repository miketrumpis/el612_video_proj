import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import datad

def npt_colormap(N, name='jet'):
    try:
        color_spec = datad.get(name)
    except KeyError:
        raise ValueError('Bad colormap name: %s'%name)
    return LinearSegmentedColormap('new_%s'%name, color_spec, N)
            
_raised_rgb_to_xyz = np.array([ [0.488718, 0.310680, 0.200602],
                                [0.176204, 0.812985, 0.0108109],
                                [0.0, 0.0102048, 0.989795] ])
_xyz_to_lab = np.array([ [0.0, 116.0, 0.0],
                         [500.0, -500.0, 0.0],
                         [0.0, 200.0, -200.0] ])
def rgb2lab(image):
    # flatten first two dimensions
    ishape = image.shape
    xyz_image = image / 255.
    xyz_image.shape = (ishape[0]*ishape[1],) + ishape[2:]
    np.power(xyz_image, 2.2, xyz_image)
    xyz_image = np.dot(xyz_image, _raised_rgb_to_xyz.T)
    xyz_image.shape = -1
    tiny = xyz_image < 216./24389
    if not tiny.any():
        np.power(xyz_image, 1/3., xyz_image)
    else:
        tiny_colors = xyz_image[tiny]
        xyz_image[tiny] = ( (24389.0/27.0)*tiny_colors + 16 )/116.0
        not_tiny_colors = xyz_image[~tiny]
        xyz_image[~tiny] = np.power(not_tiny_colors, 1/3.)
    xyz_image.shape = (ishape[0]*ishape[1],) + ishape[2:]
    lab_image = np.dot(xyz_image, _xyz_to_lab.T)
    lab_image[:,0] -= 16
    lab_image.shape = ishape
    return lab_image

yuv_offset = np.array([16, 128, 128], 'B')
yuv_rgb_xform = np.array([ [1.164,  0.000,  1.596],
                           [1.164, -0.392, -0.813],
                           [1.164,  2.017,  0.000] ])
def yuv2rgb(image):
    shape = image.shape
    image.shape = (-1, 3)
    rgb_image = image.astype('h') - yuv_offset
    ## rgb_image = np.round(np.dot(yuv_rgb_xform, rgb_image))
    rgb_image = np.round(np.dot(rgb_image, yuv_rgb_xform.T))
    rgb_image_bytes = np.empty(image.shape, 'B')
    np.clip(rgb_image, 0, 255, out=rgb_image_bytes)
    return rgb_image_bytes
    
    
