from __future__ import division
import numpy as np

class GeometricTrackingObject(object):

    def __init__(self, x, hx, hy):
        # hx and hy are HALF bandwidths -- I.E. the window size
        # is determined by (2*hx + 1, 2*hy + 1)
        self.x = x
        self.h = np.array([hx, hy])

    def extract_pts(self, image, x=None, h=None):
        pass

    def window(self, h=None, kernel='epan'):
        """
        Creates a windowing function based on the kernel type
        and the bandwidth (or a provided bandwidth)
        """

class RectangleTrackingObject(GeometricTrackingObject):
    
    def boundaries(self, x=None, h=None):
        x = self.x if x is None else x
        h = self.h if h is None else h
        mn = np.round( x - h ).astype('i')
        mx = mn + 2*h + 1
        return np.array([mn[0], mx[0], mn[1], mx[1]])

    def extract_pts(self, image, x=None, h=None):
        x_mn, x_mx, y_mn, y_mx = self.boundaries(x=x, h=h)
        slicer = (slice(y_mn, y_mx), slice(x_mn, x_mx))
        return safe_block_slice(image, slicer)
        #return image[y_mn:y_mx, x_mn:x_mx]

    def window(self, h=None, kernel='epan'):
        # a rectangle window is truncated outside of
        # x \in [-hx hx]
        # y \in [-hy hy]
        # let's make a window that's not exactly radially
        # (or elliptically) symmetric, but rather l-infty symmetric
        # (i.e. a pyramid)

        h = self.h if h is None else h
        # make this interval open ended, to avoid actual zeros

        hx, hy = h
        dx = 2.0/(2*hx+2); dy = 2.0/(2*hy+2)
        lm_x = hx*dx; lm_y = hy*dy
        y_pts, x_pts = np.mgrid[-lm_y:lm_y:1j*(2*hy+1),
                                -lm_x:lm_x:1j*(2*hx+1)]
        d = np.maximum(np.abs(y_pts), np.abs(x_pts))
        if kernel == 'epan':
            w = 1-d**2
        elif kernel == 'gauss':
            # XXX: check this -- might be just d/2
            w = np.exp(-d**2/2)
        else:
            raise NotImplementedError(
                'Only Epanechnikov and Gaussian kernels supported'
                )
        return w / w.sum()


# XXX: this may be helpful
def safe_block_slice(frame, slicer):
    # safely extend zeros if the block slicer extends beyond the bounds
    # of the frame -- slice objects must specify start and stop!!
    Ny, Nx = frame.shape[:2]
    cdim = 1 if len(frame.shape) == 2 else frame.shape[-1]
    yslice, xslice = slicer[:2]
    y_start = yslice.start; y_stop = yslice.stop
    x_start = xslice.start; x_stop = xslice.stop
    y_step = yslice.step or 1
    x_step = xslice.step or 1
    dy = (y_stop - y_start)
    dx = (x_stop - x_start)
    y_hi = y_stop > Ny; y_lo = y_start < 0
    x_hi = x_stop > Nx; x_lo = x_start < 0
    # check easy case
    if not (y_hi or y_lo or x_hi or x_lo):
        return frame[slicer]
    # slightly harder case -- if entire x or y range is out of bounds,
    # then block is zero
    y_lo_hi = y_start > Ny; y_hi_lo = y_stop < 0
    x_lo_hi = x_start > Nx; x_hi_lo = x_stop < 0
    block = np.squeeze(np.zeros((dy,dx,cdim), frame.dtype))
    if y_lo_hi or y_hi_lo or x_lo_hi or x_hi_lo:
        if cdim > 1:
            return block[::y_step, ::x_step, :]
        return block[::y_step, ::x_step]
    # otherwise one of both edges are out of bounds in at least one direction
    safe_yslice = slice(max(0, y_start), min(Ny, y_stop)) #, yslice.step)
    safe_xslice = slice(max(0, x_start), min(Nx, x_stop)) #, xslice.step)
    # where does this correspond in the extended block?
    byslice = slice(
        safe_yslice.start - y_start,
        dy - (y_stop - safe_yslice.stop) #, yslice.step
        )
    bxslice = slice(
        safe_xslice.start - x_start,
        dx - (x_stop - safe_xslice.stop) #, xslice.step
        )
    bcslice = slice(None)
    if cdim > 1:
        block[ (byslice, bxslice, bcslice) ] = \
            frame[ (safe_yslice, safe_xslice, bcslice) ]
    else:
        block[ (byslice, bxslice) ] = frame[ (safe_yslice, safe_xslice) ]
    return block[::y_step, ::x_step]
