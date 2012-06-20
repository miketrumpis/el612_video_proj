"""
Module for various video importing tools
"""
from __future__ import division

from glob import glob
import numpy as np
import os

from util import Bucket
import colors

def _safe_file(f, mode='rb'):
    if isinstance(f, str):
        return open(f, mode)
    else:
        return f

def yuv_image(b_arr, Nx, Ny, yuv_mode='420', nbytes=-1, rgb=True):
    """
    Convert an array of bytes containing YUV information into an
    RGB frame with dimensions in shape.
    """
    b_arr = np.fromfile(
        _safe_file(b_arr, mode='rb'), count=nbytes, dtype=np.uint8
        )
    if yuv_mode == '444':
        c_sub = 1
        raise NotImplementedError
    elif yuv_mode == '420':
        # color bytes comes at the end of the frame
        cbytes = Nx*Ny/4
        y = b_arr[:Nx*Ny]
        u = b_arr[Nx*Ny:Nx*Ny+cbytes]
        v = b_arr[Nx*Ny+cbytes:]
        y.shape = (Ny, Nx); u.shape = (Ny/2, Nx/2); v.shape = (Ny/2, Nx/2)
        u = np.repeat(u, 2, axis=0); u = np.repeat(u, 2, axis=1)
        v = np.repeat(v, 2, axis=0); v = np.repeat(v, 2, axis=1)
    elif yuv_mode == '422':
        cbytes = Nx*Ny/2
        y = b_arr[:Nx*Ny]
        u = b_arr[Nx*Ny:Nx*Ny+cbytes]
        v = b_arr[Nx*Ny+cbytes:]
        y.shape = (Ny, Nx); u.shape = (Ny, Nx/2); v.shape = (Ny, Nx/2)
        u = np.repeat(u, 2, axis=1)
        v = np.repeat(v, 2, axis=1)
    else:
        raise NotImplementedError

    image = np.c_[y.ravel(), u.ravel(), v.ravel()]
    if rgb:
        image = colors.yuv2rgb(image).reshape(Ny,Nx,3)
    else:
        image = np.reshape(image, (Ny, Nx, 3))
    return image
    
def yuv_sequence(v_root, Nx, Ny, yuv_mode='420', t_range=(), rgb=True):
    v_files = glob(v_root)
    if t_range:
        v_files = v_files[t_range[0]:t_range[1]]
    vid = np.empty((len(v_files), Ny, Nx, 3), np.uint8)
    for t,f in enumerate(v_files):
        vid[t] = yuv_image(f, Nx, Ny, yuv_mode=yuv_mode, rgb=rgb)
    return vid

def y4m_sequence(v_file, t_range=(), rgb=True):
    v_file = _safe_file(v_file)
    hdr = parse_y4m_header(v_file)
    n_frames = hdr.n_frames
    Nx = hdr.width
    Ny = hdr.height
    yuv_mode = hdr.color_mode
    f_bytes = hdr.frame_bytes
    if not t_range:
        t_range = xrange(n_frames)
        vid = np.empty( (n_frames, Ny, Nx, 3), np.uint8 )
    else:
        t0, t1 = t_range
        t_range = xrange(t1-t0)
        vid = np.empty( (t1-t0, Ny, Nx, 3), np.uint8 )
        # go and throw away t0 frames
        # ...
        for i in xrange(t0):
            s = v_file.readline()
        v_file.seek(-6, 1)

    for i in t_range:
        chk = v_file.readline()
        if chk != 'FRAME\n':
            raise RuntimeError('File position seems to be incorrect!')
        vid[i] = yuv_image(
            v_file, Nx, Ny, yuv_mode=yuv_mode,
            nbytes=f_bytes, rgb=rgb
            )
    v_file.close()
    return vid, hdr.f_rate

_y4m_char_conversion = dict(
    W = 'width', H = 'height', F = 'f_rate', I = 'interlacing',
    A = 'par', C = 'color_mode'
    )
def parse_y4m_header(y4m_file):
    y4m_file.seek(0)
    h_string = y4m_file.readline()
    h_fields = h_string.split(' ')
    h_conversion = []
    f_size = 1
    for f in h_fields:
        ## f = f.upper()
        if f == 'YUV4MPEG2':
            continue
        ctrl, code = f[0], f[1:].strip()
        expanded_ctrl = _y4m_char_conversion[ctrl]
        if ctrl in ('W', 'H'):
            code = int(code)
            f_size *= code
        elif ctrl == 'F':
            c1, c2 = code.split(':')
            # unrefined .. ignores 30000:1001 and 24000:1001 possibilities
            code = int(c1[:2])
        # otherwise just store it as-is??
        h_conversion.append( (expanded_ctrl, code) )
    h_conversion = dict(h_conversion)
    if 'color_mode' in h_conversion:
        # if the color info was present, compute bytes that way
        cmode = h_conversion['color_mode']
        if cmode == '444':
            f_bytes = f_size * 3
        elif cmode == '422':
            f_bytes = f_size * 2
        elif cmode == '420':
            f_bytes = (f_size * 3)//2
        h_conversion['frame_bytes'] = f_bytes
    else:
        # otherwise, count bytes in a frame
        fplace = y4m_file.tell()
        # read past 1st "FRAME\n" code
        s = y4m_file.readline()
        # then read 1st frame (including 2nd "FRAME\n" code)
        s = y4m_file.readline()
        f_bytes = len(s) - 6
        h_conversion['frame_bytes'] = f_bytes
        bpp = f_bytes / float(f_size)
        if bpp == 3:
            cmode = '444'
        elif bpp == 2:
            cmode = '422'
        elif bpp == 1.5:
            cmode = '420'
        h_conversion['color_mode'] = cmode
        y4m_file.seek(fplace, 0)
        
    # last, find the number of frames
    sr = os.stat_result(os.stat(y4m_file.name))
    nbytes = (sr.st_size - y4m_file.tell())
    # add 6 bytes to each frame for the "FRAME\n" code -- round
    # down in case there's any crap at the end
    h_conversion['n_frames'] = nbytes // (f_bytes + 6)
    return Bucket( **dict(h_conversion) )
        
        

    
