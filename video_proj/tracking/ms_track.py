import numpy as np

def track_frames(frames, seed, D):
    # frames is (n_frame, ny, nx, [n_cdim])
    # seed is ??? a GeometricTrackingObject perhaps?
    # D is a Function type?

    # this should be fixed
    nrm = D.fn_grid.max()

    Nframe, Ny, Nx = frames.shape[:3]
    yy, xx = np.mgrid[:Ny, :Nx]
    x = seed.x
    cdim = 1 if len(frames.shape) < 4 else frames.shape[3]
    locs = np.zeros((Nframe,2))
    bws = np.zeros((Nframe,2))
    locs[0] = seed.x
    bws[0] = seed.h
    w_fn = seed.window()
    w_fn = w_fn.ravel()

    # build model distribution
    frame = frames[0]
    sub_img = seed.extract_pts(frame)
    sub_img = np.reshape(sub_img, (-1, cdim))
    qx = D(sub_img)
    vx = np.minimum( qx.min()/qx, qx.sum() )
    qx = qx * w_fn * vx
    qx /= qx.sum()
    
    for n, frame in enumerate(frames[1:]):
        
        # could map entire frame's features now, or extract
        # regions at each iteration
        n_iter = 0
        x0 = seed.x.copy()
        xi = x0
        while n_iter < 20:
            n_iter += 1
            
            sub_img = seed.extract_pts(frame, x=xi, h=seed.h)
            sub_x = seed.extract_pts(xx, x=xi, h=seed.h)
            sub_y = seed.extract_pts(yy, x=xi, h=seed.h)
            sub_img = np.reshape(sub_img, (-1, cdim))
            sub_x = np.reshape(sub_x, -1)
            sub_y = np.reshape(sub_y, -1)
            ## sub_img.shape = (-1, cdim)
            ## sub_x.shape = -1
            ## sub_y.shape = -1
            px = D(sub_img) * w_fn
	    px = D(sub_img)
	    vx = np.minimum( px.min()/px, px.sum() ) # background weighting
	    px = px * w_fn * vx
	    px /= px.sum()
	    
	    px_m = px > 1e-8
	    px_sub = px[px_m]
	    qx_sub = qx[px_m]
	    sub_x_sub = sub_x[px_m]
	    sub_y_sub = sub_y[px_m]
	    
	    wi = np.sqrt(qx_sub/px_sub)
	    wipx = np.sqrt(qx*px)

            ## m0 = np.sum(wipx)
            ## mx = np.sum(sub_x*wipx)
            ## my = np.sum(sub_y*wipx)
            ## x0 = xi.copy()
            ## xi = np.array([mx, my])/m0

	    m0 = np.sum(wi)
	    x0 = xi.copy()
	    xi = np.array([ np.sum(sub_x_sub*wi), np.sum(sub_y_sub*wi) ])/m0
	    
            if np.abs(xi-x0).max() < 2:
                break

        seed.x = xi
        locs[n+1] = xi
        #h_new = 2 * np.sqrt(m0/nrm)
        #seed.h = np.array([h_new, h_new]) # bad!
        bws[n+1] = seed.h
    return locs, bws
            
            
            
            
