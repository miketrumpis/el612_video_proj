import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import datad

def npt_colormap(N, name='jet'):
    try:
        color_spec = datad.get(name)
    except KeyError:
        raise ValueError('Bad colormap name: %s'%name)
    return LinearSegmentedColormap('new_%s'%name, color_spec, N)
            
