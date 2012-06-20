from distutils.core import setup, Extension
from numpy.distutils.command import build_src
import Cython
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
from Cython.Distutils import build_ext
import Cython
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs) + ['mstools']
## dirs.extend(Cython.__path__)
## dirs.append('.')

## cell_label = Extension(
##     'mstools.mean_shift.cell_labels',
##     ['mstools/mean_shift/cell_labels.pyx'], 
##     include_dirs = dirs,
##     language='c++',
##     extra_compile_args=['-O3']
##     )
topological = Extension(
    'mstools.mean_shift.topological',
    ['mstools/mean_shift/topological.pyx'], 
    include_dirs = dirs,
    language='c++',
    extra_compile_args=['-O3']
    )
histogram = Extension(
    'mstools.mean_shift.histogram',
    ['mstools/mean_shift/histogram.pyx'], 
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )
gmshift = Extension(
    'mstools.mean_shift.grid_mean_shift',
    ['mstools/mean_shift/grid_mean_shift.pyx'], 
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )
indexing = Extension(
    'mstools.indexing',
    ['mstools/indexing.pyx'], 
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )

c_util = Extension(
    'mstools._c_util',
    ['mstools/_c_util.pyx'], 
    include_dirs = dirs, 
    extra_compile_args=['-O3']
    )

if __name__=='__main__':
    setup(
        name = 'mstools',
        version = '1.0',
        packages = ['mstools', 'mstools.mean_shift'],
        ext_modules = [ topological, histogram, indexing, gmshift, c_util],
        cmdclass = {'build_ext': build_ext}
    )
