from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("cmputils", sources=["cmputils_module.pyx"],
                           libraries=["${CMAKE_CURRENT_BINARY_DIR}/../${LIBRARY_NAME}"],
                           language='c++', extra_compile_args=["-std=c++11"])],
    include_dirs=[numpy.get_include()]
)