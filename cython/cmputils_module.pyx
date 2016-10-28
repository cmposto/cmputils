import cython
cimport numpy as np
np.import_array()

cdef extern from "${CMAKE_CURRENT_SOURCE_DIR}/../include/cmputils/dtw.h":
    double dtw(double* a, double* b, int size_a, int size_b, int warp_window, int verbose);

@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_dist(np.ndarray[double, ndim=1, mode="c"] a not None , np.ndarray[double, ndim=1, mode="c"] b not None, int warp_window, int verbose = 0):
    """
    dtw(a, b, warp_window, verbose)
    Takes two numpy arrays as input, and computes the dtw distance between them
    param: a - a numpy array
    param: b - a numpy array
    param: warp_window - the max number of elements part of a warp window (horizontal or vertical)
    param: verbose - between 0 and 2, different levels of logging
    """
    return dtw(<double*> np.PyArray_DATA(a), <double*> np.PyArray_DATA(b), a.shape[0], b.shape[0], warp_window, verbose)