import contextlib
import time

import numpy as np
from numpy import testing
from numpy.testing import assert_array_almost_equal as assert_close

from scipy.ndimage import convolve as ndconvolve
from skimage.convolution import pyconvolve

try:
    import cv
    opencv_available = True
except ImportError:
    opencv_available = False


kernel = np.array([
    [20,  50,  80,  50, 20, 1],
    [50, 100, 140, 100, 50, 1],
    [90, 160, 200, 160, 90, 1],
    [50, 100, 140, 100, 50, 1],
    [20,  50,  80,  50, 20, 1]], dtype=np.float32)


@contextlib.contextmanager
def timed_exec(times):
    """Add time used inside a with-block to list of `times`.

    From ActiveState Code Recipe 498113
    """
    start = time.clock()
    try:
        yield
    finally:
        end = time.clock()
        times.append(end - start)


def profile():
    a = np.random.randn(1000, 1000).astype(np.float32)

    out_skimage = np.zeros_like(a)
    out_opencv = np.zeros_like(a)

    times = []
    pkgs = ['skimage', 'ndimage', 'opencv']

    anchor = (0, 0)
    with timed_exec(times):
        pyconvolve(a, out_skimage, kernel, anchor=anchor)

    with timed_exec(times):
        out_ndimage = ndconvolve(a, kernel)

    if opencv_available:
        with timed_exec(times):
            cv.Filter2D(a, out_opencv, kernel, anchor=anchor)

    print "Timings:"
    print "~" * 20
    for p, t in zip(pkgs, times):
        print '%s: %.3f secs' % (p, t)

    if opencv_available:
        assert_close(out_skimage, out_opencv)
    # Convolution output from skimage does not match that of scipy.ndimage.
    #assert_close(out_skimage, out_ndimage)


def test_vs_opencv():

    for i in range(10):
        kx, ky = [np.random.randint(1, 10) for k in range(2)]
        image_x, image_y = [np.random.randint(1, 1000) for k in range(2)]
        kernel = np.random.randn(ky, kx).astype(np.float32)
        image = np.random.randn(image_y, image_x).astype(np.float32)

        sci_out = np.empty_like(image)
        cv_out = np.empty_like(image)
        anchor = (np.random.randint(0, kx),np.random.randint(0, ky))

        pyconvolve(image, sci_out, kernel, anchor=anchor)

        cv.Filter2D(image, cv_out, kernel, anchor=anchor)
        assert_close(sci_out, cv_out)


if __name__ == '__main__':
    profile()
    testing.run_module_suite()

