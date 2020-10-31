import numba
import numpy as np


@numba.njit(parallel=True)
def increase_histogram(hist: np.array, image: np.array, selection: np.array):
    """Takes a histogram and an image. Increases histogram counters for this image in-place.

    Parameters
    ----------
    hist : np.array(shape=(H, W, 255), dtype=np.float32)
        Histogram that will be changed in-place.
    image : np.array(shape=(H, W))
        Counts for color values from this image will be increased in the histogram.
    selection : np.array(shape=(H, W), dtype=np.bool)
        Image mask for pixels that should be used.
    """
    # Ignore false positive, see: github.com/PyCQA/pylint/issues/2910
    # pylint: disable=not-an-iterable
    for x in numba.prange(image.shape[0]):
        for y in numba.prange(image.shape[1]):
            value = image[x, y]
            if selection[x, y]:
                hist[x, y, value] += 1.0
