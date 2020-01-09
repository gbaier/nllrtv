import itertools
import numbers
import random

import numpy as np


def soft_thresh(arr, thresh):
    """ soft thresholding function """
    shrink = np.abs(arr) - thresh
    shrink[shrink < 0] = 0

    return np.sign(arr) * shrink


def svd_soft_thresh(arr, thresh):
    """
    Apply the shrinkage operator to the singular values of arr.
    The parameter tau is used as the scaling parameter to the shrink function.
    Returns the matrix obtained by computing U * shrink(s) * V where
        U are the left singular vectors
        V are the right singular vectors
        s are the singular values as a diagonal matrix

    Taken and slightly modified from
    https://github.com/nwbirnie/rpca/blob/master/rpca.py.

    """

    U, s, Vh = np.linalg.svd(arr, full_matrices=False)

    return U @ np.diag(soft_thresh(s, thresh)) @ Vh


def random_coords(shape, k):
    """ return k random coordinates inside the interval defined by shape

    Parameters
    ----------

    shape: tuple or list
        defines the interval of the random coordinates
    k: int
        how many samples to draw

    """

    pos_coords = list(itertools.product(*(range(x) for x in shape)))
    return random.sample(pos_coords, k)


def mdim_range(starts, stops=None, steps=1):
    """
    multidimensional range.
    mdim_range(stops)
    mdim_range(starts, stops[, steps])

    Parameters
    ----------
    shape : tuple
        shape of the tensor
    step : integer or tuple of the same length as shape
        Indicates step size of the iterator.
        If integer is given, then the step is uniform in all dimensions.
    """

    if stops is None:
        starts, stops = 0, starts

    ndim = len(stops)

    if isinstance(steps, numbers.Number):
        steps = (steps,) * ndim
    if not len(steps) == ndim:
        raise ValueError("`steps` is incompatible with `stops`")

    if isinstance(starts, numbers.Number):
        starts = (starts,) * ndim
    if not len(starts) == ndim:
        raise ValueError("`starts` is incompatible with `stops`")

    return itertools.product(*(range(*sss) for sss in zip(starts, stops, steps)))
