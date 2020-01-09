""" Functions for computing derivatives, divergence, ... """

import numpy as np
from scipy.ndimage import convolve1d


def forward(arr, ax):
    """forward difference

    .. math::
        d_n = x_{n+1} - x{n}

    Parameters
    ----------
    arr: array_like
    ax: integer
        along which axes to compute the difference

    Returns
    -------
    array_like
        forward difference

    """

    return convolve1d(arr, np.array([1, -1]), axis=ax)


def backward(arr, ax):
    """backward difference

    .. math::
        d_n = x_{n} - x{n-1}

    Parameters
    ----------
    arr: array_like
    ax: integer
        along which axes to compute the difference

    Returns
    -------
    array_like
        backward difference

    """

    return convolve1d(arr, np.array([1, -1]), axis=ax, origin=-1)


def divergence(arr, axes=None):
    """ Divergence of arr

    Parameters
    ----------
    arr: array_like
    axes: list or tuple
        must have same length as arr has dimensions.
        Along which axes to compute the difference for the nth dimension.

    Returns
    -------
    array_like
        divergence

    """
    if axes is None:
        axes = range(arr.ndim)
    return sum(backward(db, ax) for db, ax in zip(arr, axes))


def ndim_discrete_laplacian(axes, ndim=None):
    """ returns an n-dimensional discrete Laplace operator

    Parameters
    ----------

    axes: list or tuple
        axes that are considered for the multidimensional Laplace operator

    """

    if ndim is None:
        ndim = max(axes) + 1
    if max(axes) >= ndim:
        raise ValueError("number of dimensions and axes do not match")

    kernel = np.zeros(ndim * (3,))
    kernel[(1,) * ndim] = -2 * len(axes)

    for ax in axes:
        lower = (1,) * ax
        upper = (1,) * (ndim - ax - 1)

        kernel[(*lower, 0, *upper)] = 1
        kernel[(*lower, 2, *upper)] = 1

    return kernel
