""" general functions for dealing with tensors """

from itertools import product
import numbers

import numpy as np


def hyperrect_slice(coords, edge_lengths):
    """ extracts the hyperrectangle surrounding coord defined by edge_lenghts

    Paramters
    ---------

    coords : tuple of corner coordinates of the slice
    edge_lengths : tuple of same length as coords

    """

    if len(coords) != len(edge_lengths):
        raise ValueError(
            "length of coords {} and edges {} do not match".format(
                len(coords), len(edge_lengths)
            )
        )

    stops = tuple(s + el for s, el in zip(coords, edge_lengths))

    return tuple(slice(c, s) for c, s in zip(coords, stops))


def apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """ Almost a carbon copy of numpy.apply_along_axis.

    This implementation however applys a function along the axis of multiple arrays.
    In brief func1d(v1, v2, v3) where v1, v2, v3 are vectors along on axis
    of arrs = (arr1, arr2, arr3)

    """

    if not all(arrs[0].shape == arr.shape for arr in arrs):
        raise ValueError("all arrays must have the same shape")

    # arr, with the iteration axis at the end
    in_dims = list(range(next(iter(arrs)).ndim))

    axes = in_dims[:axis] + in_dims[axis + 1 :] + [axis]

    inarrs_view = [np.transpose(arr, axes) for arr in arrs]

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = np.ndindex(inarrs_view[0].shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError("Cannot apply_along_axis when any iteration dimensions are 0")
    res = np.asanyarray(func1d(*(iv[ind0] for iv in inarrs_view), *args, **kwargs))

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = np.zeros(inarrs_view[0].shape[:-1] + res.shape, res.dtype)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0:axis]
        + buff_dims[buff.ndim - res.ndim : buff.ndim]
        + buff_dims[axis : buff.ndim - res.ndim]
    )

    # matrices have a nasty __array_prepare__ and __array_wrap__
    if not isinstance(res, np.matrix):
        buff = res.__array_prepare__(buff)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = np.asanyarray(
            func1d(*(iv[ind] for iv in inarrs_view), *args, **kwargs)
        )

    # wrap the array, to preserve subclasses
    buff = res.__array_wrap__(buff)

    # finally, rotate the inserted axes back to where they belong
    return np.transpose(buff, buff_permute)
