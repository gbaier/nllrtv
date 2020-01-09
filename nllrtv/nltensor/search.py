""" routines for search windows """

import numbers
from collections import namedtuple

from . import tensor
from .. import util

SearchWinEl = namedtuple("SearchWinEl", "patch idxs")


def get_rect_window_patches(arr_in, corner_coordinates, win_shape, pat_shape, steps=1):
    """ extracts the patches of a rectangular search window

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    coordner_coordinates : lower corner coordinates oif the patche
    win_shape : integer or tuple of length arr_in.ndim
        Indicates the shape of the search window.
    pat_shape : integer or tuple of length arr_in.ndim
        Indicates the shape of the patches that are extracted from the
        rectangular search window
    steps : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Notes
    -----

    Inspired and partly copied from scikit-image.util.view_as_windows

    """

    ndim = arr_in.ndim

    if isinstance(win_shape, numbers.Number):
        win_shape = (win_shape,) * ndim
    if not len(win_shape) == ndim:
        raise ValueError("`win_shape` is incompatible with `arr_in.shape`")

    if isinstance(pat_shape, numbers.Number):
        pat_shape = (pat_shape,) * ndim
    if not len(pat_shape) == ndim:
        raise ValueError("`pat_shape` is incompatible with `arr_in.shape`")

    if isinstance(steps, numbers.Number):
        if steps < 1:
            raise ValueError("`step` must be >= 1")
        steps = (steps,) * ndim
    if len(steps) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    # distance from the patch corner coordinates to the window corner coordinates
    win_lower_dist = tuple((w - p) // 2 for w, p in zip(win_shape, pat_shape))

    # window corner coordinates
    wcc = tuple(max(p - w, 0) for p, w in zip(corner_coordinates, win_lower_dist))

    # window stop coordinates for patch coordinates
    stops = tuple(
        min(p + w + 1, dim)
        for p, w, dim in zip(corner_coordinates, win_lower_dist, arr_in.shape)
    )

    for idx in util.mdim_range(wcc, stops, steps):
        sub = tensor.hyperrect_slice(idx, pat_shape)
        yield SearchWinEl(arr_in[sub], sub)
