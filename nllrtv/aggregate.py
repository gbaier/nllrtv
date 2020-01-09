""" functions for aggregating overlapping patches """

import itertools
import numbers

import numpy as np
from . import util


def overlap_aggregate(x_ov, chunks, depth):
    """ aggregates overlapping regions from neighboring chunks

   +--------------------+
   | a  b bbbbbbbb b  c | a -> 1
   | a  b bbbbbbbb b  c | b -> 123
   |   +------------+   | c -> 3
   | d |1 22222222 3| e | d -> 146
   | d |1 22222222 3| e | e -> 358
   |   |            |   | f -> 6
   | d |4          5| e | g -> 678
   | d |4          5| e | h -> 8
   | d |4          5| e |
   |   |            |   |
   | d |6 77777777 8| e |
   | d |6 77777777 8| e |
   |   +------------+   |
   | f  g gggggggg g  h |
   | f  g gggggggg g  h |
   +--------------------+

    """
    # shape of overlapped areas, i.e. the aa, bb, cc...
    ov_shs = overlap_shapes(chunks, depth)

    # start of overlapped areas
    ov_sts = overlap_starts(chunks, depth)

    # start of inner areas
    in_sts = inner_starts(chunks, depth)

    x_aggr = x_ov.copy()
    aggr_cnt = np.ones_like(x_aggr)

    for pos, in_st in in_sts.items():
        in_slice, ov_slice = valid_slices(x_ov.shape, chunks, pos)

        ov_shape = ov_shs[pos]
        ov_st = ov_sts[get_matching_ov(pos)]

        in_itp = block_index_tuples(ov_shape, x_aggr[in_slice].shape, in_st, chunks)
        ov_itp = block_index_tuples(ov_shape, x_ov[ov_slice].shape, ov_st, chunks)

        for ins, ovs in zip(in_itp, ov_itp):
            # Only aggregate valid entries
            # I.e. sometimes the overlapped area does not contain valid results.
            overlap = x_ov[ov_slice][ovs]
            mask = np.isfinite(overlap)

            x_aggr[in_slice][ins][mask] += overlap[mask]
            aggr_cnt[in_slice][ins][mask] += 1

    return x_aggr, aggr_cnt


def get_matching_ov(pos):
    match = {"l": "u", "m": "m", "u": "l"}
    return tuple(match[p] for p in pos)


def valid_slices(arr_shape, chunks, pos):
    """ returns index tuples which give valid views of the boundaries

    Parameters
    ----------
    arr_shape: tuple
        shape of the data
    chunks: tuple
        chunk dimension
    pos: string
       made up of 'l' 'm' 'u' delineating the position where data
       will be aggregated to


    """
    if len(arr_shape) != len(pos):
        raise ValueError("Dimensions do not match")

    inner_sel = {
        "l": lambda ax: (chunks[ax], arr_shape[ax]),
        "m": lambda ax: (0, arr_shape[ax]),
        "u": lambda ax: (0, arr_shape[ax] - chunks[ax]),
    }

    # selects the area containing the corresponding chunks
    outer_sel = {
        "l": lambda ax: (0, arr_shape[ax] - chunks[ax]),
        "m": lambda ax: (0, arr_shape[ax]),
        "u": lambda ax: (chunks[ax], arr_shape[ax]),
    }

    # inner slice
    inner_slice = tuple(slice(*inner_sel[p](ax)) for ax, p in enumerate(pos))

    # overlap slice
    overlap_slice = tuple(slice(*outer_sel[p](ax)) for ax, p in enumerate(pos))

    return inner_slice, overlap_slice


def _map_ndims(funcs, ndims):
    """ Applies functions too all corners and edges of a hypercube

    Used to compute coordinates depending on the corner or edge.

    Parameters
    ----------
    funcs: set of functions for lower, middle and upper positions
    ndims: integer, dimension of the hypercube

    """

    # lower, middle or upper position at a specific axis of the overlap region
    poss = [p for p in itertools.product("lmu", repeat=ndims) if p != ("m",) * ndims]

    return {
        pos: tuple(funcs[p](ax) for p, ax in zip(pos, range(ndims))) for pos in poss
    }


def inner_starts(chunk_shape, depth):
    sel = {
        "l": lambda ax: depth[ax],
        "m": lambda ax: depth[ax],
        "u": lambda ax: chunk_shape[ax] - 2 * depth[ax],
    }

    return _map_ndims(sel, len(chunk_shape))


def overlap_shapes(chunk_shape, depth):
    sel = {
        "l": lambda ax: depth[ax],
        "m": lambda ax: chunk_shape[ax] - 2 * depth[ax],
        "u": lambda ax: depth[ax],
    }

    return _map_ndims(sel, len(chunk_shape))


def overlap_starts(chunk_shape, depth):
    sel = {
        "l": lambda ax: 0,
        "m": lambda ax: depth[ax],
        "u": lambda ax: chunk_shape[ax] - depth[ax],
    }

    return _map_ndims(sel, len(chunk_shape))


def block_index_tuples(block_shape, arr_shape, starts=0, steps=1):
    """ yields index tuples to slice an array

    Parameters
    ----------

    """
    ndim = len(arr_shape)

    if not len(block_shape) == ndim:
        raise ValueError("`blcok_shape` is incompatible with `arr_shape`")

    if isinstance(steps, numbers.Number):
        steps = (steps,) * ndim
    if not len(steps) == ndim:
        raise ValueError("`steps` is incompatible with `arr_shape`")

    if isinstance(starts, numbers.Number):
        starts = (starts,) * ndim
    if not len(starts) == ndim:
        raise ValueError("`starts` is incompatible with `arr_shape`")

    block_starts = util.mdim_range(starts, arr_shape, steps)

    init_block_stop = tuple(s + b for s, b in zip(starts, block_shape))
    final_block_stop = tuple(s + 1 for s in arr_shape)
    block_stops = list(util.mdim_range(init_block_stop, final_block_stop, steps))

    # iterate over blocks in an array
    for b_sta, b_sto in zip(block_starts, block_stops):
        # print(b_sta, b_sto)
        # iterate over dimensions in a block
        yield tuple(slice(bx, by) for bx, by in zip(b_sta, b_sto))
