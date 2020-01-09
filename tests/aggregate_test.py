import numpy as np

import pytest
from nllrtv import aggregate as aggr


def test_block_index_tuples():
    block_shape = (4, 2)
    arr_shape = (8, 6)
    steps = (4, 3)
    starts = (0, 0)

    des_block_index_tuples = [
        (slice(0, 4), slice(0, 2)),
        (slice(0, 4), slice(3, 5)),
        (slice(4, 8), slice(0, 2)),
        (slice(4, 8), slice(3, 5)),
    ]
    assert (
        list(aggr.block_index_tuples(block_shape, arr_shape, starts, steps))
        == des_block_index_tuples
    )


def test_overlap_shapes():
    chunk_shape = (11, 12)
    depth = {0: 2, 1: 1}
    des_overlap_shapes = {
        ("l", "l"): tuple(depth.values()),
        ("l", "m"): (2, 10),
        ("l", "u"): tuple(depth.values()),
        ("m", "l"): (7, 1),
        ("m", "u"): (7, 1),
        ("u", "l"): tuple(depth.values()),
        ("u", "m"): (2, 10),
        ("u", "u"): tuple(depth.values()),
    }

    assert aggr.overlap_shapes(chunk_shape, depth) == des_overlap_shapes


def test_overlap_starts():
    chunk_shape = (11, 12)
    depth = {0: 2, 1: 1}
    des_overlap_shapes = {
        ("l", "l"): (0, 0),
        ("l", "m"): (0, 1),
        ("l", "u"): (0, 11),
        ("m", "l"): (2, 0),
        ("m", "u"): (2, 11),
        ("u", "l"): (9, 0),
        ("u", "m"): (9, 1),
        ("u", "u"): (9, 11),
    }

    assert aggr.overlap_starts(chunk_shape, depth) == des_overlap_shapes


def test_inner_starts():
    chunk_shape = (11, 12)
    depth = {0: 2, 1: 1}
    des_inner_starts = {
        ("l", "l"): (2, 1),
        ("l", "m"): (2, 1),
        ("l", "u"): (2, 10),
        ("m", "l"): (2, 1),
        ("m", "u"): (2, 10),
        ("u", "l"): (7, 1),
        ("u", "m"): (7, 1),
        ("u", "u"): (7, 10),
    }

    assert aggr.inner_starts(chunk_shape, depth) == des_inner_starts


@pytest.mark.parametrize(
    "pos, des_inner_slice, des_overlap_slice",
    [
        ("ll", (slice(4, 16), slice(6, 18)), (slice(0, 12), slice(0, 12))),
        ("lm", (slice(4, 16), slice(0, 18)), (slice(0, 12), slice(0, 18))),
        ("lu", (slice(4, 16), slice(0, 12)), (slice(0, 12), slice(6, 18))),
        ("ml", (slice(0, 16), slice(6, 18)), (slice(0, 16), slice(0, 12))),
        ("mu", (slice(0, 16), slice(0, 12)), (slice(0, 16), slice(6, 18))),
        ("ul", (slice(0, 12), slice(6, 18)), (slice(4, 16), slice(0, 12))),
        ("um", (slice(0, 12), slice(0, 18)), (slice(4, 16), slice(0, 18))),
        ("uu", (slice(0, 12), slice(0, 12)), (slice(4, 16), slice(6, 18))),
    ],
)
def test_valid_slices(pos, des_inner_slice, des_overlap_slice):
    arr_shape = (16, 18)
    chunks = (4, 6)

    assert aggr.valid_slices(arr_shape, chunks, pos) == (
        des_inner_slice,
        des_overlap_slice,
    )


def test_aggr_add():
    in_arr = np.array(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    )
    chunks = (3, 3)
    des_out_arr = np.array(
        [[0, 1, 2, 3, 4, 5], [6, 8, 8, 9, 9, 11], [12, 13, 14, 15, 16, 17]]
    )

    depth = {0: 1, 1: 1}

    out_arr, aggr_cnt = aggr.overlap_aggregate(in_arr, chunks, depth)

    np.testing.assert_array_equal(out_arr / aggr_cnt, des_out_arr)


def test_aggr_add_large():
    in_arr = np.arange(180).reshape((18, 10))
    chunks = (9, 5)
    depth = {0: 2, 1: 1}

    des_out_arr = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 48, 24, 25, 50, 27, 28, 29],
            [30, 31, 32, 68, 34, 35, 70, 37, 38, 39],
            [40, 41, 42, 88, 44, 45, 90, 47, 48, 49],
            [50, 142, 144, 296, 54, 55, 300, 154, 156, 59],
            [60, 162, 164, 336, 64, 65, 340, 174, 176, 69],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [110, 182, 184, 376, 114, 115, 380, 194, 196, 119],
            [120, 202, 204, 416, 124, 125, 420, 214, 216, 129],
            [130, 131, 132, 268, 134, 135, 270, 137, 138, 139],
            [140, 141, 142, 288, 144, 145, 290, 147, 148, 149],
            [150, 151, 152, 308, 154, 155, 310, 157, 158, 159],
            [160, 161, 162, 163, 164, 165, 166, 167, 168, 169],
            [170, 171, 172, 173, 174, 175, 176, 177, 178, 179],
        ]
    )

    des_aggr_cnt = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 2, 2, 4, 1, 1, 4, 2, 2, 1],
            [1, 2, 2, 4, 1, 1, 4, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 4, 1, 1, 4, 2, 2, 1],
            [1, 2, 2, 4, 1, 1, 4, 2, 2, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 2, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    out_arr, aggr_cnt = aggr.overlap_aggregate(in_arr, chunks, depth)

    np.testing.assert_array_equal(aggr_cnt, des_aggr_cnt)
    np.testing.assert_array_equal(out_arr, des_out_arr)
