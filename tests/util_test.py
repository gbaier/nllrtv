import numpy as np

import nllrtv.util as util
import pytest


def test_soft_thresh():
    arr = np.array([-2, -1, 0, 1, 2])
    thresh = 1
    des_arr = np.array([-1, 0, 0, 0, 1])
    np.testing.assert_array_equal(util.soft_thresh(arr, thresh), des_arr)


@pytest.mark.parametrize("shape, n_coords", [
    ((5, 10), 10),
    ((5, 30), 30),
    ((50, 300), 3000),
])
def test_random_coords(shape, n_coords):
    print(shape, n_coords)

    coords = util.random_coords(shape, n_coords)

    assert len(coords) == n_coords
    assert len(set(coords)) == n_coords  # no double entries

    for coord in coords:
        assert all(c >= 0 for c in coord)
        assert all(c < s for c, s in zip(coord, shape))


@pytest.mark.parametrize(
    "starts, stops, steps, des_idxs",
    [
        (0, (3,), 1, [(0,), (1,), (2,)]),
        (0, (3, 2), 1, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]),
        (0, (3,), 2, [(0,), (2,)]),
        (0, (4, 5), (2, 2), [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4)]),
        (1, (4, 5), (2, 2), [(1, 1), (1, 3), (3, 1), (3, 3)]),
        ((0, 1), (5, 5), (2, 2), [(0, 1), (0, 3), (2, 1), (2, 3), (4, 1), (4, 3)]),
    ],
)
def test_mdim_range(starts, stops, steps, des_idxs):
    assert list(util.mdim_range(starts, stops, steps)) == des_idxs


def test_mdim_range_raise_valueerror():
    with pytest.raises(ValueError):
        util.mdim_range(0, stops=(5, 3, 2), steps=(2, 2))


def test_mdim_range_raise_typeerror():
    with pytest.raises(TypeError):
        util.mdim_range(0, stops=(5, 3, 2), steps=3.2)
