""" tensor unit tests """

import pytest
import numpy as np

from nllrtv.nltensor import tensor


@pytest.mark.parametrize("coords, edge_lengths, des_slice", [
    ((0, 0), (2, 2), (slice(0, 2), slice(0, 2))),
    ((2, 3, 4), (1, 2, 3), (slice(2, 3), slice(3, 5), slice(4, 7))),
])
def test_hyperrect_slice(coords, edge_lengths, des_slice):
    assert tensor.hyperrect_slice(coords, edge_lengths) == des_slice


def test_apply_along_axis():
    t1 = np.arange(9).reshape((3, 3))
    t2 = np.ones_like(t1)

    def max_diff(v1, v2):
        """ maximum element-wise difference between two vectors """
        return np.abs(v1 - v2).max()

    np.testing.assert_array_equal(
        tensor.apply_along_axis(max_diff, 1, [t1, t2]), np.array((1, 4, 7)))

def test_apply_along_axis_raise():
    t1 = np.arange(9).reshape((3, 3))
    t2 = np.ones((3, 3, 2))

    def max_diff(v1, v2):
        """ maximum element-wise difference between two vectors """
        return np.abs(v1 - v2).max()

    with pytest.raises(ValueError):
        tensor.apply_along_axis(max_diff, 1, [t1, t2])
