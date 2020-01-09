import numpy as np
import pytest

from nllrtv.nltensor import search


@pytest.mark.parametrize("win_shape", [(3, 3), (5, 5)])
def test_get_rect_window_0(win_shape):
    arr_in = np.arange(9).reshape((3, 3))
    corner_coordinates = (1, 1)
    pat_shape = (1, 1)

    search_window = list(
        search.get_rect_window_patches(arr_in, corner_coordinates, win_shape,
                                       pat_shape))

    patches = [x.patch for x in search_window]

    assert patches == [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[2]]),
        np.array([[3]]),
        np.array([[4]]),
        np.array([[5]]),
        np.array([[6]]),
        np.array([[7]]),
        np.array([[8]]),
    ]


@pytest.mark.parametrize("corner_coordinates, des_patches", [
    ((0, 0), [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[5]]),
        np.array([[6]]),
    ]),
    ((1, 1), [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[2]]),
        np.array([[5]]),
        np.array([[6]]),
        np.array([[7]]),
        np.array([[10]]),
        np.array([[11]]),
        np.array([[12]])
    ]),
])
def test_get_rect_window_1(corner_coordinates, des_patches):
    arr_in = np.arange(25).reshape((5, 5))
    win_shape = (3, 3)
    pat_shape = (1, 1)

    search_window = list(
        search.get_rect_window_patches(arr_in, corner_coordinates, win_shape,
                                       pat_shape))
    patches = [x.patch for x in search_window]

    assert patches == des_patches


def test_get_rect_window_2():
    arr_in = np.arange(24).reshape((4, 6))
    corner_coordinates = (2, 2)
    win_shape = (6, 6)
    pat_shape = (2, 2)
    steps = (2, 2)

    search_window = list(
        search.get_rect_window_patches(arr_in, corner_coordinates, win_shape,
                                       pat_shape, steps))
    patches = [x.patch for x in search_window]
    des_patches = [
        np.array([[0, 1], [6, 7]]),
        np.array([[2, 3], [8, 9]]),
        np.array([[4, 5], [10, 11]]),
        np.array([[12, 13], [18, 19]]),
        np.array([[14, 15], [20, 21]]),
        np.array([[16, 17], [22, 23]]),
    ]
    for p, dp in zip(patches, des_patches):
        np.testing.assert_array_equal(p, dp)
