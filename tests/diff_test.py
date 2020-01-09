import numpy as np

from nllrtv import diff


def test_forward():
    in_arr = np.array([[0, 1, 2, 3],
                       [4, 5, 5, 5],
                       [6, 7, 8, 9]])

    np.testing.assert_array_equal(
        diff.forward(in_arr, 0), np.array([[4, 4, 3, 2],
                                           [2, 2, 3, 4],
                                           [0, 0, 0, 0]])
    )
    np.testing.assert_array_equal(
        diff.forward(in_arr, 1), np.array([[1, 1, 1, 0],
                                           [1, 0, 0, 0],
                                           [1, 1, 1, 0]])
    )


def test_backward():
    in_arr = np.array([[0, 1, 2, 3],
                       [4, 5, 5, 5],
                       [6, 7, 8, 9]])

    np.testing.assert_array_equal(
        diff.backward(in_arr, 0), np.array([[0, 0, 0, 0],
                                            [4, 4, 3, 2],
                                            [2, 2, 3, 4]])
    )
    np.testing.assert_array_equal(
        diff.backward(in_arr, 1), np.array([[0, 1, 1, 1],
                                            [0, 1, 0, 0],
                                            [0, 1, 1, 1]])
    )


def test_divergence():
    in_arr = np.array(
        [
            [[0, 1, 2, 3],
             [4, 5, 5, 5],
             [6, 7, 8, 9]],
            [[0, 1, 2, 3],
             [4, 5, 5, 5],
             [6, 7, 8, 9]],
        ]
    )

    # Compare result to test_backward.
    # This is the sum of both results of test_backward
    des_out = np.array([[0, 1, 1, 1],
                        [4, 5, 3, 2],
                        [2, 3, 4, 5]])

    np.testing.assert_array_equal(diff.divergence(in_arr), des_out)


def test_divergence_1dim():
    in_arr = np.array([[[0, 1, 2, 3],
                        [4, 5, 5, 5],
                        [6, 7, 8, 9]]])

    np.testing.assert_array_equal(
        diff.divergence(in_arr, axes=(0,)),
        np.array([[0, 0, 0, 0],
                  [4, 4, 3, 2],
                  [2, 2, 3, 4]]),
    )
    np.testing.assert_array_equal(
        diff.divergence(in_arr, axes=(1,)),
        np.array([[0, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 1, 1, 1]]),
    )


def test_ndim_gauss_seidel_kernel():
    np.testing.assert_array_equal(
        diff.ndim_discrete_laplacian(axes=(0, 1)),
        np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]),
    )
    np.testing.assert_array_equal(
        diff.ndim_discrete_laplacian(axes=(1,), ndim=2),
        np.array([[0, 0, 0],
                  [1, -2, 1],
                  [0, 0, 0]]),
    )
    np.testing.assert_array_equal(
        diff.ndim_discrete_laplacian(axes=(0,), ndim=2),
        np.array([[0, 1, 0],
                  [0, -2, 0],
                  [0, 1, 0]]),
    )
    np.testing.assert_array_equal(
        diff.ndim_discrete_laplacian(axes=(0, 1, 2)),
        np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ]
        ),
    )
