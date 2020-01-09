import numpy as np

from nllrtv import pde


def test_jacobi():
    """ 1D Laplace, solve Ax = b """

    b = np.array([2, 0, 0, 0, 0, 0, 2], dtype=np.float)
    kernel = np.array([1, -2, 1], dtype=np.float)  # discrete laplacian

    # matrix form of convolution with discrete laplacian
    A = np.array(
        [
            [-2, 1, 0, 0, 0, 0, 0],
            [1, -2, 1, 0, 0, 0, 0],
            [0, 1, -2, 1, 0, 0, 0],
            [0, 0, 1, -2, 1, 0, 0],
            [0, 0, 0, 1, -2, 1, 0],
            [0, 0, 0, 0, 1, -2, 1],
            [0, 0, 0, 0, 0, 1, -2],
        ],
        dtype=np.float,
    )

    x = np.zeros_like(b)
    for _ in range(1000):
        x = pde.jacobi(x, kernel, b)

    np.testing.assert_array_almost_equal(A @ x, b)

    # Analytical solution
    np.testing.assert_array_almost_equal(x, -2*np.ones_like(x))
