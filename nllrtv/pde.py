""" Implements the Jacobi method for solving partial differential equations """

from scipy.ndimage import convolve


def jacobi(x, kernel, b):
    """ computes one step of the Jacobi method solving

    .. math::
        k \ast x = b,

    where :math:`\ast` denotes convolution.

    Check https://en.wikipedia.org/wiki/Jacobi_method for more details.

    Parameters
    ----------

    x: array_like
        solution of current iteration
    kernel: array_like
        convolutional kernel
    b: array_like
        right-hand side vector b

    """

    if kernel.ndim > 2:
        raise ValueError("kernel must be one- or two-dimensional")
    if len(set(kernel.shape)) != 1:
        raise ValueError("kernel must have dimensions of same size")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("kernel must be of odd size")

    center = (kernel.shape[0] // 2,) * kernel.ndim

    k = kernel.copy()
    k[center] = 0

    return (b - convolve(x, k, mode="constant", cval=0)) / kernel[center]
