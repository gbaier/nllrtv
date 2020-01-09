""" Implementation of split Bregman methods for solving various TV and :math:`L_1` or :math:`L_2` problems

[1] Goldstein, T & Osher, S., The Split Bregman method for L1 regularized problems,
    2009, SIAM J Imaging Sci. 2. 323-343.

"""

import numpy as np
from scipy.ndimage import convolve

from .util import soft_thresh
from . import diff
from . import pde


def jacobi(u, image, mu, beta, d_s, b_s, axes):
    r""" performs one iteration step of the Jacobi method to solve


    .. math::

        \arg \min_u \frac{\mu}{2} \lVert u - f \rVert_2^2 + \sum \lvert \nabla u \rvert

    Parameters
    ----------

    u: array_like
        solution of current iteration
    image: array_like
        noisy image
    mu: float
        regularization parameter for the data fitting term
    beta: float
        lagrange multipliers for the TV norm
    d_s: array_like
        decoupled TV variables
    b_s: array_like
        bregman update variables
    axes: list or tuple
        along which axes to compute total variation

    """

    b = mu * image - beta * diff.divergence(d_s - b_s, axes)

    kernel = -beta * diff.ndim_discrete_laplacian(axes, ndim=image.ndim)
    kernel[(1,) * kernel.ndim] += mu

    return pde.jacobi(u, kernel, b)


def jacobi_ln(u, image, mu, lbda, alpha, beta, d_s, w, b_s, b_w, axes):
    """
    Parameters
    ----------

    u: array_like
        solution of current iteration
    image: array_like
        noisy image
    mu: float
        regularization parameter for the data fitting term
    alpha: float
        lagrange multipliers for the L1, L2 term
    beta: float
        lagrange multipliers for the TV norm
    d_s: array_like
        decoupled TV variables
    b_s: array_like
        bregman update variables
    b_w: array_like
        bregman update variables
    axes: list or tuple
        along which axes to compute total variation

    """
    b = mu * image + alpha * (w - b_w) - beta * diff.divergence(d_s - b_s, axes)

    kernel = -beta * diff.ndim_discrete_laplacian(axes, ndim=image.ndim)
    kernel[(1,) * kernel.ndim] += mu + alpha

    return pde.jacobi(u, kernel, b)


def tv(image, mu, beta, axes=None, max_iter=20):
    r""" Split Bregman isotropic TV Denoising from [1]

    This code solves the optimization problem

    .. math::
        \arg \min_u \frac{\mu}{2} \lVert u - f \rVert_2^2 + \sum \lvert \nabla u \rvert ,

    where :math:`f` is the original, noisy image.
    So a square error as a data fidelity term and a and total variation as regulizers.
    Total variation is applied along axes.

    Parameters
    ----------

    image: numpy array
        the image to denoise
    mu: float
        regularization parameter for the data fitting term
    beta: float
        lagrange multipliers for the TV norm
    axes: tuple
        along which axes to compute total variation
    max_iter: int
        number of iterations

    """

    if axes is None:
        axes = tuple(range(image.ndim))

    u = image.copy()

    # latent variables for separate optimization
    d_s = np.zeros((len(axes), *image.shape), image.dtype)

    # bregman iteration variables
    b_s = np.zeros_like(d_s)

    for _ in range(max_iter):
        # Really odd, 1 or 3 or more iterations lead so slight
        # checkerboard artifacts in some regions
        for _ in range(2):
            # solve u sub problem
            u = jacobi(u, image, mu, beta, d_s, b_s, axes)

            gu_s = np.array([diff.forward(u, ax) for ax in axes])

            # solve d sub problem
            d_s = optimal_ds(gu_s, b_s, beta)

        # update bs
        b_s += gu_s - d_s

    return u


def _ln_tv(image, mu, lbda, alpha, beta, axes, max_iter, norm):
    def optimal_w(u, b_w, lbda, beta):
        if norm == "l1":
            return soft_thresh(u + b_w, lbda / beta)
        if norm == "l2":
            return (u + b_w) * (beta / (lbda + beta))
        raise ValueError("norm must be l1 or l2")

    if axes is None:
        axes = tuple(range(image.ndim))

    u = image.copy()

    # latent variables for separate optimization
    d_s = np.zeros((len(axes), *image.shape), image.dtype)
    w = np.zeros_like(image)

    # bregman iteration variables
    b_s = np.zeros_like(d_s)
    b_w = np.zeros_like(image)

    for _ in range(max_iter):
        # Really odd, 1 or 3 or more iterations lead so slight
        # checkerboard artifacts in some regions. BUT only for L2 not L1
        for _ in range(2):
            # solve u sub problem
            u = jacobi_ln(u, image, mu, lbda, alpha, beta, d_s, w, b_s, b_w, axes)

            gu_s = np.array([diff.forward(u, ax) for ax in axes])

            # solve d sub problem
            d_s = optimal_ds(gu_s, b_s, beta)

            w = optimal_w(u, b_w, lbda, alpha)

        # update bs
        b_s += gu_s - d_s
        b_w += u - w

    return u


def l1_tv(image, mu, lbda, alpha, beta, axes=None, max_iter=20):
    r""" Unconstrained CS Optimization Algorithm from [1]

    This code solves the optimization problem

    .. math::
        \arg \min_u \frac{\mu}{2} \lVert u - f \rVert_2^2 + \lambda \lVert u \rVert_1 + \sum \lvert \nabla u \rvert ,

    where :math:`f` is the original, noisy image.
    So a square error as a data fidelity term and a sparsity inducing L1 and total variation as regulizers.
    Total variation is applied along axis.

    Parameters
    ----------

    image: numpy array
         the image to denoise
    mu: float
         regularization parameter for the data fitting term
    lbda: float
         regularization parameter for the L1 sparsity inducing term
    alpha: float
         lagrange multiplier for the L1 term
    beta: float
         lagrange multipliers for the TV norm
    axes: list or tuple
         along which axes to compute total variation
    max_iter: int
         number of iterations

    """

    return _ln_tv(image, mu, lbda, alpha, beta, axes, max_iter, "l1")


def l2_tv(image, mu, lbda, alpha, beta, axes=None, max_iter=20):
    r""" Unconstrained CS Optimization Algorithm from [1]

    This code solves the optimization problem

    .. math::
        \arg \min_u \frac{\mu}{2} \lVert u - f \rVert_2^2 + \frac{\lambda}{2} \lVert u \rVert_2^2 + \sum \lvert \nabla u \rvert ,

    where :math:`f` is the original, noisy image.
    So a square error as a data fidelity term and a sparsity inducing L1 and total variation as regulizers.
    Total variation is applied along axis.

    Parameters
    ----------

    image: numpy array
         the image to denoise
    mu: float
         regularization parameter for the data fitting term
    lbda: float
         regularization parameter for the L2 ridge regression
    alpha: float
         lagrange multiplier for the L2 term
    beta: float
         lagrange multipliers for the TV norm
    axes: list or tuple
         along which axes to compute total variation
    max_iter: int
         number of iterations

    """

    return _ln_tv(image, mu, lbda, alpha, beta, axes, max_iter, "l2")


def optimal_ds(gu_s, b_s, alpha):
    """ finds the optimal d variables for isotropic TV

    Check in [1] the algorithms for
        "Isotropic TV denoising" and "Unconstrained CS Optimization"

    """

    t_s = gu_s + b_s

    s_k = np.sqrt(np.sum(np.abs(t_s) ** 2, axis=0))

    scale = np.divide(
        soft_thresh(s_k, 1 / alpha), s_k, out=np.zeros_like(s_k), where=s_k != 0
    )
    return scale * t_s
