"""weighted nuclear norm minimization """

import logging

import numpy as np
import scipy.linalg as la

from skimage.restoration import denoise_tv_chambolle

from . import bregman
from . import util

logger = logging.getLogger(__name__)


def wnnm(Y, w):
    r""" performs weighted nuclear norm minimization with an L2 data fidelity term

    .. math::
        \arg \min_X \lVert Y - X \rVert^2_F + \lVert X \rVert_{w, \ast}
    """

    ws = np.sort(w)

    if np.all(w != ws):
        raise ValueError("weights are not sorted")

    return util.svd_soft_thresh(Y, w / 2)


def rpca(M, max_iter=100):
    r""" robust principal component analysis

    solves

    .. math::
        \min_{S, L} \lVert L \rVert_{*} + \lambda \lVert S \rVert_1, \text{ s.t. } M = L + S

    [1] Candes et. al, "Robust Principal Component Analysis?", 2009

    Parameters
    ----------

    M: array_like
        observed image
    max_iter: int
        number of iterations

    """

    lbda = 1 / np.sqrt(np.max(M.shape))  # taken from [1]

    # Taken from [1], chapter 5 Algorithms. This uses the entrywise L1 norm, which is also defined
    # in [1]. Not the induced L1 matrix norm, i.e. the maximum over the column sum.
    mu = np.prod(M.shape) / (4 * np.sum(np.abs(M)))

    # stopping criteria
    theta = 1e-7
    S = Y = np.zeros_like(M)
    for itr in range(max_iter):
        L = util.svd_soft_thresh(M - S + Y / mu, 1 / mu)
        S = util.soft_thresh(M - L + Y / mu, lbda / mu)
        Y = Y + mu * (M - L - S)

        # stopping criterion
        rel_err = la.norm(M - L - S, "fro") / la.norm(M, "fro")
        logger.debug("relative error: {:9.7f}".format(rel_err))
        if rel_err < theta:
            logger.debug(
                "relative error {:.2e} smaller than {:.2e}. Stopping optimization.".format(
                    rel_err, theta
                )
            )
            break

    return L, S


def mat_comp(M, mask, tau, max_iter=100, eps=1e-4):
    r""" matrix completion using the algorithm introduced in [1] Algorithm 1

    [1] Cai et. al, "A Singular Value Thresholding Algorithm for Matrix Completion", 2010

    Parameters
    ----------

    M: 2dim numpy array
        incomplete matrix
    mask: 2dim boolean array
        where entries of M are valid
    tau: float
        regularization parameter
    eps: float
        stopping criterion

    """

    weights = tau * np.ones(np.min(M.shape))

    delta = 1.2 * M.size / np.sum(mask)  # taken from [1]
    delta = 2  # convergence guaraenteed

    # Eq. 5.3 in [1]
    k0 = np.ceil(tau / (delta * la.norm(M[mask])))

    Y = np.zeros_like(M)
    Y[mask] = k0 * delta * M[mask]

    for idx in range(max_iter):
        X = wnnm(Y, weights)
        diff = M - X
        err = la.norm(diff[mask]) / la.norm(M[mask])
        logger.debug("{:>3d} error={:>8.6f}".format(idx, err))
        if err < eps:
            break
        Y[mask] += delta * diff[mask]

    return X


def rpca_wnnm(Y, w, mu=1, max_iter=100):
    r""" weighted nuclear norm minimization for robust principal component analysis

    [1] Gu et. al, "Weighted Nuclear Norm Minimization and Its Applications
        to Low Level Vision", 2017
    [2] Boyd et. al, "Distributed Optimization and Statistical Learning via
        the Alternating Direction Method of Multipliers". 2010

    .. math::
        \min_{E, X} \lVert X \rVert_{w,*} + \lVert E \rVert_1, \text{ s.t. } Y = X + E

    Parameters
    ----------

    Y: 2 dimensional numpy array
        the matrix that is to be decomposed
    w: 1 dimensional numpy array
        the weighting vector for the singular values
    mu: float
        Lagrange multiplier FIXME

    """

    # initialization
    X = Y.copy()
    L = E = np.zeros_like(Y)

    # stopping criteria
    theta = 1e-7

    rho = 1.05
    for _ in range(max_iter):
        # update X
        X = wnnm(Y + L / mu - E, w * 2 / mu)

        # update E
        E = util.soft_thresh(Y + L / mu - X, 1 / mu)

        # update L
        L = L + mu * (Y - X - E)

        # Updating mu seems to be an extension of the original ADMM [2]
        mu *= rho
        # logger.info('mu: {:7.4f}'.format(mu))

        # stopping criterion
        rel_err = la.norm(Y - X - E, "fro") / la.norm(Y, "fro")
        logger.debug("relative error: {:9.7f}".format(rel_err))
        # print('relative error: {:9.7f}'.format(rel_err))
        if rel_err < theta:
            logger.debug(
                "relative error {:.2e} smaller than {:.2e}. Stopping optimization.".format(
                    rel_err, theta
                )
            )
            break

    return X, E


def rpca_wnnm_tv(
    g, w, lbda, mu, nu, alpha, beta, gamma, axes, max_iter=100, noise_norm="l1"
):
    r""" weighted nuclear norm minimization with L1 sparsity and TV regularization

    .. math::
        \arg \min_{U, N, S} \lVert U \rVert_{w,} + \lambda \sum \lvert \nabla U \rvert +
        \mu \lVert N \rVert_1 + \nu \sum \lvert \nabla N \rvert + \lVert S \rVert_1 ,

    subject to :math:`G = U + N + S`, where :math:`G` is the received signal,
    :math:`U` the true reflectivity, :math:`N` captures speckle noise, and :math:`S` outliers.

    [1] Gu et. al, "Weighted Nuclear Norm Minimization and Its Applications
        to Low Level Vision", 2017
    [2] Lin et. al, "The Augmented Lagrange Multiplier Method for Exact Recovery of
        Corrupted Low-Rank Matrices", 2010
    [3] Goldstein et. al, "The split Bregman method for L1 regularized problems",
        2009


    Parameters
    ----------

    g: array_like
        observed image
    w: array_like
        weighting vector for g's singular values
    lbda: float
        regularization parameter for :math:`B` total variation
    mu: float
        regularization parameter for :math:`N` L1 penalty
    nu: float
        regularization parameter for :math:`N` total variation
    alpha: float
        lagrange multiplier for :math:`G=U+N+S` constraint
    beta: float
        lagrange multiplier TV-L1
    gamma: float
        lagrange multiplier TV-L1
    axes: tuple
        along which axes to compute for :math:`N` total variation
    max_iter: int
        number of iterations
    noise_norm: string
        use **l1** or **l2** norm for regularizing speckle

    """

    if noise_norm not in ["l1", "l2"]:
        raise ValueError("norm must be l1 or l2")
    noise_update = getattr(bregman, noise_norm + "_tv")

    # initialization
    u = g.copy()
    L1 = L2 = t = n = s = np.zeros_like(u)

    # stopping criteria
    theta = 1e-6

    rho = 1.05

    logger.debug("alpha: {:6.4f}".format(alpha))
    logger.debug("rho:   {:6.4f}".format(rho))

    for iter_cnt in range(max_iter):
        # update s
        s = util.soft_thresh(g + L1 / alpha - u - n, 1 / alpha)

        # update u
        u = wnnm((g + t + (L1 + L2) / alpha - n - s) / 2, w / alpha)

        # update t
        # chambolle seems to be more numerical stable than bregman
        t = denoise_tv_chambolle(u - L2 / alpha, lbda / alpha, n_iter_max=50)

        # upate n
        params = {
            "mu": alpha / nu,
            "lbda": mu / nu,
            "alpha": gamma * alpha,
            "beta": beta * alpha,
            "max_iter": 10,
            "axes": axes,
        }
        n = noise_update(g + L1 / alpha - u - s, **params)

        # update L
        L1 = L1 + alpha * (g - (u + n + s))
        L2 = L2 + alpha * (t - u)

        # Updating alpha seems to be an extension of the original ADMM [2]
        alpha *= rho

        # stopping criterion
        rel_err = la.norm(g - (u + n + s), "fro") / la.norm(g, "fro")
        logger.debug("{:d} alpha: {:9.4f}".format(iter_cnt, alpha))
        logger.debug("{:d} relative error: {:9.7f}".format(iter_cnt, rel_err))
        if rel_err < theta:
            logger.debug(
                "relative error {:.2e} smaller than {:.2e}. Stopping optimization after {:d} iterations.".format(
                    rel_err, theta, iter_cnt
                )
            )
            break

    logger.debug("relative error: {:9.7f}".format(rel_err))
    logger.debug(
        "low_rank {:6.3f}, sparse {:6.3f}, speckle {:6.3f}".format(
            *[np.linalg.norm(x) for x in [u, s, n]]
        )
    )

    return u, s, n
