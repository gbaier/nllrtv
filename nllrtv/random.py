""" Functions for creating SAR amplitude stacks with temporally correlated, fully-developed speckle """
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.linalg import toeplitz


def multivariate_complex_normal(cov, size=1):
    """ multivariate complex normal distribution

    Parameters
    ----------

    cov: array_like
        covariance matrix. Must be positive-definite
    size: int
        number of samples
    """

    n_dim = cov.shape[0]

    # complex vector space isomorphism
    cov_real = np.kron(np.eye(2), cov.real) + np.kron(
        np.array([[0, -1], [1, 0]]), cov.imag
    )

    # normalization
    cov_real /= 2

    xy = __multivariate_normal(cov_real, size)

    if xy.ndim == 1:
        return xy[:n_dim] + 1j * xy[-n_dim:]
    return xy[:, :n_dim] + 1j * xy[:, -n_dim:]


def __multivariate_normal(cov, size):
    # for some covariance matrices the cholesky decomposition may fail
    # in this case fall back to scipy's, i.e. numpy's routine, which relies
    # on singular value decomposition.
    # We try to use the Cholesky decomposition for performance reasons.
    # See https://github.com/numpy/numpy/pull/3938 for a dicussion.
    try:
        # Choleksy decomposition: COV = LL^H
        L = np.linalg.cholesky(cov)

        xy = norm.rvs(size=cov.shape[0] * size).reshape((-1, size))
        # so L @ xy has covariance LL^T = cov
        xy = np.dot(L, xy)

        # first dimension: number of samples:
        xy = xy.T

        if size == 1:
            xy = xy.flatten()

    except np.linalg.linalg.LinAlgError:
        xy = multivariate_normal.rvs(cov=cov, size=size)

    return xy


def exp_decay_coh_mat(M, lbda):
    """ generates a coherence matrix with exponential decay

    Parameters
    ----------

    M: int
        dimension of matrix
    lbda: float
        exponential decay
    """

    coh_vec = np.exp(-np.arange(0, M) * lbda)

    return toeplitz(coh_vec)


def stack(amp_base, amp_temp_profile, lbda):
    """ generates a random amplitude stack

    Parameters
    ----------

    amp_base: 2-dim numpy array
        spatial amplitude image
    amp_temp_profile: 1-dim numpy array
        temporal amplitude profile
    lbda: float
        exponential decay
    """

    # build covariance matrix
    coh_vals = exp_decay_coh_mat(amp_temp_profile.size, lbda)
    phi_vals = np.exp(1j * np.random.uniform(-np.pi, np.pi, amp_temp_profile.size))
    phi_vals = np.outer(phi_vals, phi_vals.conj())

    cov_mat = phi_vals * coh_vals * np.outer(amp_temp_profile, amp_temp_profile)

    # get correlated speckle
    stack_speckle = multivariate_complex_normal(cov_mat, size=amp_base.size).T.reshape(
        (-1, *amp_base.shape)
    )

    return amp_base * stack_speckle
