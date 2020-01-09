import numpy as np

import scipy.sparse

import skimage.data
import skimage.color
import skimage.util
from skimage.restoration import denoise_tv_chambolle

from nllrtv import bregman
from nllrtv import util


def test_l1_tv_onlytv():
    original = skimage.color.rgb2grey(skimage.data.coffee())[::3, ::3]

    sigma2 = 0.02
    noisy = skimage.util.random_noise(original, var=sigma2)

    # for additive noise
    params = {
        "mu": 10,
        "lbda": 0.00001,  # ignore L1 minimization
        "beta": 1.0,
        "alpha": 1.0,
        "max_iter": 200,
    }

    denoised = bregman.l1_tv(noisy, **params)
    denoised_ski_cham = denoise_tv_chambolle(
        noisy, weight=1 / params["mu"], n_iter_max=params["max_iter"]
    )

    atol = 0.05  # absolute tolerance
    sub = np.s_[2:-2, 2:-2]  # ignore borders
    np.testing.assert_allclose(denoised[sub], denoised_ski_cham[sub], atol=atol)
    assert noisy.std() > denoised.std()


def test_l1_tv_onlyl1():
    shape = (20, 60)
    density = 0.2

    # generate sparse matrix with 0 and 1 elements
    x_true = scipy.sparse.random(shape[0], shape[1], density, data_rvs=lambda n: np.ones(n)).A

    sigma_noise = 0.1
    x_noisy = x_true + sigma_noise * np.random.normal(size=x_true.shape)

    params = {
        "mu": 5000,  # ignore TV
        "lbda": 1000,  # ignore TV
        "beta": 200.0,
        "alpha": 200.0,
        "max_iter": 500,
    }

    x_est_soft = util.soft_thresh(x_noisy, params["lbda"] / params["mu"])
    x_est_breg = bregman.l1_tv(x_noisy, **params)

    np.testing.assert_equal(np.round(x_est_soft).astype(np.int), x_true)
    np.testing.assert_equal(np.round(x_est_breg).astype(np.int), x_true)

    sub = np.s_[2:-2, 2:-2]
    np.testing.assert_almost_equal(x_est_breg[sub], x_est_soft[sub], decimal=3)
