""" Script comparing different RPCA algorithms """

from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.restoration import denoise_tv_bregman

import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import nllrtv.wnnm as wnnm


original = img_as_float(rgb2grey(data.coffee()))[::2, ::2]

# add some outliers, i.e., salt and pepper noise
noisy = random_noise(original, "s&p", salt_vs_pepper=1)
# outlier positions
outliers = original != noisy

# weights
U, s, Vh = la.svd(noisy)
C = np.sqrt(noisy.size) * 0.08
w = C / (np.sqrt(s) + 0.001)


params = {
    "w": w,
    "lbda": 0.1,     # TV regularization for additional smoothing
    "mu": 100.0,     # ignore noise, only consider sparse outliers
    "nu": 100.0,     # ignore noise, only consider sparse outliers
    "alpha": 2,
    "beta": 2,
    "gamma": 2,
    "axes": (0, 1),  # regularize both in x- and y-direction
}

filters = {wnnm.rpca: {}, wnnm.rpca_wnnm: {"w": w}, wnnm.rpca_wnnm_tv: params}

# plotting
plotopts = {"cmap": "gray", "vmin": 0, "vmax": 1}
fig = plt.figure(figsize=(12, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.3, share_all=True)

grid[0].imshow(original, **plotopts)
grid[0].set_title("original")

grid[1].imshow(noisy, **plotopts)
grid[1].set_title("noisy")

grid[2].imshow(denoise_tv_bregman(noisy, weight=3.0), **plotopts)
grid[2].set_title("Total variation")

for idx, (method, params) in enumerate(filters.items(), 3):
    name = method.__name__.upper().replace('_', '+')
    print("running {}".format(name))

    low_rank, sparse, *_ = method(noisy, **params)

    grid[idx].imshow(low_rank, **plotopts)
    grid[idx].set_title(name)

    grid[idx + 3].imshow(original - low_rank, cmap="bwr", vmin=-0.2, vmax=0.2)
    grid[idx + 3].set_title('original - {}'.format(name))

plt.show()
