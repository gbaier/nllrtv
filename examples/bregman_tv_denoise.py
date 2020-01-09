""" Script comparing different TV restoration algorithms """

from skimage import data
from skimage.color import rgb2grey
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
from skimage.util import random_noise

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import nllrtv.bregman as bregman

original = rgb2grey(data.coffee())[::2, ::2]

noisy = random_noise(original, mode='speckle', var=0.05)

NORM_CONST = 255
plotopts = {'cmap': 'gray', 'vmin': 0, 'vmax': NORM_CONST}
noisy *= NORM_CONST
original *= NORM_CONST

params = {'mu': 0.04, 'beta': 0.08, 'max_iter': 100}
filters = {
        'ski TV bregman': (denoise_tv_bregman, {'weight': params['mu'], 'max_iter': params['max_iter']}),
        'ski TV chambolle': (denoise_tv_chambolle, {'weight': 1/params['mu'], 'n_iter_max': 2*params['max_iter']}),
        'TV': (bregman.tv, params),
        'TV+L2': (bregman.l2_tv, {**params, 'lbda': 0.002, 'alpha': 0.004}),
        'TV+L1': (bregman.l1_tv, {**params, 'lbda': 0.2, 'alpha': 0.4}),
        'TV+L1+dx': (bregman.l1_tv, {**params, 'lbda': 0.02, 'alpha': 0.04, 'axes': (0, )}),
        'TV+L1+dy': (bregman.l1_tv, {**params, 'lbda': 0.02, 'alpha': 0.04, 'axes': (1, )}),
}

# plotting
fig = plt.figure(figsize=(12, 6))
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(3, 4),
    axes_pad=0.3,
    share_all=True,
)

grid[0].imshow(original, **plotopts)
grid[0].set_title('original')

grid[1].imshow(noisy, **plotopts)
grid[1].set_title('noisy')

results = {}

for idx, (title, (method, m_params)) in enumerate(filters.items(), 2):
    print('filtering with {}'.format(title))
    results[title] = method(noisy, **m_params)
    grid[idx].imshow(results[title], **plotopts)
    grid[idx].set_title(title)

for idx, (p1, p2) in enumerate([('TV', 'TV+L1'), ('TV', 'TV+L2'), ('TV+L1', 'TV+L2')], idx+1):
    grid[idx].imshow(results[p1] - results[p2], cmap='bwr', vmin=-20, vmax=20)
    grid[idx].set_title("{} - {}".format(p1, p2))

plt.show()
