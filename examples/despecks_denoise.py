""" Script comparing different RPCA algorithms """

import matplotlib.pyplot as plt
import numpy as np

import nllrtv
import nllrtv.data

##########################################
#                                        #
# Generate stack with correlated speckle #
#                                        #
##########################################

# Only take a small subset. Otherwise this example takes long to run.
sub = np.s_[140:-100, 140:-100]
fuji_amp = nllrtv.data.fuji[sub]

stack_size = 24
amp_profile = 1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, stack_size))

stack = np.abs(nllrtv.random.stack(fuji_amp, amp_profile, 0.3))

##############
#            #
# Parameters #
#            #
##############

alpha = 0.25
win_shape = (stack.shape[0], 21, 21)

# Minimum number of statistically homogeneous pixels needed for DespecKS
# to perform filtering. This preserves point scatters.
min_n_shp = 10

# Maximum number of iteration. Should be something like 50-100.
# We only pick a small number for quick results as an example.
max_iter = 3

filters = {
    nllrtv.despecks.despecks: {'min_n_shp': min_n_shp},
    nllrtv.despecks.despecks_lrtv: {'C': 5, 'tv': 0.5, 'max_iter': max_iter},
}

############
#          #
# Plotting #
#          #
############

plot_idx = 3


fig = plt.figure(figsize=(6, 6))
plotopts = {"cmap": "gray", "vmin": 20, "vmax": 55}

ax = fig.add_subplot(2, 2, 1)
ax.imshow(20 * np.log10(fuji_amp*amp_profile[plot_idx]), **plotopts)
ax.set_title('original')

ax_noisy = fig.add_subplot(2, 2, 2, sharex=ax, sharey=ax)
ax_noisy.imshow(20 * np.log10(stack[plot_idx]), **plotopts)
ax_noisy.set_title('noisy')

for idx, (method, params) in enumerate(filters.items(), 3):
    title = 'DespecKS'
    if method.__name__.find('_') != -1:
        title += '+' + method.__name__.split('_')[1].upper()

    print("Running {}.".format(title), end=' ')
    print("This might take a while...")

    ax_filt = fig.add_subplot(2, 2, idx, sharex=ax, sharey=ax)
    ret_val = method(stack, alpha, win_shape, **params)

    # deal with different returned arrays
    if ret_val.ndim == 4:
        stack_out = ret_val[0]
    else:
        stack_out = ret_val

    ax_filt.imshow(20 * np.log10(stack_out[plot_idx]), **plotopts)
    ax_filt.set_title(title)

plt.show()
