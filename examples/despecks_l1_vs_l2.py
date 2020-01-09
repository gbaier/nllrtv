""" Script comparing different RPCA algorithms """

import matplotlib.pyplot as plt
import numpy as np

# ugly hack for adding parent directory
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[1]))

import data
import nllrtv

##########################################
#                                        #
# Generate stack with correlated speckle #
#                                        #
##########################################

sub = np.s_[140:-100, 140:-100]

fuji_amp = data.fuji[sub]

stack_size = 48
amp_profile = 1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, stack_size))

stack = np.abs(nllrtv.random.stack(fuji_amp, amp_profile, 0.3))

##############
#            #
# Parameters #
#            #
##############

max_iter = 50

params_common = {
    "alpha": 0.25,
    "win_shape": (stack.shape[0], 21, 21),
    "tv": 0.5,
    "C": 5.0,
    "max_iter": max_iter,
}

params_diff = [
        {"mu": 0.04, "noise_norm": "l2"},
        {"mu": 0.8, "noise_norm": "l1"},
]


############
#          #
# Plotting #
#          #
############

plot_idx = 3


fig = plt.figure(figsize=(6, 10))
fig.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.95, hspace=0.15, wspace=0.20)
plotopts = {"cmap": "gray", "vmin": 20, "vmax": 55}

ax = fig.add_subplot(4, 2, 1)
ax.imshow(20 * np.log10(fuji_amp * amp_profile[plot_idx]), **plotopts)
ax.set_title("original")
ax.axis('off')

ax_noisy = fig.add_subplot(4, 2, 2, sharex=ax, sharey=ax)
ax_noisy.imshow(20 * np.log10(stack[plot_idx]), **plotopts)
ax_noisy.set_title("noisy")
ax_noisy.axis('off')

for idx, params in enumerate(params_diff, 3):
    #title = "DespecKS+NLLRTV+" + params["noise_norm"]
    title = params["noise_norm"].upper()
    print("running {}".format(title))

    stack_out, outlier, speckle = nllrtv.despecks.despecks_lrtv(
        stack, **{**params, **params_common}
    )

    ax_filt = fig.add_subplot(4, 2, idx, sharex=ax, sharey=ax)
    ax_filt.imshow(20 * np.log10(stack_out[plot_idx]), **plotopts)
    ax_filt.set_title(title)
    ax_filt.axis('off')

    ax_outlier = fig.add_subplot(4, 2, idx+2, sharex=ax, sharey=ax)
    ax_outlier.imshow(outlier[plot_idx], vmin=-100, vmax=100, cmap='bwr')
    ax_outlier.set_title(title + ' outliers')
    ax_outlier.axis('off')

    ax_speckle = fig.add_subplot(4, 2, idx+4, sharex=ax, sharey=ax)
    ax_speckle.imshow(speckle[plot_idx], vmin=-100, vmax=100, cmap='bwr')
    ax_speckle.set_title(title + ' speckle')
    ax_speckle.axis('off')

plt.show()
