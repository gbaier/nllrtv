""" SAR stack despeckling algorithms.

These module implements the original DespecKS algorithms [1] and the proposed
extension, which replaces DespecKS's mean with low-rank and TV regularization.

[1] A. Ferretti, A. Fumagalli, F. Novali, C. Prati, F. Rocca and A. Rucci,
"A New Algorithm for Processing Interferometric Data-Stacks: SqueeSAR," in
IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 9, pp.
3460-3470, Sept. 2011.

"""

import logging
import numbers

import numpy as np
import numpy.linalg as la

from . import nltensor as nlt
from . import wnnm
from . import util
from .parallel import daskify

logger = logging.getLogger(__name__)


def ks_sim(func, stack, win_shape, alpha=0.1, aggr=None, depth=0):
    """ uses Kilmogorov similarity test to identify similar vectors

    The collection is then denoised using func

    Parameters
    ----------

    func: denoising function
        takes the aggregator, a target slice and a list of good patches as parameters
    stack: numpy array
         SAR image stack
    alpha: float
        confidence level for KS test
    win_shape: tuple
        dimensions of the search window
    aggr: array_like
        aggregator used by func
    depth: integer or tuple
        Skip outer parts of the spatial dimensions.
        Useful for parallel processing using DASK.

    """

    # filtering parameters
    pat_shape = (stack.shape[0], 1, 1)

    if aggr is None:
        aggr = np.zeros_like(stack)

    # setting up overlap for possible processing with dask
    spatial_ndim = 2
    if isinstance(depth, numbers.Number):
        depth = (depth,) * spatial_ndim

    starts = (0, *depth)
    stops = (1, *(s - d for s, d in zip(stack.shape[1:], depth)))

    # iterate over the spatial dimension
    for coords in util.mdim_range(starts, stops):
        target_sub = nlt.tensor.hyperrect_slice(coords, pat_shape)
        target_pat = stack[target_sub]

        search_window = list(
            nlt.search.get_rect_window_patches(stack, coords, win_shape, pat_shape)
        )

        def dist_func(v1, v2):
            return nlt.tensor.apply_along_axis(
                nlt.dists.kolmogorov_smirnov, 0, (v1, v2)
            )

        dists = list(
            nlt.dists.calc_dists(
                dist_func, target_pat, (swe.patch for swe in search_window)
            )
        )

        # good source patches
        gsp = list(sw for sw, d in zip(search_window, dists) if d < alpha)

        # dirty hack to not have an empty list
        if len(gsp) < 5:
            gsp = list(
                sw for sw, d in sorted(zip(search_window, dists), key=lambda x: x[1])
            )

        logger.debug("length of good patches %d", len(gsp))

        # An empty good patch list should never happen, due to the source patch
        # having a distance of zero. This check enables Dask to properly infer
        # the return type for a 1 by 1 chunk,
        if gsp:
            aggr = func(aggr, target_sub, gsp)

    return aggr


@daskify(chunks=(20, 20), new_axis=None)
def despecks(stack, alpha, win_shape, min_n_shp):
    """ implments the despecKS algorithm

    Parameters
    ----------

    stack: array_like
        amplitude SAR stack
    alpha: float
        significance test for the Kolmogorov-Smirnov similarity test
    win_shape: tuple
        shape of the search window
    min_n_shp: int
        minimum number of statistically homogenous pixels to perform denoising.
        This preserves point targets.

    """

    def sw_denoiser(stack_out, target_sub, gsp):
        if len(gsp) > min_n_shp:
            for source_pat, _ in gsp:
                stack_out[target_sub] += source_pat / len(gsp)
        else:
            stack_out[target_sub] += stack[target_sub]
        return stack_out

    return ks_sim(
        sw_denoiser,
        stack,
        win_shape,
        alpha,
        depth=tuple((x // 2 for x in win_shape[1:])),
    )


@daskify(chunks=(20, 20), new_axis=[0], aggr=True)
def despecks_lrtv(
    stack, alpha, win_shape, C=5, tv=0.5, max_iter=50, noise_norm="l1", mu=0.8
):
    """ implments the DespecKS algorithm combined with low-rank and TV regularization

    Parameters
    ----------

    stack: array_like
        amplitude SAR stack
    alpha: float
        significance test for the Kolmogorov-Smirnov similarity test
    win_shape: tuple
        shape of the search window
    C: float
        weighting factor for weighted nuclear norm
    tv: float
        regularization constant for total variation of signal
    max_iter: int
        number of iterations

    """

    def sw_denoiser(aggr, target_sub, gsp):
        low_rank, outlier, speckle, counter = aggr

        # build matrix from collected vectors
        mat = np.stack(tuple(sw.patch.flatten() for sw in gsp))

        # get weights
        _, sing_vals, _ = la.svd(mat)
        weights = C * np.sqrt(mat.size) / (np.sqrt(sing_vals) + 0.0001)

        params = {
            "lbda": tv,  # image TV regularization
            "mu": mu,  # speckle L1 / L2 normalization
            "nu": 0.15,  # speckle TV regularization
            "alpha": np.prod(mat.shape) / (4 * np.sum(np.abs(mat))),
            "beta": 5.0,
            "gamma": 5.0,
            "max_iter": max_iter,
            "axes": (1,),
            "noise_norm": noise_norm,
        }
        low_rank_sub, outlier_sub, speckle_sub = wnnm.rpca_wnnm_tv(
            mat, weights, **params
        )

        vecs_lr = (
            v.reshape((-1, 1, 1)) for v in np.split(low_rank_sub, low_rank_sub.shape[0])
        )
        vecs_out = (
            v.reshape((-1, 1, 1)) for v in np.split(outlier_sub, low_rank_sub.shape[0])
        )
        vecs_speck = (
            v.reshape((-1, 1, 1)) for v in np.split(speckle_sub, low_rank_sub.shape[0])
        )

        for idx, vec_lr, vec_out, vec_speck in zip(
            (sw.idxs for sw in gsp), vecs_lr, vecs_out, vecs_speck
        ):
            low_rank[idx] += vec_lr
            outlier[idx] += vec_out
            speckle[idx] += vec_speck
            counter[idx] += 1

        return np.stack((low_rank, outlier, speckle, counter))

    low_rank, outlier, speckle, counter = ks_sim(
        sw_denoiser,
        stack,
        win_shape,
        alpha,
        aggr=np.zeros((4, *stack.shape), dtype=stack.dtype),
        depth=tuple((x // 2 for x in win_shape[1:])),
    )

    # Catch division warnings.
    # This occurs at the overlapping areas because the filter did not write necessarily to all of them
    # Invalid data will later be checked and ignored in the aggregation step.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack((low_rank / counter, outlier / counter, speckle / counter))
