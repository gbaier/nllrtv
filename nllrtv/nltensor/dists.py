""" some general distance functions """

from scipy.stats import ks_2samp


def calc_dists(func, arr_in, arrs):
    """ calculates all distances between arr_in and the arrays in arrs using func

    Parameters
    ----------

    func: function

    arr_in: ndarray
        represents the source patch

    arrs: sequence of ndarrays
        the target patches

    """

    for arr in arrs:
        yield func(arr_in, arr)


def kolmogorov_smirnov(vec1, vec2):
    """ 2sample Kolmogorov-Smirnov test

    used in [1] for identifying statistically homogeneous pixels.

    [1] A. Ferretti, A. Fumagalli, F. Novali, C. Prati, F. Rocca and A. Rucci,
    "A New Algorithm for Processing Interferometric Data-Stacks: SqueeSAR," in
    IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 9, pp.
    3460-3470, Sept. 2011.

    """
    return ks_2samp(vec1, vec2)[0]
