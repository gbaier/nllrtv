import numpy as np

from nllrtv.nltensor import dists as df


def test_calc_dists():
    def dist_func(v1, v2):
        """ L2 norm """
        return np.sqrt(np.sum(np.power(v1 - v2, 2)))

    patch_shape = (3, 3)
    n_patches = 10
    arr_in = np.zeros(patch_shape)

    # true distances
    arrs = np.random.normal(size=(n_patches, *patch_shape))
    true_dists = np.sqrt(np.sum(np.power(arrs, 2), axis=(1, 2)))

    np.testing.assert_array_equal(
        list(df.calc_dists(dist_func, arr_in, (x for x in arrs))), true_dists)
