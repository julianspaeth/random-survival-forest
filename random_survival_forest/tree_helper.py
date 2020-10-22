import numpy as np


def select_new_feature_indices(x, n_features):
    lf_idxs = np.random.permutation(x.shape[1])[:n_features]
    rf_idxs = np.random.permutation(x.shape[1])[:n_features]

    return lf_idxs, rf_idxs
