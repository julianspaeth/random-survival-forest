import numpy as np


def select_new_feature_indices(random_state, x, n_features):

    if random_state is None:
        lf_idxs = np.random.permutation(x.shape[1])[:n_features]
        rf_idxs = np.random.permutation(x.shape[1])[:n_features]
    else:
        lf_idxs = np.random.RandomState(seed=random_state).permutation(x.shape[1])[:n_features]
        rf_idxs = np.random.RandomState(seed=random_state).permutation(x.shape[1])[:n_features]

    return lf_idxs, rf_idxs
