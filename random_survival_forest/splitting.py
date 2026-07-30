import numpy as np


def _find_split(node):
    """
    Find the best split for a Node.
    :param node: Node to find best split for.
    :return: score of best split, value of best split, variable to split, left indices, right indices.
    """
    score_opt = 0
    split_val_opt = None
    lhs_idxs_opt = None
    rhs_idxs_opt = None
    split_var_opt = None
    for i in node.f_idxs:
        score, split_val, lhs_idxs, rhs_idxs = _find_best_split_for_variable(node, i)

        if score > score_opt:
            score_opt = score
            split_val_opt = split_val
            lhs_idxs_opt = lhs_idxs
            rhs_idxs_opt = rhs_idxs
            split_var_opt = i

    return score_opt, split_val_opt, split_var_opt, lhs_idxs_opt, rhs_idxs_opt


def _find_best_split_for_variable(node, var_idx):
    """
    Find best split for a variable of a Node. Best split for a variable is the split with the highest log rank
    statistics. The logrank_test function of the lifelines package is used here.
    :param node: Node
    :param var_idx: Index of variable
    :return: score, split value, left indices, right indices.
    """
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(x=node.x, y=node.y,
                                                              feature=var_idx,
                                                              min_leaf=node.min_leaf)
    return score, split_val, lhs_idxs, rhs_idxs


def logrank_statistics(x, y, feature, min_leaf):
    """
    Compute the log-rank split statistic for every candidate split value of a feature in one vectorized pass,
    and return the best one. Uses the martingale-residual identity for the log-rank numerator (each subject's
    residual event_j - H(time_j), H being the pooled Nelson-Aalen cumulative hazard, is split-independent and
    computed once) plus the standard hypergeometric log-rank variance, evaluated for all candidates via a single
    numpy broadcast. Numerically equivalent to lifelines.statistics.logrank_test for a given split.
    :param x: Input samples
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """
    x_feature = x.reset_index(drop=True).iloc[:, feature].to_numpy()
    event = y.iloc[:, 0].to_numpy()
    time = y.iloc[:, 1].to_numpy()
    n = len(x_feature)

    order_t = np.argsort(time, kind='mergesort')
    t_sorted = time[order_t]
    e_sorted = event[order_t]

    unique_times, first_idx = np.unique(t_sorted, return_index=True)
    at_risk = n - first_idx
    deaths = np.add.reduceat(e_sorted, first_idx)

    with np.errstate(divide='ignore', invalid='ignore'):
        hazard_increment = np.where(at_risk > 0, deaths / at_risk, 0.0)
    cum_hazard_at_unique = np.cumsum(hazard_increment)

    subject_time_rank = np.searchsorted(unique_times, time, side='right') - 1
    residual = event - cum_hazard_at_unique[subject_time_rank]

    order_x = np.argsort(x_feature, kind='mergesort')
    x_sorted = x_feature[order_x]
    time_by_x = time[order_x]
    numerator_cum = np.cumsum(residual[order_x])

    candidate_k = np.nonzero(np.diff(x_sorted) != 0)[0] + 1
    valid = (candidate_k >= min_leaf) & (n - candidate_k >= min_leaf)
    candidate_k = candidate_k[valid]

    if len(candidate_k) == 0:
        return 0, None, None, None

    numerator = numerator_cum[candidate_k - 1]

    at_risk_group1 = np.cumsum(time_by_x[:, None] >= unique_times[None, :], axis=0)[candidate_k - 1, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        proportion = at_risk_group1 / at_risk[None, :]
        variance_terms = (
            proportion * (1 - proportion)
            * np.where(at_risk[None, :] > 1, (at_risk[None, :] - deaths[None, :]) / (at_risk[None, :] - 1), 0.0)
            * deaths[None, :]
        )
    variance = np.nan_to_num(variance_terms, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(variance > 0, numerator ** 2 / variance, 0.0)

    best_i = np.argmax(score)
    if score[best_i] <= 0:
        return 0, None, None, None

    split_val_exact = x_sorted[candidate_k[best_i] - 1]
    lhs_idxs = np.where(x_feature <= split_val_exact)[0]
    rhs_idxs = np.where(x_feature > split_val_exact)[0]

    return round(float(score[best_i]), 3), round(float(split_val_exact), 3), lhs_idxs, rhs_idxs
