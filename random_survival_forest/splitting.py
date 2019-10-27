from lifelines.statistics import logrank_test


def find_split(node):
    """
    Find the best split for a Node.
    :param node: Node to find best split for.
    :return: score of best split, value of best split, variable to split, left indices, right indices.
    """
    results = [find_best_split_for_variable(node, i) for i in node.f_idxs]
    scores = [item[0] for item in results]
    max_idx = scores.index(max(scores))
    score_opt, split_val_opt, lhs_idxs_opt, rhs_idxs_opt, split_var_opt = results[max_idx]
    return score_opt, split_val_opt, split_var_opt, lhs_idxs_opt, rhs_idxs_opt


def find_best_split_for_variable(node, var_idx):
    """
    Find best split for a variable of a Node. Best split for a variable is the split with the highest log rank
    statistics. The logrank_test function of the lifelines package is used here.
    :param node: Node
    :param var_idx: Index of variable
    :return: score, split value, left indices, right indices, feature index.
    """
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(x=node.x, y=node.y,
                                                              feature=var_idx,
                                                              min_leaf=node.min_leaf)
    return score, split_val, lhs_idxs, rhs_idxs, var_idx


def logrank_statistics(x, y, feature, min_leaf):
    """
    Compute logrank_test of liflines package.
    :param x: Input samples
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """
    x_feature = x.reset_index(drop=True).iloc[:, feature]
    sorted_values = x_feature.sort_values(ascending=True, kind="quicksort").unique()
    results = [compute_score(x_feature, y, split_val, min_leaf) for split_val in sorted_values]
    scores = [item[0] for item in results]
    max_idx = scores.index(max(scores))
    score_opt, split_val_opt, lhs_idxs, rhs_idxs = results[max_idx]

    return score_opt, split_val_opt, lhs_idxs, rhs_idxs


def compute_score(x_feature, y, split_val, min_leaf):
    feature1 = list(x_feature[x_feature <= split_val].index)
    feature2 = list(x_feature[x_feature > split_val].index)
    if len(feature1) < min_leaf or len(feature2) < min_leaf:
        score = 0
    else:
        durations_a = y.iloc[feature1, 0]
        event_observed_a = y.iloc[feature1, 1]
        durations_b = y.iloc[feature2, 0]
        event_observed_b = y.iloc[feature2, 1]
        results = logrank_test(durations_A=durations_a, durations_B=durations_b,
                               event_observed_A=event_observed_a, event_observed_B=event_observed_b)
        score = results.test_statistic
    return [score, split_val, feature1, feature2]
