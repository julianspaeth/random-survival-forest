from lifelines.statistics import logrank_test


def find_split(node):
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
        score, split_val, lhs_idxs, rhs_idxs = find_best_split_for_variable(node, i)
        if score > score_opt:
            score_opt = score
            split_val_opt = split_val
            lhs_idxs_opt = lhs_idxs
            rhs_idxs_opt = rhs_idxs
            split_var_opt = i

    return score_opt, split_val_opt, split_var_opt, lhs_idxs_opt, rhs_idxs_opt


def find_best_split_for_variable(node, var_idx):
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
    Compute logrank_test of liflines package.
    :param x: Input samples
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    x = x.iloc[:, feature]
    score_opt = 0
    split_val_opt = None
    lhs_idxs = None
    rhs_idxs = None

    for split_val in x.sort_values(ascending=True, kind="quicksort").unique():
        feature1 = x[x <= split_val].index
        feature2 = x[x > split_val].index
        if feature1.shape[0] < min_leaf or feature2.shape[0] < min_leaf:
            continue
        y1 = y.iloc[feature1, :]
        y2 = y.iloc[feature2, :]
        t1 = y1.iloc[:, 0]
        e1 = y1.iloc[:, 1]
        t2 = y2.iloc[:, 0]
        e2 = y2.iloc[:, 1]
        results = logrank_test(durations_A=t1, durations_B=t2,
                               event_observed_A=e1, event_observed_B=e2)
        score = results.test_statistic

        if score > score_opt:
            score_opt = round(score, 3)
            split_val_opt = round(split_val, 3)
            lhs_idxs = list(feature1)
            rhs_idxs = list(feature2)

    return score_opt, split_val_opt, lhs_idxs, rhs_idxs
