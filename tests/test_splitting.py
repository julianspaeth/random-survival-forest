from types import SimpleNamespace

import pytest
from lifelines.statistics import logrank_test

from random_survival_forest.splitting import _find_split, logrank_statistics


def test_logrank_statistics_finds_split_with_custom_column_names(synthetic_x, synthetic_y):
    # Regression test: logrank_statistics used to hardcode "arrest"/"week" as column names and
    # would silently find no split for any differently-named y (see splitting.py fix).
    feature_idx = synthetic_x.columns.get_loc("feature_3")
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(
        x=synthetic_x, y=synthetic_y, feature=feature_idx, min_leaf=3
    )

    assert split_val is not None
    assert lhs_idxs is not None
    assert rhs_idxs is not None
    assert len(lhs_idxs) >= 3
    assert len(rhs_idxs) >= 3


def test_find_split_selects_a_valid_splitting_variable(synthetic_x, synthetic_y):
    node = SimpleNamespace(x=synthetic_x, y=synthetic_y, f_idxs=list(range(synthetic_x.shape[1])), min_leaf=3)

    score, split_val, split_var, lhs_idxs, rhs_idxs = _find_split(node)

    assert split_var is not None
    assert 0 <= split_var < synthetic_x.shape[1]
    assert score > 0
    assert lhs_idxs is not None
    assert rhs_idxs is not None


def test_logrank_score_matches_lifelines_logrank_test(synthetic_x, synthetic_y):
    # Regression test locking in the vectorized rewrite's numerical equivalence with the
    # well-established lifelines implementation of the log-rank chi-squared statistic.
    feature_idx = synthetic_x.columns.get_loc("feature_3")
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(
        x=synthetic_x, y=synthetic_y, feature=feature_idx, min_leaf=3
    )

    event = synthetic_y.iloc[:, 0]
    time = synthetic_y.iloc[:, 1]
    result = logrank_test(
        time.iloc[lhs_idxs], time.iloc[rhs_idxs],
        event_observed_A=event.iloc[lhs_idxs], event_observed_B=event.iloc[rhs_idxs],
    )

    assert score == pytest.approx(result.test_statistic, abs=1e-3)


def test_min_leaf_prevents_any_split(synthetic_x, synthetic_y):
    # min_leaf larger than the whole dataset means no split can ever satisfy both branches.
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(
        x=synthetic_x, y=synthetic_y, feature=0, min_leaf=synthetic_x.shape[0]
    )

    assert split_val is None
    assert lhs_idxs is None
    assert rhs_idxs is None
