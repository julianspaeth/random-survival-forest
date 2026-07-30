import multiprocessing

import numpy as np
import pandas as pd
import pytest

from random_survival_forest.models import RandomSurvivalForest


def _collect_nodes(node):
    nodes = [node]
    if not node.terminal:
        nodes.extend(_collect_nodes(node.lhs))
        nodes.extend(_collect_nodes(node.rhs))
    return nodes


class _StubTree:
    def __init__(self, chf_value):
        self.chf_value = chf_value

    def predict(self, sample):
        return pd.Series([self.chf_value])


def test_fit_predict_end_to_end(synthetic_x, synthetic_y):
    rsf = RandomSurvivalForest(n_estimators=5, random_state=42)
    rsf.fit(synthetic_x, synthetic_y)

    assert rsf.trees, "expected at least one usable tree after fitting"

    predictions = rsf.predict(synthetic_x)

    assert len(predictions) == len(synthetic_x)
    for chf in predictions:
        assert isinstance(chf, pd.Series)
        assert not chf.empty
        assert (chf.values >= 0).all()


def test_fit_predict_on_rossi_subset(rossi_subset):
    x, y = rossi_subset
    rsf = RandomSurvivalForest(n_estimators=5, random_state=10)
    rsf.fit(x, y)

    assert rsf.trees, "expected at least one usable tree after fitting"

    predictions = rsf.predict(x)
    assert len(predictions) == len(x)


def test_reproducible_with_same_random_state(synthetic_x, synthetic_y):
    rsf_a = RandomSurvivalForest(n_estimators=5, random_state=7).fit(synthetic_x, synthetic_y)
    rsf_b = RandomSurvivalForest(n_estimators=5, random_state=7).fit(synthetic_x, synthetic_y)

    preds_a = rsf_a.predict(synthetic_x)
    preds_b = rsf_b.predict(synthetic_x)

    assert len(preds_a) == len(preds_b)
    for chf_a, chf_b in zip(preds_a, preds_b):
        pd.testing.assert_series_equal(chf_a, chf_b)


def test_n_jobs_normalization_after_fit(synthetic_x, synthetic_y):
    cpu_count = multiprocessing.cpu_count()

    rsf_default = RandomSurvivalForest(n_estimators=2, n_jobs=None, random_state=0)
    rsf_default.fit(synthetic_x, synthetic_y)
    assert rsf_default.n_jobs == 1

    rsf_all = RandomSurvivalForest(n_estimators=2, n_jobs=-1, random_state=0)
    rsf_all.fit(synthetic_x, synthetic_y)
    assert rsf_all.n_jobs == cpu_count

    rsf_clamped = RandomSurvivalForest(n_estimators=2, n_jobs=cpu_count + 10, random_state=0)
    rsf_clamped.fit(synthetic_x, synthetic_y)
    assert rsf_clamped.n_jobs == cpu_count


def test_fit_raises_value_error_for_unsortable_timeline(synthetic_x):
    # Regression test for commit d69b863: an inferred timeline that can't be sorted must raise
    # ValueError (not TypeError) with guidance to pass a custom timeline.
    class Unsortable:
        def __lt__(self, other):
            raise TypeError("not sortable")

    y = pd.DataFrame({
        "event": [1] * len(synthetic_x),
        "time": [Unsortable() for _ in range(len(synthetic_x))],
    })
    rsf = RandomSurvivalForest(n_estimators=1, random_state=0)

    with pytest.raises(ValueError, match="Timeline seems to contain float values"):
        rsf.fit(synthetic_x, y)


def test_unique_deaths_propagates_to_nested_nodes(synthetic_x, synthetic_y):
    # Regression test: Node.grow_tree used to drop unique_deaths when constructing child Nodes,
    # silently resetting nested nodes to the default (3) regardless of the configured value.
    configured_unique_deaths = 7
    rsf = RandomSurvivalForest(n_estimators=3, unique_deaths=configured_unique_deaths, min_leaf=2, random_state=1)
    rsf.fit(synthetic_x, synthetic_y)

    assert rsf.trees, "expected at least one usable tree after fitting"

    all_nodes = []
    for tree in rsf.trees:
        all_nodes.extend(_collect_nodes(tree.lhs))
        all_nodes.extend(_collect_nodes(tree.rhs))

    # Make sure the tree actually grew deep enough to exercise Node's recursive child construction
    # -- the bug only manifests below the tree's direct children.
    assert len(all_nodes) > 2 * len(rsf.trees)
    assert all(node.unique_deaths == configured_unique_deaths for node in all_nodes)


def test_compute_oob_ensembles_keeps_correct_sample_indices():
    # Regression test for the OOB index-misalignment fix: a sample that is in-bag for every tree
    # must be excluded together with its index, not just dropped from the values list.
    rsf = RandomSurvivalForest(n_estimators=2, random_state=0)
    rsf.trees = [_StubTree(1.0), _StubTree(2.0)]
    rsf.bootstraps = [np.array([0, 1]), np.array([0, 2])]  # sample 0 is in-bag for both trees
    xs = pd.DataFrame({"feature_1": [0, 1, 2]})

    oob_ensembles, kept_idxs = rsf._compute_oob_ensembles(xs)

    assert kept_idxs == [1, 2]
    assert [round(chf.iloc[0], 3) for chf in oob_ensembles] == [2.0, 1.0]


def test_compute_oob_score_pairs_predictions_with_matching_samples():
    # Regression test: compute_oob_score must match each OOB prediction with its own sample's
    # actual time/event, not the time/event at the prediction's position in the filtered list.
    rsf = RandomSurvivalForest(n_estimators=2, random_state=0)
    rsf.trees = [_StubTree(1.0), _StubTree(2.0)]
    rsf.bootstraps = [np.array([0, 1]), np.array([0, 2])]
    x = pd.DataFrame({"feature_1": [0, 1, 2]})
    y = pd.DataFrame({"event": [1, 1, 1], "time": [10, 5, 20]})

    # Only samples 1 and 2 get OOB predictions: sample 1 -> chf 2.0 (time 5), sample 2 -> chf 1.0
    # (time 20). Higher predicted hazard for the shorter survival time is concordant -> c-index 1.0.
    c = rsf.compute_oob_score(x, y)

    assert c == pytest.approx(1.0)


def test_fit_with_oob_score_returns_value_in_unit_interval(synthetic_x, synthetic_y):
    rsf = RandomSurvivalForest(n_estimators=10, oob_score=True, random_state=3)
    rsf.fit(synthetic_x, synthetic_y)

    assert isinstance(rsf.oob_score, float)
    assert 0.0 <= rsf.oob_score <= 1.0
