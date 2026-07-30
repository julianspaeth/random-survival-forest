import pandas as pd
import pytest

from random_survival_forest.scoring import concordance_index


def test_perfect_concordance():
    y_time = pd.Series([1, 2, 3, 4])
    y_event = pd.Series([1, 1, 1, 1])
    y_pred = [pd.Series([4.0]), pd.Series([3.0]), pd.Series([2.0]), pd.Series([1.0])]

    assert concordance_index(y_time=y_time, y_pred=y_pred, y_event=y_event) == pytest.approx(1.0)


def test_perfect_discordance():
    y_time = pd.Series([1, 2, 3, 4])
    y_event = pd.Series([1, 1, 1, 1])
    y_pred = [pd.Series([1.0]), pd.Series([2.0]), pd.Series([3.0]), pd.Series([4.0])]

    assert concordance_index(y_time=y_time, y_pred=y_pred, y_event=y_event) == pytest.approx(0.0)


def test_tied_time_and_event_with_equal_predictions_counts_as_concordant():
    y_time = pd.Series([5, 5])
    y_event = pd.Series([1, 1])
    y_pred = [pd.Series([2.0]), pd.Series([2.0])]

    assert concordance_index(y_time=y_time, y_pred=y_pred, y_event=y_event) == pytest.approx(1.0)


def test_censored_pair_with_shorter_censored_time_is_not_permissible():
    # t1 < t2 and e1 == 0 (censored) -> the pair carries no information and must not count.
    y_time = pd.Series([1, 5])
    y_event = pd.Series([0, 1])
    y_pred = [pd.Series([1.0]), pd.Series([2.0])]

    # Only one pair exists and it is skipped -> permissible stays 0 -> ZeroDivisionError.
    with pytest.raises(ZeroDivisionError):
        concordance_index(y_time=y_time, y_pred=y_pred, y_event=y_event)


def test_all_censored_pairs_raise_zero_division_error():
    # Regression test documenting current behavior: no comparable pairs -> permissible stays 0.
    y_time = pd.Series([1, 2])
    y_event = pd.Series([0, 0])
    y_pred = [pd.Series([1.0]), pd.Series([2.0])]

    with pytest.raises(ZeroDivisionError):
        concordance_index(y_time=y_time, y_pred=y_pred, y_event=y_event)
