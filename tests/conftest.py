import numpy as np
import pandas as pd
import pytest
from lifelines import datasets


def _make_survival_dataset(n=60, seed=0):
    """Deterministic synthetic survival data with column names other than 'arrest'/'week',
    to exercise the fix for the previously hardcoded column names in splitting.py."""
    rng = np.random.RandomState(seed)
    feature_1 = rng.normal(size=n)
    feature_2 = rng.normal(size=n)
    feature_3 = rng.choice([0.0, 1.0], size=n)

    hazard = np.exp(0.5 * feature_1 + 1.2 * feature_3)
    true_time = rng.exponential(scale=1.0, size=n) / hazard
    censor_time = rng.exponential(scale=2.0, size=n)
    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)

    x = pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3})
    y = pd.DataFrame({"death_event": event, "survival_weeks": np.round(time, 3)})
    return x, y


@pytest.fixture
def synthetic_data():
    return _make_survival_dataset(n=60, seed=0)


@pytest.fixture
def synthetic_x(synthetic_data):
    return synthetic_data[0]


@pytest.fixture
def synthetic_y(synthetic_data):
    return synthetic_data[1]


@pytest.fixture
def rossi_subset():
    rossi = datasets.load_rossi().iloc[:80].reset_index(drop=True)
    y = rossi.loc[:, ["arrest", "week"]]
    x = rossi.drop(["arrest", "week"], axis=1)
    return x, y
