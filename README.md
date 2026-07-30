# Random Survival Forest

[![DOI](https://zenodo.org/badge/201053930.svg)](https://zenodo.org/badge/latestdoi/201053930)

The Random Survival Forest package provides a python implementation of the survival prediction method originally published by Ishwaran et al. (2008).

Reference:
Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
Random survival forests.
The annals of applied statistics, 2(3), 841-860.

## Requirements

Python >= 3.13.

## Installation

```sh
pip install random-survival-forest
```

## Contribute

- Source Code: https://github.com/julianspaeth/random-survival-forest

## Development

Dependencies and environment are managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --group dev     # install the package and dev dependencies
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run pip-audit        # check dependencies for known vulnerabilities
```

## Performance

The log-rank split search is vectorized with numpy, so training scales well with sample size and number of trees. For very large datasets, `n_jobs` lets you parallelize tree construction across cores.

## Getting Started

```python
import time

from lifelines import datasets
from sklearn.model_selection import train_test_split

from random_survival_forest.models import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index

# Rossi recidivism dataset: "arrest" (1 = rearrested, 0 = censored), "week" = time in weeks.
rossi = datasets.load_rossi()
# y needs the event column first, then time: event before duration.
y = rossi.loc[:, ["arrest", "week"]]
X = rossi.drop(["arrest", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

print("Start training...")
start_time = time.time()
# n_jobs=-1 uses all CPU cores, random_state makes the run reproducible.
rsf = RandomSurvivalForest(n_estimators=50, n_jobs=-1, random_state=42)
rsf = rsf.fit(X, y)
print(f'--- {round(time.time() - start_time, 3)} seconds ---')

# Predicted cumulative hazard function per test sample.
y_pred = rsf.predict(X_test)

# C-index: fraction of comparable pairs ranked correctly (0.5 random, 1.0 perfect).
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print(f'C-index {round(c_val, 3)}')
```

## Feedback

If you are having issues or feedback, please let me know. I am happy to fix some bug or implement feature requests.

<spaethju@posteo.de>

## License

MIT
