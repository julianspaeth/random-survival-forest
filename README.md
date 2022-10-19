# Random Survival Forest

[![DOI](https://zenodo.org/badge/201053930.svg)](https://zenodo.org/badge/latestdoi/201053930)

The Random Survival Forest package provides a python implementation of the survival prediction method originally published by Ishwaran et al. (2008).

Reference: 
Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). 
Random survival forests. 
The annals of applied statistics, 2(3), 841-860.

## Installation
```sh
$ pip install random-survival-forest
```

## Contribute

- Source Code: https://github.com/julianspaeth/random-survival-forest

## Getting Started

```python
import time

from lifelines import datasets
from sklearn.model_selection import train_test_split

from random_survival_forest.models import RandomSurvivalForest
from random_survival_forest.scoring import concordance_index

rossi = datasets.load_rossi()
# Attention: duration column must be index 0, event column index 1 in y
y = rossi.loc[:, ["arrest", "week"]]
X = rossi.drop(["arrest", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

print("Start training...")
start_time = time.time()
rsf = RandomSurvivalForest(n_estimators=10, n_jobs=-1, random_state=10)
rsf = rsf.fit(X, y)
print(f'--- {round(time.time() - start_time, 3)} seconds ---')
y_pred = rsf.predict(X_test)
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print(f'C-index {round(c_val, 3)}')
```

## Feedback

If you are having issues or feedback, please let me know. I am happy to fix some bug or implement feature requests.

julian.alexander.spaeth@uni-hamburg..de

This package is open-source. If it helped you or you even use it comercially, I would be happy about a little support:

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/julianspaeth)

## License
MIT


