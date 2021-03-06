# Random Survival Forest

[![Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/random-survival-forest/community)

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
>>> from random_survival_forest import RandomSurvivalForest, concordance_index
>>> from lifelines import datasets
>>> from sklearn.model_selection import train_test_split


>>> rossi = datasets.load_rossi()
# Attention: duration column must be index 0, event column index 1 in y
>>> y = rossi.loc[:, ["week", "arrest"]]
>>> X = rossi.drop(["arrest", "week"], axis=1)
>>> X, X_test, y, y_test = train_test_split(X, y, test_size=0.25)


>>> rsf = RandomSurvivalForest(n_estimators=20, n_jobs=-1)
>>> rsf = rsf.fit(X, y)
>>> y_pred = rsf.predict(X_test)
>>> c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
>>> print("C-index", round(c_val, 3))
```

## Support

If you are having issues or feedback, please let me know.

julian.spaeth@wzw.tum.de

## License
MIT
