.. Read the Docs Template documentation master file, created by
   sphinx-quickstart on Tue Aug 26 14:19:49 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contents:

.. toctree::
   :maxdepth: 2
   :glob:

   *


Random Survival Forest
======================

The Random Survival Forest package provides a python implementation of the survival prediction method originally published by Ishwaran et al. (2008).

Reference:
Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
Random survival forests.
The annals of applied statistics, 2(3), 841-860.


Getting Started
---------------

.. code:: python

    >>> from random_survival_forest import RandomSurvivalForest
    >>> timeline = range(0, 10, 1)
    >>> rsf = RandomSurvivalForest(n_estimators=20, timeline=timeline)
    >>> rsf.fit(X, y)
    >>> round(rsf.oob_score, 3)
    0.76
    >>> y_pred = rsf.predict(X_val)
    >>> c_val = concordance_index(y_val["time"], y_pred, y_val["event"])
    >>> round(c_val, 3)
    0.72


Support
-------

If you are having issues or feedback, please let me know.

julian.spaeth@student.uni-tuebinden.de


Contribute
----------

- Source Code: https://github.com/julianspaeth/random-survival-forest


License
-------
The project is licensed under the MIT license.



