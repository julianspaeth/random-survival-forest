import numpy as np
import pandas as pd
from .SurvivalTree import SurvivalTree
from .scoring import concordance_index
from joblib import Parallel, delayed
import multiprocessing


class RandomSurvivalForest:

    def __init__(self, timeline, n_estimators=100, min_leaf=3, unique_deaths=3,
                 n_jobs=None, parallelization_backend="multiprocessing", random_state=None):
        """
        A Random Survival Forest is a prediction model especially designed for survival analysis.
        :param timeline: The timeline used for the prediction. e.g. range(0, 10, 1)
        :param n_estimators: The numbers of trees in the forest.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param random_state: The random state to create reproducible results.
        :param n_jobs: The number of jobs to run in parallel for fit. None means 1.
        """
        self.timeline = timeline
        self.n_estimators = n_estimators
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.n_jobs = n_jobs
        self.parallelization_backend=parallelization_backend
        self.random_state = random_state
        self.bootstrap_idxs = None
        self.bootstraps = []
        self.oob_idxs = None
        self.oob_score = None
        self.trees = []
        self.random_states = []


    def fit(self, x, y):
        """
        Build a forest of trees from the training set (X, y).
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event
        in the second with the shape [n_samples, 2]
        :return: self: object
        """
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs is None:
            self.n_jobs = 1
        self.random_states = np.random.RandomState(seed=self.random_state).randint(0, 2**32-1, self.n_estimators)
        self.bootstrap_idxs = self.draw_bootstrap_samples(x)

        trees = Parallel(n_jobs=self.n_jobs, backend=self.parallelization_backend)(delayed(self.create_tree)(x, y, i)
                                                                                   for i in range(self.n_estimators))

        for i in range(len(trees)):
            if trees[i].prediction_possible:
                self.trees.append(trees[i])
                self.bootstraps.append(self.bootstrap_idxs[i])

        self.oob_score = self.compute_oob_score(x, y)

        return self

    def create_tree(self, x, y, i):
        """
        Grows a survival tree for the bootstrap samples.
        :param y: label data frame y with survival time as the first column and event as second
        :param x: feature data frame x
        :param i: Indices
        :return: SurvivalTree
        """
        n_features = int(round(np.sqrt(x.shape[1]), 0))
        if self.random_state is None:
            f_idxs = np.random.permutation(x.shape[1])[:n_features]
        else:
            f_idxs = np.random.RandomState(seed=self.random_states[i]).permutation(x.shape[1])[:n_features]

        tree = SurvivalTree(x=x.iloc[self.bootstrap_idxs[i], :], y=y.iloc[self.bootstrap_idxs[i], :],
                            f_idxs=f_idxs, n_features=n_features, timeline=self.timeline,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf, random_state=self.random_states[i])

        return tree

    def compute_oob_ensembles(self, xs):
        """
        Compute OOB ensembles.
        :return: List of oob ensemble for each sample.
        """
        results = [compute_oob_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees,
                                            bootstraps=self.bootstraps) for sample_idx in range(xs.shape[0])]
        oob_ensemble_chfs = [i for i in results if not i.empty]
        return oob_ensemble_chfs

    def compute_oob_score(self, x, y):
        """
        Compute the oob score (concordance-index).
        :return: c-index of oob samples
        """
        oob_ensembles = self.compute_oob_ensembles(x)
        c = concordance_index(y_time=y.iloc[:, 0], y_pred=oob_ensembles, y_event=y.iloc[:, 1])
        return c

    def predict(self, xs):
        """
        Predict survival for xs.
        :param xs: The input samples
        :return: List of the predicted cumulative hazard functions.
        """
        ensemble_chfs = [compute_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees)
                         for sample_idx in range(xs.shape[0])]
        return ensemble_chfs

    def draw_bootstrap_samples(self, data):
        """
        Draw bootstrap samples
        :param data: Data to draw bootstrap samples of.
        :return: Bootstrap indices for each of the trees
        """
        bootstrap_idxs = []
        for i in range(self.n_estimators):
            no_samples = len(data)
            data_rows = range(no_samples)
            if self.random_state is None:
                bootstrap_idx = np.random.choice(data_rows, no_samples)
            else:
                np.random.seed(self.random_states[i])
                bootstrap_idx = np.random.choice(data_rows, no_samples)
            bootstrap_idxs.append(bootstrap_idx)

        return bootstrap_idxs


def compute_ensemble_chf(sample_idx, xs, trees):
    denominator = 0
    numerator = 0
    for b in range(len(trees)):
        sample = xs.iloc[sample_idx].to_list()
        chf = trees[b].predict(sample)
        denominator = denominator + 1
        numerator = numerator + 1 * chf
    ensemble_chf = numerator / denominator
    return ensemble_chf


def compute_oob_ensemble_chf(sample_idx, xs, trees, bootstraps):
    denominator = 0
    numerator = 0
    for b in range(len(trees)):
        if sample_idx not in bootstraps[b]:
            sample = xs.iloc[sample_idx].to_list()
            chf = trees[b].predict(sample)
            denominator = denominator + 1
            numerator = numerator + 1 * chf
    if denominator != 0:
        oob_ensemble_chf = numerator / denominator
    else:
        oob_ensemble_chf = pd.Series()
    return oob_ensemble_chf
