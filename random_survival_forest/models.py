import multiprocessing

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter
from sklearn.utils import check_random_state

from random_survival_forest.scoring import concordance_index
from random_survival_forest.splitting import _find_split


class RandomSurvivalForest:

    def __init__(self, n_estimators: int = 100, min_leaf: int = 3, unique_deaths: int = 3, n_jobs: int or None = None,
                 oob_score: bool = False, timeline=None, random_state=None):
        """
        A Random Survival Forest is a prediction model especially designed for survival analysis.
        :param n_estimators: The numbers of trees in the forest.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param n_jobs: The number of jobs to run in parallel for fit. None means 1.
        """
        self.n_estimators = n_estimators
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.n_jobs = n_jobs
        self.bootstrap_idxs = None
        self.bootstraps = []
        self.oob_idxs = None
        self.oob_score = oob_score
        self.trees = []
        self.timeline = timeline
        self.random_state = random_state
        self.random_instance = check_random_state(self.random_state)

    def fit(self, x, y):
        """
        Build a forest of trees from the training set (X, y).
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event
        in the second with the shape [n_samples, 2]
        :return: self: object
        """

        try:
            if self.timeline is None:
                self.timeline = y.iloc[:, 1].sort_values().unique()
        except Exception:
            raise (
                "Timeline seems to contain float values. Please provide a custom timeline in the RandomSurvivalForest "
                "constructor. "
                "For example: RandomSurivalForest(timeline=range(y.iloc[:, 1].min(), y.iloc[:, 1].max(), 0.1)")

        self.bootstrap_idxs = self._draw_bootstrap_samples(x)

        num_cores = multiprocessing.cpu_count()

        if self.n_jobs > num_cores or self.n_jobs == -1:
            self.n_jobs = num_cores
        elif self.n_jobs is None:
            self.n_jobs = 1

        trees = Parallel(n_jobs=self.n_jobs)(delayed(self._create_tree)(x, y, i) for i in range(self.n_estimators))

        for i in range(len(trees)):
            if trees[i].prediction_possible:
                self.trees.append(trees[i])
                self.bootstraps.append(self.bootstrap_idxs[i])

        if self.oob_score:
            self.oob_score = self.compute_oob_score(x, y)

        return self

    def _create_tree(self, x, y, i: list):
        """
        Grows a survival tree for the bootstrap samples.
        :param y: label data frame y with survival time as the first column and event as second
        :param x: feature data frame x
        :param i: Indices
        :return: SurvivalTree
        """
        n_features = int(round(np.sqrt(x.shape[1]), 0))
        f_idxs = self.random_instance.permutation(x.shape[1])[:n_features]
        tree = SurvivalTree(x=x.iloc[self.bootstrap_idxs[i], :], y=y.iloc[self.bootstrap_idxs[i], :],
                            f_idxs=f_idxs, n_features=n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

        return tree

    def _compute_oob_ensembles(self, xs):
        """
        Compute OOB ensembles.
        :return: List of oob ensemble for each sample.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_oob_ensemble_chf)(sample_idx, xs, self.trees, self.bootstraps) for sample_idx in
            range(xs.shape[0]))
        oob_ensemble_chfs = [i for i in results if not i.empty]
        return oob_ensemble_chfs

    def compute_oob_score(self, x, y):
        """
        Compute the oob score (concordance-index).
        :return: c-index of oob samples
        """
        oob_ensembles = self._compute_oob_ensembles(x)
        c = concordance_index(y_time=y.iloc[:, 1], y_pred=oob_ensembles, y_event=y.iloc[:, 0])
        return c

    def predict(self, xs):
        """
        Predict survival for xs.
        :param xs: The input samples
        :return: List of the predicted cumulative hazard functions.
        """
        ensemble_chfs = [self._compute_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees)
                         for sample_idx in range(xs.shape[0])]
        return ensemble_chfs

    def _draw_bootstrap_samples(self, data):
        """
        Draw bootstrap samples
        :param data: Data to draw bootstrap samples of.
        :return: Bootstrap indices for each of the trees
        """
        bootstrap_idxs = []
        for i in range(self.n_estimators):
            no_samples = len(data)
            data_rows = range(no_samples)
            bootstrap_idx = self.random_instance.choice(data_rows, no_samples)
            bootstrap_idxs.append(bootstrap_idx)

        return bootstrap_idxs

    def _compute_ensemble_chf(self, sample_idx: int, xs, trees: list):
        denominator = 0
        numerator = 0
        for b in range(len(trees)):
            sample = xs.iloc[sample_idx].to_list()
            chf = trees[b].predict(sample)
            denominator = denominator + 1
            numerator = numerator + 1 * chf
        ensemble_chf = numerator / denominator
        return ensemble_chf

    def _compute_oob_ensemble_chf(self, sample_idx: int, xs, trees: list, bootstraps: list):
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


class SurvivalTree:

    def __init__(self, x, y, f_idxs, n_features, random_instance, timeline, unique_deaths=3, min_leaf=3):
        """
        A Survival Tree to predict survival.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.score = 0
        self.index = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.prediction_possible = None
        self.timeline = timeline
        self.random_instance = random_instance
        self.grow_tree()

    def grow_tree(self):
        """
        Grow the survival tree recursively as nodes.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 0].reset_index().drop_duplicates().sum()[1]

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = _find_split(self)

        if self.split_var is not None and unique_deaths > self.unique_deaths:
            self.prediction_possible = True
            lf_idxs, rf_idxs = _select_new_feature_indices(self.x, self.n_features, self.random_instance)

            self.lhs = Node(x=self.x.iloc[lhs_idxs_opt, :], y=self.y.iloc[lhs_idxs_opt, :],
                            tree=self, f_idxs=lf_idxs, n_features=self.n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

            self.rhs = Node(x=self.x.iloc[rhs_idxs_opt, :], y=self.y.iloc[rhs_idxs_opt, :],
                            tree=self, f_idxs=rf_idxs, n_features=self.n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

            return self
        else:
            self.prediction_possible = False
            return self

    def predict(self, x):
        """
        Predict survival for x.
        :param x: The input sample.
        :return: The predicted cumulative hazard function.
        """
        if x[self.split_var] <= self.split_val:
            self.lhs.predict(x)
        else:
            self.rhs.predict(x)
        return self.chf


class Node:

    def __init__(self, x, y, tree: SurvivalTree, f_idxs: list, n_features: int, timeline, random_instance,
                 unique_deaths: int = 3, min_leaf: int = 3):
        """
        A Node of the Survival Tree.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param tree: The corresponding Survival Tree
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.tree = tree
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.unique_deaths = unique_deaths
        self.min_leaf = min_leaf
        self.score = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.chf_terminal = None
        self.terminal = False
        self.timeline = timeline
        self.random_instance = random_instance
        self.grow_tree()

    def grow_tree(self):
        """
        Grow tree by calculating the Nodes recursively.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 0].reset_index().drop_duplicates().sum()[1]

        if unique_deaths <= self.unique_deaths:
            self.compute_terminal_node()
            return self

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = _find_split(self)

        if self.split_var is None:
            self.compute_terminal_node()
            return self

        lf_idxs, rf_idxs = _select_new_feature_indices(self.x, self.n_features, self.random_instance)

        self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree, lf_idxs,
                        self.n_features, min_leaf=self.min_leaf, timeline=self.timeline,
                        random_instance=self.random_instance)

        self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree, rf_idxs,
                        self.n_features, min_leaf=self.min_leaf, timeline=self.timeline,
                        random_instance=self.random_instance)

        return self

    def compute_terminal_node(self):
        """
        Compute the terminal node if condition has reached.
        :return: self
        """
        self.terminal = True
        self.chf = NelsonAalenFitter()
        t = self.y.iloc[:, 1]
        e = self.y.iloc[:, 0]
        self.chf.fit(t, event_observed=e, timeline=self.timeline)
        return self

    def predict(self, x):
        """
        Predict the cumulative hazard function if its a terminal node. If not walk through the tree.
        :param x: The input sample.
        :return: Predicted cumulative hazard function if terminal node
        """
        if self.terminal:
            self.tree.chf = self.chf.cumulative_hazard_
            self.tree.chf = self.tree.chf.iloc[:, 0]
            return self.tree.chf.dropna()

        else:
            if x[self.split_var] <= self.split_val:
                self.lhs.predict(x)
            else:
                self.rhs.predict(x)


def _select_new_feature_indices(x, n_features: int, random_instance):
    lf_idxs = random_instance.permutation(x.shape[1])[:n_features]
    rf_idxs = random_instance.permutation(x.shape[1])[:n_features]

    return lf_idxs, rf_idxs
