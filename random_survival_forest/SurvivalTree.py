from .Node import Node
from .splitting import find_split
from .tree_helper import select_new_feature_indices


class SurvivalTree:

    def __init__(self, x, y, f_idxs, n_features, timeline, unique_deaths=3, min_leaf=3, random_state=None):
        """
        A Survival Tree to predict survival.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param timeline: The timeline used for the prediction.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.timeline = timeline
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.random_state = random_state
        self.score = 0
        self.index = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.prediction_possible = None
        self.grow_tree()

    def grow_tree(self):
        """
        Grow the survival tree recursively as nodes.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 1].reset_index().drop_duplicates().sum()[1]

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = find_split(self)

        if self.split_var is not None and unique_deaths > self.unique_deaths:
            self.prediction_possible = True
            lf_idxs, rf_idxs = select_new_feature_indices(self.random_state, self.x, self.n_features)

            self.lhs = Node(x=self.x.iloc[lhs_idxs_opt, :], y=self.y.iloc[lhs_idxs_opt, :],
                            tree=self, f_idxs=lf_idxs, n_features=self.n_features, timeline=self.timeline,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf, random_state=self.random_state)

            self.rhs = Node(x=self.x.iloc[rhs_idxs_opt, :], y=self.y.iloc[rhs_idxs_opt, :],
                            tree=self, f_idxs=rf_idxs, n_features=self.n_features, timeline=self.timeline,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf, random_state=self.random_state)

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
