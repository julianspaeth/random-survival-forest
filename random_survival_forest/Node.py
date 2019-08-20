import numpy as np
from lifelines import NelsonAalenFitter
from . import splitting


class Node:

    score = 0
    split_val = None
    split_var = None
    lhs = None
    rhs = None
    chf = None
    chf_terminal = None
    terminal = False

    def __init__(self, x, y, tree, f_idxs, n_features, unique_deaths=1, min_leaf=1, random_state=None):
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
        self.random_state = random_state
        self.min_leaf = min_leaf
        self.grow_tree()

    def grow_tree(self):
        """
        Grow tree by calculating the Nodes recursively.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 1].reset_index().drop_duplicates().sum()[1]

        if unique_deaths <= self.unique_deaths:
            self.compute_terminal_node()
            return self

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = splitting.find_split(self)

        if self.split_var is None:
            self.compute_terminal_node()
            return self

        if self.random_state is None:
            lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
            rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        else:
            lf_idxs = np.random.RandomState(seed=self.random_state).permutation(self.x.shape[1])[:self.n_features]
            rf_idxs = np.random.RandomState(seed=self.random_state).permutation(self.x.shape[1])[:self.n_features]

        self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree,
                        lf_idxs, self.n_features, min_leaf=self.min_leaf, random_state=self.random_state)

        self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree,
                        rf_idxs, self.n_features, min_leaf=self.min_leaf, random_state=self.random_state)

        return self

    def compute_terminal_node(self):
        """
        Compute the terminal node if condition has reached.
        :return: self
        """
        self.terminal = True
        self.chf = NelsonAalenFitter()
        t = self.y.iloc[:, 0]
        e = self.y.iloc[:, 1]
        self.chf.fit(t, event_observed=e, timeline=self.tree.timeline)

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
            return self.tree.chf

        else:
            if x[self.split_var] <= self.split_val:
                self.lhs.predict(x)
            else:
                self.rhs.predict(x)
