from lifelines import NelsonAalenFitter
from .splitting import find_split
from .tree_helper import select_new_feature_indices


class Node:

    def __init__(self, x, y, tree, f_idxs, n_features, timeline, unique_deaths=3, min_leaf=3, random_state=None):
        """
        A Node of the Survival Tree.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param tree: The corresponding Survival Tree
        :param f_idxs: The indices of the features to use.
        :param timeline: The timeline used for the prediction.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.tree = tree
        self.f_idxs = f_idxs
        self.timeline = timeline
        self.n_features = n_features
        self.unique_deaths = unique_deaths
        self.random_state = random_state
        self.min_leaf = min_leaf
        self.score = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.chf_terminal = None
        self.terminal = False
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

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = find_split(self)

        if self.split_var is None:
            self.compute_terminal_node()
            return self

        lf_idxs, rf_idxs = select_new_feature_indices(self.random_state, self.x, self.n_features)

        self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree, lf_idxs,
                        self.n_features, timeline=self.timeline, min_leaf=self.min_leaf, random_state=self.random_state)

        self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree, rf_idxs,
                        self.n_features, timeline=self.timeline, min_leaf=self.min_leaf, random_state=self.random_state)

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
            return self.tree.chf

        else:
            if x[self.split_var] <= self.split_val:
                self.lhs.predict(x)
            else:
                self.rhs.predict(x)
