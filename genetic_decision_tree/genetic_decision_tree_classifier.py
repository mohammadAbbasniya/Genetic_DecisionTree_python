import numpy as np
from .decision_tree_node import Node
from .decision_tree_criteria import possible_criterion, get_criterion_by_name
from .genetic_split_founder import GeneticSplitFinder


class GeneticDecisionTreeClassifier:
    """
    Genetic Based Decision Tree Classifier.
    this only applies binary splits
    """

    def __init__(self,
                 criterion: str = 'gini',  # ``````````````\
                 max_depth: int = None,  # ````````````````| --> Decision Tree parameters
                 min_samples_split: int = 2,  # ``````````/

                 random_state: int = None,  # ````````````````\
                 genetic_max_iterations: int = 20,  # ````````| --> Genetic parameters
                 genetic_k: int = 16):  # ```````````````````/

        assert criterion in possible_criterion, f'possible criteria are: {possible_criterion}'
        assert max_depth is None or max_depth > 1, f'max_depth must be grater than 1, give: {max_depth}'
        assert min_samples_split > 1, f'min_samples_split must be grater than 1, give: {min_samples_split}'
        assert genetic_max_iterations > 0, f'genetic_max_iterations must be grater than 0, give: {genetic_max_iterations}'
        assert genetic_k > 0, f'genetic_k must be grater than 0, give: {genetic_k}'

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.genetic_max_iterations = genetic_max_iterations
        self.genetic_k = genetic_k
        self.n = None  # number of training samples
        self.m = None  # number of features in each sample
        self.data = None  # tuple like (X, y)
        self.head = None  # head of the tree
        self.feature_bounds = None  # min and max of each feature

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape must be a like a matrix (n, m)'
        self.n, self.m = X.shape
        assert len(y) == self.n, 'X and y must have the same number of samples'

        self._fit_scaler(X)
        _X = self._transfer_scales(X)
        _y = np.array([y[i] for i in range(self.n)])

        criterion = get_criterion_by_name(name=self.criterion, data=(_X, _y))
        genetic_split_finder = GeneticSplitFinder(criterion=criterion,
                                                  K=self.genetic_k,
                                                  max_iterations=self.genetic_max_iterations,
                                                  random_state=self.random_state)

        # perform level order Decision-Tree creation
        self.head = Node(indices=[i for i in range(self.n)], depth=0, split_finder=genetic_split_finder)
        q = [self.head]
        while len(q) > 0:
            node = q.pop(0)
            if node.impurity == 0:  # best impurity is zero
                continue
            if len(node.indices) <= self.min_samples_split:
                continue
            if self.max_depth is not None and node.depth >= self.max_depth:
                continue

            if node.split() is not None:
                q.append(node.left)
                q.append(node.right)

        return self

    def predict(self, X_test):
        assert len(X_test.shape) == 2 and X_test.shape[1] == self.m, 'Incompatible test shape'
        assert self.head is not None, 'You must call fit(X, y) first'
        _X_test = self._transfer_scales(X_test)
        res = [self.classify(_X_test[i]) for i in range(len(X_test))]
        return np.array(res)

    def classify(self, x_sample):
        assert self.head is not None, 'Decision tree didnt fit on dataset properly'
        return self.head.classify(x_sample)

    def _fit_scaler(self, X):
        self.feature_bounds = [{} for _ in range(self.m)]
        for i in range(self.m):
            self.feature_bounds[i]['min'] = X[:, i].min()
            self.feature_bounds[i]['max'] = X[:, i].max()

    def _transfer_scales(self, X):
        n, m = X.shape

        assert self.feature_bounds is not None, 'scaler did not fit'
        assert self.m == m, 'incompatible number of features'

        # take a copy of X and also scale all features to 0-100
        _X = np.zeros(shape=(n, m))
        for i in range(m):
            min_feature_i = self.feature_bounds[i]['min']
            max_feature_i = self.feature_bounds[i]['max']
            diff = max_feature_i - min_feature_i
            for r in range(n):
                _X[r][i] = ((X[r][i] - min_feature_i) / diff) * 100

        return _X
