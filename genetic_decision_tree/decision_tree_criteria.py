from abc import ABC, abstractmethod

possible_criterion = ['gini']


def get_criterion_by_name(name: str, data):
    name = name.lower()
    if name == 'gini':
        return Gini(data)


class Criterion(ABC):
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
        self.n_classes = len(set(self.y))
        self.n_features = self.X.shape[1]

    @abstractmethod
    def gain(self, n1, n2, impurity1, impurity2):
        pass

    @abstractmethod
    def impurity(self, indices):
        pass

    def apply_chromo(self, indices, chromo):
        left_indices, right_indices = [], []
        for i in indices:
            if chromo.perform_on_sample(self.X[i]):
                right_indices.append(i)
            else:
                left_indices.append(i)

        return left_indices, right_indices

    def prediction(self, indices):
        n_each_class = [0 for _ in range(self.n_classes)]
        for i in indices:
            n_each_class[self.y[i]] += 1

        return n_each_class.index(max(n_each_class))


class Gini(Criterion):
    def gain(self, n1, n2, impurity1, impurity2):
        n = n1 + n2
        return n1/n * impurity1 + n2/n * impurity2

    def impurity(self, indices):
        n_each_class = [0 for _ in range(self.n_classes)]
        for i in indices:
            n_each_class[self.y[i]] += 1

        n = len(indices)
        if n == 0:
            return 0
        res_gini = 1
        for i in range(self.n_classes):
            res_gini -= (n_each_class[i]/n)**2

        return res_gini

