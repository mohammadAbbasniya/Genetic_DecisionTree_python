from enum import Enum
from random import Random
from .decision_tree_criteria import Criterion


class Operator(Enum):
    L = 0  # < : lower than
    G = 1  # > : grater than
    LE = 2  # <= : lower or equal to
    GE = 3  # >= : grater or equal to

    @staticmethod
    def from_value(value):
        return list(Operator)[value]


class Chromo:
    def __init__(self, feature: int, opr: Operator, percent: int):
        self.feature = feature
        self.opr = opr
        self.percent = percent
        self.fitness = None

    def crossover(self, other):
        return Chromo(self.feature, self.opr, other.percent)

    def mutate(self, n_features, random: Random):
        choice = random.choices([0, 1, 2], weights=[20, 30, 50], k=1)[0]
        if choice == 0:
            self.feature = random.randint(0, n_features - 1)
        elif choice == 1:
            self.opr = Operator.from_value(random.randint(0, 3))
        elif choice == 2:
            self.percent = random.randint(0, 100)

    def calc_fitness(self, indices, criterion: Criterion):
        left_indices, right_indices = criterion.apply_chromo(indices, self)
        left_impurity = criterion.impurity(left_indices)
        right_impurity = criterion.impurity(right_indices)
        self.fitness = criterion.gain(len(left_indices), len(right_indices), left_impurity, right_impurity)

    def perform_on_sample(self, x):
        if self.opr == Operator.L:
            return x[self.feature] < self.percent
        elif self.opr == Operator.G:
            return x[self.feature] > self.percent
        elif self.opr == Operator.LE:
            return x[self.feature] <= self.percent
        elif self.opr == Operator.GE:
            return x[self.feature] >= self.percent
        return None

    def __str__(self):
        return f'Chromo [{self.fitness:.3f}] {"{"} feature: X{self.feature}, operator: {self.opr}, percent: {self.percent} {"}"}'


