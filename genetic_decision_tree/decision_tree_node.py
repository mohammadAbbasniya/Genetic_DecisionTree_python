from enum import Enum
from .genetic_split_founder import GeneticSplitFinder


class NodeType(Enum):
    M = 0  # middle node
    T = 1  # terminal node


class Node:
    def __init__(self,
                 indices: list,
                 depth: int,
                 split_finder: GeneticSplitFinder):
        self.type = NodeType.T
        self.indices = indices
        self.depth = depth
        self.split_finder = split_finder
        self.impurity = split_finder.criterion.impurity(indices)
        self.prediction = split_finder.criterion.prediction(indices)
        self.right = None
        self.left = None
        self.chromo = None

    def split(self):
        assert self.type != NodeType.M, "A middle node can't perform split"

        self.chromo = self.split_finder.run(self.indices)
        left_indices, right_indices = self.split_finder.criterion.apply_chromo(self.indices, self.chromo)

        self.left = Node(left_indices, self.depth + 1, self.split_finder)
        self.right = Node(right_indices, self.depth + 1, self.split_finder)
        self.type = NodeType.M
        return True  # Send a flag to show the success of progression

    def classify(self, x_sample):
        if self.type == NodeType.T:
            return self.prediction
        elif self.type == NodeType.M:
            if self.chromo.perform_on_sample(x_sample):
                return self.right.classify(x_sample)
            else:
                return self.left.classify(x_sample)
        else:
            return None
