# Genetic based Decision-Tree in Python
This repository contains an implementation of **Decision Tree Classifier** which employs **Genetic Algorithm** for finding optimal split on every node of the tree. No matter how many columns or how many classes is in your dataset, as long as it contains numerical rows, this package would classify it for you!

### Requirements
1. `numpy`
2. `sklearn`
  <br> But wait 😅, it's not going to be a part of FROM-SCRATCH implementation, I only used it in [main.ipynb](https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/main.ipynb) and [main.py](https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/main.py) for the use of **train_test_split**, **confusion_matrix** and **accuracy_score** functions, not more, I promise.
3. `graphviz` 
  <br> This is a relatively large package, it is only used in [main.ipynb](https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/main.ipynb) for the visualisation of decision tree expansion. You don't have to install it if you are **Not** going to export the decision tree as an image.

## About Decision-tree
A decision tree is a decision support hierarchical model that uses a tree-like model of decisions and their possible consequences. Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning. A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules [[Wiki](https://en.wikipedia.org/wiki/Decision_tree)]. An example of decision tree:
<p align='center'>
  <img alt="decision-tree-example" src="https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/README.imgs/decision-tree-example.png">
</p>

## About Genetic-algorithm
In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection. Some examples of GA applications include optimizing decision trees for better performance, solving sudoku puzzles, hyperparameter optimization, causal inference, etc. [[Wiki](https://en.wikipedia.org/wiki/Genetic_algorithm)].
<p align='center'>
  <img height="300" alt="genetic-example" src="https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/README.imgs/genetic-example.jpg">
</p>

## Project structure
- ### 📂 directory [genetic_decision_tree]
  - #### 📄 [genetic_chromo.py]
      This file contains `Chromo` class. It's the representation of problem in genetic. Each chromosome should represent a valid 'Split' on dataset. Chromosomes consist of three Genes: <br> 1) feature (column) to be used for splitting data <br> 2) operator for comparing each sample (can be >, <, >= or <=) <br> 3) A percentage by which we decide each sample should go to right-set or left-set <br><br> *For example*, chromosome [X1, >, 20] will split data in this way: for each sample, if the value of X1 is greater than 20% it will be in righ-set, otherwise left-set. For more explaintion, consider these values to be in column X1 of dataset: {4, 1, 8, 3, 6, 25}, applying that chromosome on these values will split them into right:{8, 6, 25} and left:{4, 1, 3}.

  - #### 📄 [genetic_split_founder.py]
      This file contains `GeneticSplitFinder` class. It's where the genetic algorithm of project is implemented. This class takes indices of rows in dataset (instead of rows themself, to decrease memory usage) and finds a good chromosome that can perform optimal split on these rows. Note that parameters of genetic algorithm like population size (K), number of iterations, criteria etc., are taken through the `constructor (__init__)` of this class, but indices of intended rows to be split are taken by `run` method.

  - #### 📄 [genetic_decision_tree_classifier.py]
      This file contains `GeneticDecisionTreeClassifier` class. This is the class in which the decision-tree is implemented. When `fit(X, y)` is called, it takes a copy of X and y as internal values and the scales each column of X to [0-100]. Then, expands the tree level-order by constantly asking `GeneticSplitFinder.run` for an optimal split.

  - #### 📄 [decision_tree_node.py]
      This file contains `Node` class. This is a node in our decision-tree; each node can perform `classify` on a sample; if the node is  terminal so it knows the class, but if it's a middle node, it decides left or right child should classify that sample.

  - #### 📄 [decision_tree_criteria.py]
      This file contains `Criterion` and `Gini` classes. The `Criterion` class holds internal X and y; it can apply any chromosome on any part of dataset mentioned by indices of rows, but it's an abstract class with two unimplemented methods `gain` and `impurity`. These methods should be implemented in subclasses which are going to be a criterion for splitting data. The `Gini` class is a subclass for `Criterion` and implemented its abstract methods based on gini index formula: <br>
```math
impurity\left(t\right)=1-\sum_{j}\left[p\left(j\middle| t\right)\right]^2
~~~~~~~~~~~
{gain}_{split} = \sum_{i=1}^{k} \frac{n_i}{n} impurity \left( i \right)
```

- ### 📄 file [main.ipynb] and [main.py]
    These files are my Demo files that show how to make in-use this classifier on a dataset, although it's very simple. At the end of `main.ipynb` there is a cell that uses `graphviz` library and exports the whole decision tree as an image like this:
<p align='center'>
  <img height="400" alt="genetic-example" src="https://github.com/mohammadAbbasniya/Genetic_DecisionTree_python/blob/main/outputs/render.png">
</p>

- ### 📂 directory [inputs]
    This directory constains some input data for classification test.

- ### 📂 directory [outputs]
    This directory contains output files of project (exports of `graphviz`).


## TODO list
1. Implement other criteria like `entropy` and `Error`
2. Apply the model on well-knowns datasets
3. Create `Pypi` package







