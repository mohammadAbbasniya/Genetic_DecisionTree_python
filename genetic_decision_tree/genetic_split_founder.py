from .decision_tree_criteria import Criterion
from .genetic_chromo import Chromo, Operator
import random
from math import floor


class GeneticSplitFinder:
    def __init__(self,
                 criterion: Criterion,
                 K: int,
                 max_iterations: int,
                 random_state: int):
        self.criterion = criterion
        self.K = K
        self.max_iterations = max_iterations
        self.random = random.Random()
        if random_state is not None:
            self.random.seed(random_state)

    def random_generation(self, indices):
        generation = []
        for _ in range(self.K):
            chromo = Chromo(feature=self.random.randint(0, self.criterion.n_features - 1),
                            opr=Operator.from_value(self.random.randint(0, 3)),
                            percent=self.random.randint(0, 100))
            chromo.calc_fitness(indices, self.criterion)
            generation.append(chromo)
        return generation

    def run(self, indices):
        generation = self.random_generation(indices)
        best_chromo = generation[0]
        iteration = 0
        crossovers = floor(self.K * 0.8)

        while iteration <= self.max_iterations and best_chromo.fitness > 0:
            generation.sort(key=lambda chromo: chromo.fitness)
            new_generation = []

            # ----- select chromosomes for crossover -----
            for i in range(crossovers):
                crossover_res = generation[i].crossover(generation[-i])
                crossover_res.calc_fitness(indices, self.criterion)
                new_generation.append(crossover_res)

            # ----- mutate some chromosomes -----
            for _ in range(self.K - crossovers):
                i = self.random.randint(0, self.K - 1)
                generation[i].mutate(self.criterion.n_features, self.random)
                generation[i].calc_fitness(indices, self.criterion)
                new_generation.append(generation[i])

            # ----- update best chromosome -----
            generation = new_generation
            for i in range(self.K):
                if generation[i].fitness < best_chromo.fitness:
                    best_chromo = generation[i]

            iteration += 1

        return best_chromo
