import numpy as np
import matplotlib.pyplot as plt
import random


def cdf(x, population):
    return np.sum(population <= x) / len(population)


def empiricial_cdf(sample: np.ndarray):
    """Empirical distribution of the sample."""
    return np.cumsum(np.bincount(sample)) / len(sample)


# variance of the true population
# notice that we first need to compute the expected value of the true population = 5.5
# then we define a new r.v Y = X - E(X) which is like Y = [-4.5, ..., 0.5, ..., 4.5]
# What this Y holds is the **deviations** of the true populations's height from the average
# Then we take the expected/average on this Y to get the "average deviation" of the true population
# note we square because negative + positive cancels out sometimes.
Y = true_population - expected_value(true_population)
print(Y)


def variance(population):
    return np.sum((population - expected_value(population)) ** 2) / len(population)
