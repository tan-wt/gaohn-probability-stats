import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1992)
np.random.seed(1992)

# create a random variable X with 1000 samples of height 1-10 cm uniformly distributed
true_population = np.arange(1, 11, 1)
# repeat 100 times for each height 1-10 cm. So we have 100 samples of height 1 cm, 100 samples of height 2 cm, etc.
true_population = true_population.repeat(100)

# Let X be the random variable of the height of a person in the true population.
# Then since we know the true distribution, the PMF is completely determined
# i.e if I picked a person at random, and he turns out to be 5cm tall, then the probability
# of him being 5cm tall is 100/1000 = 0.1 P(X=1)=0.1, in fact P(X=2)=0.1, P(X=3)=0.1, P(X=4)=0.1, P(X=5)=0.1, P(X=6)=0.1, P(X=7)=0.1, P(X=8)=0.1, P(X=9)=0.1, P(X=10)=0.1

# map X(\S) to a probability
def pmf(x, population):
    return np.sum(population == x) / len(population)


print(f"PMF of X = 1: {pmf(1, true_population)}")

# ideal histogram = pmf
plt.stem(true_population, [pmf(x, true_population) for x in true_population])
plt.title("PMF of X")
plt.xlabel("x")
plt.ylabel("P(X=x)")
plt.xlim(0, 11)
plt.xticks(np.arange(1, 11, 1))
plt.show()

# this is a random sample of 100 people from the true population drawn uniformly at random replace false
sample = np.random.choice(true_population, 100, replace=False)

# empirical histogram
plt.stem(sample, [pmf(x, sample) for x in sample])
plt.title("Empirical histogram of sample drawn from X")
plt.xlabel("x")
plt.ylabel("P(X=x)")
plt.xlim(0, 11)
plt.xticks(np.arange(1, 11, 1))
plt.show()

# can tell that it is not very close to the true pmf (true distribution)
# but like prof said if we take a large enough sample, it will be close to the true pmf
# this is obvious if we take say 900 out of 1000, then it will be closer to the true pmf
# simply because we have more representation across samples. Like
# if we only take 100 samples, maybe we a bit unlucky and get only 1 sample of height 5cm, then
# that empirical histogram will be very far from the true pmf since the prob of 5 cm in that
# example became 1/100 = 0.01!

large_sample = np.random.choice(true_population, 900, replace=False)
plt.stem(large_sample, [pmf(x, large_sample) for x in large_sample])
plt.title("Empirical histogram of sample drawn from X")
plt.xlabel("x")
plt.ylabel("P(X=x)")
plt.xlim(0, 11)
plt.xticks(np.arange(1, 11, 1))
plt.show()


def expected_value(population):
    return np.sum(population) / len(population)


print(f"Expected value of the true population: {expected_value(true_population)}")


def empirical_mean(sample):
    return np.sum(sample) / len(sample)


print(f"Empirical mean of the sample: {empirical_mean(sample)}")
