import numpy as np


# The code is based on:
# https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
class Hyperband:
    def __init__(self, resource=81, eta=3):
        self.resource = resource
        self.eta = eta
        self.count = int(np.log(resource) / np.log(eta)) + 1
        self.budget = self.count * resource

    def run(self, get, test):
        for i in reversed(range(self.count)):
            r = self.resource * self.eta**(-i)
            n = int(np.ceil(
                self.budget / self.resource / (i + 1) * self.eta**i))
            c = get(n)
            for j in range(i + 1):
                r_j = r * self.eta**j
                n_j = n * self.eta**(-j)
                ranking = np.argsort(test(r_j, c))
                c = [c[k] for k in ranking[0:int(n_j / self.eta)]]
