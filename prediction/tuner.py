import numpy as np


# The code is based on:
# https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
class Hyperband:
    def __init__(self, resource=81, eta=3):
        self.resource = resource
        self.eta = eta
        self.count = int(np.log(resource) / np.log(eta)) + 1

    def run(self, generate, assess):
        best = None
        for i in reversed(range(self.count)):
            r = self.resource * self.eta**(-i)
            n = int(np.ceil(self.count // (i + 1) * self.eta**i))
            c = generate(n)
            for j in range(i + 1):
                r_j = r * self.eta**j
                n_j = n * self.eta**(-j)
                scores = assess(r_j, c)
                ranking = np.argsort(scores, kind='mergesort')
                if best is None or scores[ranking[0]] < best[-1]:
                    best = (c[ranking[0]], r_j, scores[ranking[0]])
                c = [c[k] for k in ranking[:int(n_j / self.eta)]]
        return best
