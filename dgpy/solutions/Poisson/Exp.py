import numpy as np


class Exp:
    def field(self, x):
        return np.exp(np.sum(x, axis=0))

    def auxiliary_field(self, x):
        dim = len(x)
        return np.array([
            np.exp(np.sum(x, axis=0))
            for i in range(dim)
        ])

    def source(self, x):
        return -len(x) * np.exp(np.sum(x, axis=0))
