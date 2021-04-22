import numpy as np
from numpy import sin, cos


class ProductOfSinusoids:
    def __init__(self, wave_numbers):
        self.wave_numbers = wave_numbers
        self.dim = len(wave_numbers)

    def phase(self, x):
        return np.array([self.wave_numbers[d] * x[d] for d in range(len(x))])

    def field(self, x):
        return np.prod(sin(self.phase(x)), axis=0)

    def auxiliary_field(self, x):
        phi = self.phase(x)
        return np.array([
            self.wave_numbers[i] * cos(phi[i]) *
            np.prod(sin(np.delete(phi, i, axis=0)), axis=0)
            for i in range(self.dim)
        ])

    def source(self, x):
        return np.sum(np.asarray(self.wave_numbers)**2) * self.field(x)
