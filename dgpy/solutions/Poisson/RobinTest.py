import numpy as np
from numpy import exp


class RobinTest:
    def __init__(self, length):
        self.length = length

    def field(self, coords):
        x = coords[0]
        return (exp(self.length * x) - 1.) / (exp(self.length) - 1.)

    def auxiliary_field(self, coords):
        x = coords[0]
        return np.array([
            self.length * exp(self.length * x) / (exp(self.length) - 1.)
        ])

    def source(self, coords):
        x = coords[0]
        return -self.length**2 * exp(self.length * x) / (exp(self.length) - 1.)
