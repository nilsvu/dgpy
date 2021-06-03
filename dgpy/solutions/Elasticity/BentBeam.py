import numpy as np
from ...systems.Elasticity import IsotropicHomogeneous

class BentBeam:
    def __init__(self, length, height, bending_moment, bulk_modulus, shear_modulus):
        self.length = length
        self.height = height
        self.bending_moment = bending_moment
        self.material = IsotropicHomogeneous(bulk_modulus=bulk_modulus,
                                             shear_modulus=shear_modulus)

    def displacement(self, x):
        return np.array([
            -12. * self.bending_moment / self.material.youngs_modulus /
            self.height**3 * x[0] * x[1],
            6. * self.bending_moment / self.material.youngs_modulus /
            self.height**3 * (x[0]**2 + self.material.poisson_ratio * x[1]**2 -
                              self.length**2 / 4.)
        ])

    def source(self, x):
        return np.zeros(x.shape)
