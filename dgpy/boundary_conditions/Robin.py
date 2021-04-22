import numpy as np
from dgpy.domain import BoundaryCondition


class Robin:
    def __init__(self, dirichlet_weight, neumann_weight, constant):
        self.dirichlet_weight = dirichlet_weight
        self.neumann_weight = neumann_weight
        self.constant = constant

    def nonlinear(self, u, n_dot_flux, face):
        return u, ((self.constant - self.dirichlet_weight * u) /
                   self.neumann_weight)

    def linear(self, u_correction, n_dot_flux_correction, face):
        return u_correction, (-self.dirichlet_weight / self.neumann_weight *
                              u_correction)
