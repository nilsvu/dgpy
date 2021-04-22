import numpy as np
from dgpy.domain import BoundaryCondition


class Zero:
    def __init__(self, boundary_condition_type):
        assert boundary_condition_type in [
            BoundaryCondition.DIRICHLET, BoundaryCondition.NEUMANN
        ]
        self.boundary_condition_type = boundary_condition_type

    def nonlinear(self, u, n_dot_flux, face):
        if self.boundary_condition_type == BoundaryCondition.DIRICHLET:
            return np.zeros(u.shape), n_dot_flux
        else:
            return u, np.zeros(n_dot_flux.shape)

    def linear(self, u_correction, n_dot_flux_correction, face):
        return self.nonlinear(u_correction, n_dot_flux_correction, face)
