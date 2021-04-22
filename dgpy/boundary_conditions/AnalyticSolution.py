import numpy as np
from dgpy.domain import BoundaryCondition


class AnalyticSolution:
    def __init__(self, boundary_condition_type, solution):
        assert boundary_condition_type in [
            BoundaryCondition.DIRICHLET, BoundaryCondition.NEUMANN
        ]
        self.boundary_condition_type = boundary_condition_type
        self.solution = solution

    def nonlinear(self, u, n_dot_flux, face):
        if self.boundary_condition_type == BoundaryCondition.DIRICHLET:
            return self.solution(face.inertial_coords), n_dot_flux
        else:
            return u, face.normal_dot(self.solution(face.inertial_coords))

    def linear(self, u_correction, n_dot_flux_correction, face):
        if self.boundary_condition_type == BoundaryCondition.DIRICHLET:
            return np.zeros(u_correction.shape), n_dot_flux_correction
        else:
            return u_correction, np.zeros(n_dot_flux_correction.shape)
