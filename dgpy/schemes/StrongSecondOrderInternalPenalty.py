import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..operators import *

def apply_strong_second_order_internal_penalty_operator(x, domain, fluxes, sources, penalty_parameter, lifting_scheme, order):
    domain.set_data(x, 'u', fields_valence=(0,), order=order)
    domain.apply(compute_deriv, 'u', 'grad_u')
    domain.apply(fluxes, 'grad_u', 'F')
    domain.apply(compute_div, 'F', 'divF')
    domain.apply(sources, 'u', 'S')
    for element in domain.elements:
        element.Au = compute_mass(-element.divF + element.S, element)
        element.slice_to_faces('F')
        element.slice_to_faces('u')
    for element in domain.elements:
        for face in element.get_internal_faces():
            face.F_tilde = fluxes(face.normal_times(face.jump('u')), element)
            face.jumpF = face.normal_dot(face.jump('F'))
            sigma = penalty(face, penalty_parameter)
            element.Au += lift_flux(sigma * face.normal_dot(face.F_tilde) + 0.5 * face.jumpF, face, scheme=lifting_scheme)
            element.Au -= 0.5 * lift_deriv_flux(face.F_tilde, face, scheme=lifting_scheme)
        for face in element.get_external_faces():
            face.F_tilde = fluxes(face.normal_times(face.u), element)
            sigma = penalty(face, penalty_parameter)
            element.Au += 2. * lift_flux(sigma * face.normal_dot(face.F_tilde), face, scheme=lifting_scheme)
            element.Au -= lift_deriv_flux(face.F_tilde, face, scheme=lifting_scheme)
    return domain.get_data('Au', order=order)

def compute_strong_second_order_internal_penalty_source(source_field, domain, fluxes, dirichlet_boundary_field, penalty_parameter, lifting_scheme, order):
    for element in domain.elements:
        f = getattr(element, source_field)
        element.b = compute_mass(f, element)
        for face in element.get_external_faces():
            F_tilde_dirichlet = fluxes(face.normal_times(np.take(getattr(element, dirichlet_boundary_field), face.slice_index(), axis=face.dimension)), element)
            sigma = penalty(face, penalty_parameter)
            element.b += 2. * lift_flux(sigma * face.normal_dot(F_tilde_dirichlet), face, scheme=lifting_scheme)
            element.b -= lift_deriv_flux(F_tilde_dirichlet, face, scheme=lifting_scheme)
    return domain.get_data('b', order=order)

class DgOperator(LinearOperator):
    def __init__(self, domain, fluxes, sources, field_valences=(0,), penalty_parameter=1.5, lifting_scheme='quadrature', order='C'):
        self.domain = domain
        self.fluxes = fluxes
        self.sources = sources
        self.penalty_parameter = penalty_parameter
        self.field_valences = field_valences
        self.lifting_scheme = lifting_scheme
        self.order = order
        N = domain.get_total_num_points() * np.sum(domain.dim**np.array(self.field_valences))
        super().__init__(shape=(N, N), dtype=np.float64)
    
    def _matvec(self, x):
        return apply_strong_second_order_internal_penalty_operator(x, self.domain, self.fluxes, self.sources, self.penalty_parameter, self.lifting_scheme, self.order)
    
    def compute_source(self, source_field, dirichlet_boundary_field):
        return compute_strong_second_order_internal_penalty_source(source_field, self.domain, self.fluxes, dirichlet_boundary_field, self.penalty_parameter, self.lifting_scheme, self.order)