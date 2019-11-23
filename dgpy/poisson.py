import numpy as np
from scipy.sparse.linalg import LinearOperator

from .operators import *


def apply_primal_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, 'u', fields_valence=(0,))
    for e in domain.elements:
        e.grad_u = compute_deriv(e.u, e)
#         e.Au = -quadrature(
#             e,
#             compute_div(e.grad_u, e).reshape(
#                 e.u.shape + e.dim * (1,)) * basis(e),
#             valence=0
#         )
        e.Au = -compute_mass(compute_div(e.grad_u, e), e)
        e.slice_to_faces('u')
        e.slice_to_faces('grad_u', 'internal')
    for e in domain.elements:
        for face in e.get_interior_faces():
            e.Au -= lift_internal_penalty(
                face.jump('u'),
                face,
                penalty_parameter
            )
        for face in e.get_internal_faces():
            e.Au += 0.5 * lift_flux(
                face.normal_dot(face.jump('grad_u')),
                face
            )
    return domain.get_data('Au')


def compute_primal_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    for e in domain.elements:
        f = getattr(e, source_field)
        e.b = quadrature(
            e,
            f.reshape(f.shape + e.dim * (1,)) * basis(e),
            valence=0
        )
        for face in e.get_external_faces():
            g = np.take(getattr(e, dirichlet_boundary_field),
                        face.slice_index(), axis=face.dimension)
            e.b -= lift_internal_penalty(
                2 * g,
                face,
                penalty_parameter
            )
    return domain.get_data('b')


def apply_massless_primal_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, 'u', fields_valence=(0,))
    for e in domain.elements:
        e.grad_u = compute_deriv(e.u, e)
        e.Au = -compute_div(e.grad_u, e)
        e.slice_to_faces('u')
        e.slice_to_faces('grad_u', 'internal')
    for e in domain.elements:
        for face in e.get_interior_faces():
            e.Au -= compute_inverse_mass(lift_internal_penalty(
                face.jump('u'),
                face,
                penalty_parameter
            ), e)
        for face in e.get_internal_faces():
            e.Au += 0.5 * compute_inverse_mass(lift_flux(
                face.normal_dot(face.jump('grad_u')),
                face
            ), e)
    return domain.get_data('Au')


def compute_massless_primal_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    for e in domain.elements:
        e.b = np.array(getattr(e, source_field))
        for face in e.get_external_faces():
            g = np.take(getattr(e, dirichlet_boundary_field),
                        face.slice_index(), axis=face.dimension)
            e.b -= compute_inverse_mass(lift_internal_penalty(
                2 * g,
                face,
                penalty_parameter
            ), e)
    return domain.get_data('b')


def apply_weak_primal_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, 'u', fields_valence=(0,))
    for e in domain.elements:
        e.grad_u = compute_deriv(e.u, e)
        e.Au = quadrature(
            e,
            np.einsum('d...,d...', np.reshape(
                e.grad_u, e.grad_u.shape + e.dim * (1,)), basis_deriv(e)),
            valence=0
        )
        e.slice_to_faces('u', 'interior')
        e.slice_to_faces('grad_u', 'interior')
    for e in domain.elements:
        for face in e.get_interior_faces():
            sigma = penalty(face, penalty_parameter)
            basis_avg_factor = 1 if face.is_in('external') else 0.5
            e.Au -= quadrature(
                face,
                np.reshape(face.jump('u'), face.u.shape + e.dim * (1,)) * (basis_avg_factor * np.einsum('d...,d...', np.reshape(
                    face.get_normal(), (e.dim, *face.num_points) + e.dim * (1,)), basis_deriv(e, face)) - sigma * basis(e, face)),
                valence=0
            )
            e.Au -= lift_flux(
                face.normal_dot(face.average('grad_u')),
                face
            )
    return domain.get_data('Au')


def compute_weak_primal_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    for e in domain.elements:
        e.b = quadrature(
            e,
            np.reshape(getattr(e, source_field),
                       e.num_points + e.dim * (1,)) * basis(e),
            valence=0
        )
        for face in e.get_external_faces():
            sigma = penalty(face, penalty_parameter)
            g = np.take(getattr(e, dirichlet_boundary_field),
                        face.slice_index(), axis=face.dimension)
            e.b -= quadrature(
                face,
                np.reshape(g, g.shape + e.dim * (1,)) * (1 * np.einsum('d...,d...', np.reshape(face.get_normal(
                ), (e.dim, *face.num_points) + e.dim * (1,)), basis_deriv(e, face)) - sigma * basis(e, face)),
                valence=0
            )
    return domain.get_data('b')


def apply_massless_weak_primal_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, 'u', fields_valence=(0,))
    for e in domain.elements:
        e.grad_u = compute_deriv(e.u, e)
        e.Au = compute_inverse_mass(quadrature(
            e,
            np.einsum('d...,d...', np.reshape(
                e.grad_u, e.grad_u.shape + e.dim * (1,)), basis_deriv(e)),
            valence=0
        ), e)
        e.slice_to_faces('u', 'interior')
        e.slice_to_faces('grad_u', 'interior')
    for e in domain.elements:
        for face in e.get_interior_faces():
            sigma = penalty(face, penalty_parameter)
            basis_avg_factor = 1 if face.is_in('external') else 0.5
            e.Au -= compute_inverse_mass(quadrature(
                face,
                np.reshape(face.jump('u'), face.u.shape + e.dim * (1,)) * (basis_avg_factor * np.einsum('d...,d...', np.reshape(
                    face.get_normal(), (e.dim, *face.num_points) + e.dim * (1,)), basis_deriv(e, face)) - sigma * basis(e, face)),
                valence=0
            ), e)
            e.Au -= compute_inverse_mass(lift_flux(
                face.normal_dot(face.average('grad_u')),
                face
            ), e)
    return domain.get_data('Au')


def compute_massless_weak_primal_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    for e in domain.elements:
        e.b = getattr(e, source_field)
        for face in e.get_external_faces():
            sigma = penalty(face, penalty_parameter)
            g = np.take(getattr(e, dirichlet_boundary_field),
                        face.slice_index(), axis=face.dimension)
            e.b -= compute_inverse_mass(quadrature(
                face,
                np.reshape(g, g.shape + e.dim * (1,)) * (1 * np.einsum('d...,d...', np.reshape(face.get_normal(
                ), (e.dim, *face.num_points) + e.dim * (1,)), basis_deriv(e, face)) - sigma * basis(e, face)),
                valence=0
            ), e)
    return domain.get_data('b')


def apply_first_order_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, ['u', 'v'], fields_valence=(0, 1))
    for e in domain.elements:
        e.Au = -compute_div(e.v, e)
        e.grad_u = compute_deriv(e.u, e)
        e.Av = -e.grad_u + e.v
        e.slice_to_faces('u')
        e.slice_to_faces('v', 'interior')
        e.slice_to_faces('grad_u', 'interior')
    for e in domain.elements:
        for face in e.get_interior_faces():
            sigma = penalty(face, penalty_parameter)
            e.Au -= compute_inverse_mass(lift_flux(
                face.normal_dot(face.average('grad_u') -
                                face.v) - sigma * face.jump('u'),
                face
            ), e)
            e.Av -= compute_inverse_mass(lift_flux(
                face.normal_times(face.average('u') - face.u),
                face
            ), e)
    return domain.get_data(['Au', 'Av'])


def compute_first_order_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    domain.set_data(domain.get_data(source_field), 'b_u')
    domain.set_data(lambda x: np.zeros(x.shape), 'b_v')
    for e in domain.elements:
        e.slice_to_faces(dirichlet_boundary_field, 'external')
        for face in e.get_external_faces():
            g = getattr(face, dirichlet_boundary_field)
            sigma = penalty(face, penalty_parameter)
            e.b_u += compute_inverse_mass(lift_flux(
                2 * sigma * g,
                face
            ), e)
            e.b_v += compute_inverse_mass(lift_flux(
                face.normal_times(g),
                face
            ), e)
    return domain.get_data(['b_u', 'b_v'])


def apply_massive_first_order_poisson_operator(x, domain, penalty_parameter):
    domain.set_data(x, ['u', 'v'], fields_valence=(0, 1))
    for e in domain.elements:
        e.Au = -compute_mass(compute_div(e.v, e), e)
        e.grad_u = compute_deriv(e.u, e)
        e.Av = compute_mass(-e.grad_u + e.v, e)
        e.slice_to_faces('u')
        e.slice_to_faces('v', 'interior')
        e.slice_to_faces('grad_u', 'interior')
    for e in domain.elements:
        for face in e.get_interior_faces():
            sigma = penalty(face, penalty_parameter)
            e.Au -= lift_flux(
                face.normal_dot(face.average('grad_u') -
                                face.v) - sigma * face.jump('u'),
                face
            )
            e.Av -= lift_flux(
                face.normal_times(face.average('u') - face.u),
                face
            )
    return domain.get_data(['Au', 'Av'])


def compute_massive_first_order_poisson_source(source_field, domain, dirichlet_boundary_field, penalty_parameter):
    domain.set_data(domain.get_data(source_field), 'b_u')
    domain.set_data(lambda x: np.zeros(x.shape), 'b_v')
    for e in domain.elements:
        e.slice_to_faces(dirichlet_boundary_field, 'external')
        for face in e.get_external_faces():
            g = getattr(face, dirichlet_boundary_field)
            sigma = penalty(face, penalty_parameter)
            e.b_u += lift_flux(
                2 * sigma * g,
                face
            )
            e.b_v += lift_flux(
                face.normal_times(g),
                face
            )
    return domain.get_data(['b_u', 'b_v'])


class PoissonOperator(LinearOperator):
    supported_schemes = {'first_order', 'massive_first_order', 'primal', 'massless_primal', 'weak_primal', 'massless_weak_primal'}

    def __init__(self, domain, scheme='primal', penalty_parameter=1.5):
        if scheme not in self.supported_schemes:
            raise NotImplementedError(
                "Scheme '{}' is not supported.".format(scheme))
        self.scheme = scheme

        self.domain = domain
        self.penalty_parameter = penalty_parameter

        if scheme in ['primal', 'massless_primal', 'weak_primal', 'massless_weak_primal']:
            self.field_valences = (0,)
            self.field_names = ['u']
        elif scheme in ['first_order', 'massive_first_order']:
            self.field_valences = (0, 1)
            self.field_names = ['u', 'v']
        else:
            raise NotImplementedError

        N = domain.get_total_num_points() * np.sum(domain.dim**np.array(self.field_valences))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, x):
        if self.scheme == 'primal':
            return apply_primal_poisson_operator(x, self.domain, self.penalty_parameter)
        elif self.scheme == 'massless_primal':
            return apply_massless_primal_poisson_operator(x, self.domain, self.penalty_parameter)
        elif self.scheme == 'first_order':
            return apply_first_order_poisson_operator(x, self.domain, self.penalty_parameter)
        elif self.scheme == 'massive_first_order':
            return apply_massive_first_order_poisson_operator(x, self.domain, self.penalty_parameter)
        elif self.scheme == 'weak_primal':
            return apply_weak_primal_poisson_operator(x, self.domain, self.penalty_parameter)
        elif self.scheme == 'massless_weak_primal':
            return apply_massless_weak_primal_poisson_operator(x, self.domain, self.penalty_parameter)
        else:
            raise NotImplementedError

    def compute_source(self, source_field, dirichlet_boundary_field):
        if self.scheme == 'primal':
            return compute_primal_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        elif self.scheme == 'massless_primal':
            return compute_massless_primal_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        elif self.scheme == 'first_order':
            return compute_first_order_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        elif self.scheme == 'massive_first_order':
            return compute_massive_first_order_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        elif self.scheme == 'weak_primal':
            return compute_weak_primal_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        elif self.scheme == 'massless_weak_primal':
            return compute_massless_weak_primal_poisson_source(source_field, self.domain, dirichlet_boundary_field, self.penalty_parameter)
        else:
            raise NotImplementedError
