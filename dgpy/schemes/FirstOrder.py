import numpy as np
from scipy.sparse.linalg import LinearOperator

from dgpy.operators import (compute_div, compute_mass, penalty, lift_flux,
                            lift_deriv_flux)


def apply_first_order_operator(x,
                               domain,
                               system,
                               boundary_conditions,
                               formulation,
                               scheme,
                               numerical_flux,
                               penalty_parameter,
                               lifting_scheme,
                               mass_lumping,
                               massive,
                               storage_order,
                               use_nonlinear_boundary_conditions=False):
    if formulation == 'flux-full':
        domain.set_data(x, ['v', 'u'],
                        fields_valence=tuple(i + 1 for i in system.field_valences) + system.field_valences,
                        storage_order=storage_order)
    else:
        domain.set_data(x,
                        'u',
                        fields_valence=system.field_valences,
                        storage_order=storage_order)
    if formulation == 'primal':
        raise NotImplementedError(
            "The primal formulation is not yet correctly implemented")
    scheme_u = 'strong' if scheme in ['strong', 'strong-weak'] else 'weak'
    scheme_v = 'strong' if scheme in ['strong', 'weak-strong'] else 'weak'
    assert not (scheme_v == 'weak' and numerical_flux == 'ip'), (
        "Use the strong form for the auxiliary equation with the "
        "IP numerical flux.")
    # Compute the auxiliary fields
    domain.apply(system.auxiliary_fluxes, 'u', 'F_v',
                 domain.dim)  # Essentially u
    domain.apply(
        compute_div,
        'F_v',
        'divF_v',  # Essentially grad(u)
        scheme=scheme_v,
        massive=False,
        mass_lumping=mass_lumping)
    for element in domain.elements:
        element.v_numeric = (
            element.divF_v -
            system.auxiliary_sources(element.u, element, domain.dim))
        if formulation == 'flux-full':
            element.Av = element.v_numeric - element.v
        else:
            element.v = element.v_numeric  # Essentially nabla(u)
    # --- Communication begin ---
    for element in domain.elements:
        # For v:
        element.slice_to_faces('F_v', 'interior')
        for face in element.get_interior_faces():
            face.nF_v = face.normal_dot(face.F_v)
        # For u:
        if numerical_flux == 'ip': # TODO: Fix for LLF flux
            # This is essentially F_u but without v's boundary corrections. We could
            # also use F_u for the numerical flux (LLF not IP), but would have to
            # add a second communication step after v's boundary corrections have
            # been applied.
            element.slice_to_faces('v_numeric', 'interior')
            for face in element.get_interior_faces():
                face.nF_u_divF_v = face.normal_dot(
                    system.primal_fluxes(face.v_numeric, element))
    # --- Communication end ---
    # Boundary conditions for v
    for element in domain.elements:
        element.slice_to_faces('u', 'exterior')
        for face in element.get_exterior_faces():
            bc = boundary_conditions[face.dimension][
                (face.opposite_face.direction + 1) // 2]
            bc = bc.nonlinear if use_nonlinear_boundary_conditions else bc.linear
            face.u_b = bc(face.u, np.zeros(face.u.shape), face)[0]
            face.nF_v = (2. * face.normal_dot(
                system.auxiliary_fluxes(face.u_b, face, domain.dim)) +
                         face.opposite_face.nF_v)
    # Add boundary correction to v. The primal formulation handles the
    # contribution later.
    if formulation != 'primal':
        for element in domain.elements:
            for face in element.get_interior_faces():
                boundary_correction_v = 0.5 * (face.nF_v -
                                               face.opposite_face.nF_v)
                if scheme_v == 'strong':
                    boundary_correction_v -= face.nF_v
                lifted_boundary_correction_v = lift_flux(
                    boundary_correction_v,
                    face,
                    scheme=lifting_scheme,
                    massive=False,
                    mass_lumping=mass_lumping)
                if formulation == 'flux-full':
                    element.Av += lifted_boundary_correction_v
                else:
                    element.v += lifted_boundary_correction_v
    # Compute the primal equation
    domain.apply(system.primal_fluxes, 'v', 'F_u')
    domain.apply(compute_div, 'F_u', 'divF_u', scheme_u, massive, mass_lumping)
    domain.apply(system.primal_sources, 'v', 'S_u')
    for element in domain.elements:
        if massive:
            element.S_u = compute_mass(element.S_u, element, mass_lumping)
        element.Au = element.S_u - element.divF_u
    # Boundary conditions for primal equation
    for element in domain.elements:
        # The LLF flux needs an extra communication here because the nF_u are
        # needed for the flux.
        element.slice_to_faces('F_u', 'interior')
        for face in element.get_interior_faces():
            face.nF_u = face.normal_dot(face.F_u)
        for face in element.get_exterior_faces():
            bc = boundary_conditions[face.dimension][
                (face.opposite_face.direction + 1) // 2]
            bc = bc.nonlinear if use_nonlinear_boundary_conditions else bc.linear
            face.nF_u_divF_v = np.copy(
                bc(face.u, face.opposite_face.nF_u_divF_v,
                   face.opposite_face)[1])
            face.nF_u_divF_v *= -2.
            face.nF_u_divF_v += face.opposite_face.nF_u_divF_v
    # Add boundary correction to primal equation
    for element in domain.elements:
        for face in element.get_interior_faces():
            if formulation == 'primal':
                boundary_correction_v = 0.5 * (face.nF_v -
                                               face.opposite_face.nF_v)
                if scheme_v == 'strong':
                    boundary_correction_v -= face.nF_v
                boundary_correction_v = system.primal_fluxes(
                    boundary_correction_v, face)
                lifted_boundary_correction_v = lift_deriv_flux(
                    boundary_correction_v,
                    face,
                    scheme=lifting_scheme,
                    massive=massive,
                    mass_lumping=mass_lumping)
                element.Au += lifted_boundary_correction_v
            sigma = {
                'ip': penalty(face, penalty_parameter),
                'llf': penalty_parameter
            }[numerical_flux]
            boundary_correction_Au = 0.5 * (face.nF_u_divF_v -
                                            face.opposite_face.nF_u_divF_v)
            if scheme_u == 'strong':
                # Using the nF_u_divF_v here for the IP flux seems wrong but
                # seems to work just as well as using nF_u, which we have to
                # compute just for this. Q: Does it?
                boundary_correction_Au -= face.nF_u_divF_v
            boundary_correction_Au -= (
                sigma * face.normal_dot(face.nF_v + face.opposite_face.nF_v))
            element.Au -= lift_flux(boundary_correction_Au,
                                    face,
                                    scheme=lifting_scheme,
                                    massive=massive,
                                    mass_lumping=mass_lumping)
    if formulation == 'flux-full':
        if massive:
            for element in domain.elements:
                element.Av = compute_mass(element.Av,
                                          element,
                                          mass_lumping=mass_lumping)
        return domain.get_data(['Av', 'Au'], storage_order=storage_order)
    else:
        return domain.get_data('Au', storage_order=storage_order)


def compute_first_order_source(source_field, domain, system,
                               boundary_conditions, formulation, scheme,
                               numerical_flux, penalty_parameter,
                               lifting_scheme, mass_lumping, massive,
                               storage_order):
    if formulation == 'flux-full':
        field_valences = (tuple(i + 1 for i in system.field_valences) + system.field_valences)
    else:
        field_valences = system.field_valences
    N = domain.get_total_num_points() * np.sum(domain.dim**
                                               np.array(field_valences))
    domain.set_data(apply_first_order_operator(
        np.zeros(N), domain, system, boundary_conditions, formulation, scheme,
        numerical_flux, penalty_parameter, lifting_scheme, mass_lumping,
        massive, storage_order, use_nonlinear_boundary_conditions=True),
                    ['A_v0', 'A_u0'] if formulation == 'flux-full' else 'A_u0',
                    fields_valence=field_valences)
    for element in domain.elements:
        element.b_u = np.copy(getattr(element, source_field))
        if massive:
            element.b_u = compute_mass(element.b_u,
                                       element,
                                       mass_lumping=mass_lumping)
        element.b_u -= element.A_u0
        if formulation == 'flux-full':
            element.b_v = -element.A_v0
    if formulation == 'flux-full':
        return domain.get_data(['b_v', 'b_u'], storage_order=storage_order)
    else:
        return domain.get_data('b_u', storage_order=storage_order)


class DgOperator(LinearOperator):
    def __init__(self,
                 domain,
                 system,
                 boundary_conditions,
                 formulation='flux',
                 scheme='strong',
                 numerical_flux='ip',
                 penalty_parameter=1.5,
                 lifting_scheme='mass_matrix',
                 massive=True,
                 mass_lumping=False,
                 storage_order='F'):
        self.domain = domain
        self.system = system
        self.boundary_conditions = boundary_conditions
        self.formulation = formulation
        self.scheme = scheme
        self.numerical_flux = numerical_flux
        self.penalty_parameter = penalty_parameter
        self.lifting_scheme = lifting_scheme
        self.mass_lumping = mass_lumping
        self.massive = massive
        self.storage_order = storage_order
        if formulation == 'flux-full':
            field_valences = (system.field_valences +
                              tuple(i + 1 for i in system.field_valences))
        else:
            field_valences = system.field_valences
        N = domain.get_total_num_points() * np.sum(domain.dim**
                                                   np.array(field_valences))
        super().__init__(shape=(N, N), dtype=float)

    def _matvec(self, x):
        return apply_first_order_operator(
            x, self.domain, self.system, self.boundary_conditions,
            self.formulation, self.scheme, self.numerical_flux,
            self.penalty_parameter, self.lifting_scheme, self.mass_lumping,
            self.massive, self.storage_order)

    def compute_source(self, source_field):
        return compute_first_order_source(
            source_field, self.domain, self.system,
            self.boundary_conditions, self.formulation, self.scheme,
            self.numerical_flux, self.penalty_parameter, self.lifting_scheme,
            self.mass_lumping, self.massive, self.storage_order)
