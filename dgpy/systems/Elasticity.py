import numpy as np
from numpy.lib.function_base import disp

field_valences = (1,)


class IsotropicHomogeneous:
    def __init__(self, bulk_modulus, shear_modulus):
        self.bulk_modulus = bulk_modulus
        self.shear_modulus = shear_modulus
        self.youngs_modulus = (9. * self.bulk_modulus * self.shear_modulus /
                               (3. * self.bulk_modulus + self.shear_modulus))
        self.poisson_ratio = (
            (3. * self.bulk_modulus - 2. * self.shear_modulus) /
            (6. * self.bulk_modulus + 2. * self.shear_modulus))
        self.lame_parameter = (self.bulk_modulus -
                               2. * self.shear_modulus / 3.)

    def stress(self, strain):
        dim = len(strain)
        strain_trace = np.trace(strain, axis1=0, axis2=1)
        krond_strain_trace = np.tensordot(np.identity(dim),
                                          strain_trace,
                                          axes=0)
        if dim == 3:
            return (-self.bulk_modulus * krond_strain_trace -
                    2. * self.shear_modulus *
                    (strain - krond_strain_trace / 3.))
        elif dim == 2:
            return -(2. * self.lame_parameter * self.shear_modulus /
                     (self.lame_parameter + 2. * self.shear_modulus) *
                     krond_strain_trace + 2. * self.shear_modulus * strain)
        else:
            raise NotImplementedError


def primal_fluxes(strain, e):
    return -IsotropicHomogeneous(
        bulk_modulus=79.36507936507935,
        shear_modulus=38.75968992248062).stress(strain)


def auxiliary_fluxes(displacement, e, dim):
    a = np.tensordot(np.identity(dim), displacement, axes=0)
    return 0.5 * (a + a.transpose(
        (0, 2, 1) + tuple(np.arange(3, 3 + displacement.ndim - 1))))


def primal_sources(strain, e):
    return np.zeros(strain[0].shape)


def auxiliary_sources(displacement, e, dim):
    return np.zeros((dim,) + displacement.shape)
