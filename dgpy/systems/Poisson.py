import numpy as np

field_valences = (0,)


def primal_fluxes(v, e):
    return v


def auxiliary_fluxes(u, e, dim):
    return np.tensordot(np.identity(dim), u, axes=0)


def primal_sources(v, e):
    return np.zeros(v[0].shape)


def auxiliary_sources(u, e, dim):
    return np.zeros((dim,) + u.shape)
