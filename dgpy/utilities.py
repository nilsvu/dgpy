import numpy as np

from .operators import quadrature


def build_matrix(operator):
    return np.transpose(np.array([operator @ np.eye(operator.shape[0])[i] for i in range(operator.shape[1])]))


def l2_error(domain, field, analytic_field, method='pointwise'):
    if method == 'quadrature':
        return np.sqrt(domain.reduce(
            lambda x, y: x + y,
            lambda e: quadrature(e, (getattr(
                e, field) - getattr(e, analytic_field))**2, getattr(e, field).ndim - e.dim),
            0
        ))
    elif method == 'pointwise':
        return np.sqrt(domain.reduce(
            lambda x, y: x + y,
            lambda e: np.sum((getattr(e, field) - getattr(e, analytic_field))**2, axis=tuple(
                range(getattr(e, field).ndim - e.dim, getattr(e, field).ndim - e.dim + e.dim))),
            0
        ) / domain.get_total_num_points())
    else:
        raise NotImplementedError


def symmetrize(u, axes=(0, 1)):
    assert len(axes) is 2
    return (u + np.moveaxis(u, axes, (axes[1], axes[0]))) / 2
