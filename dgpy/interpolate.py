from .domain import *

from scipy.interpolate import lagrange as lagrange_interpolate


def lagrange_coeffs(f, xi, element):
    x = inertial_coords(xi, element)
    while x.ndim < 2:
        x = np.expand_dims(x, axis=0)
    return np.squeeze(f(*x))


def lagrange_polynomial(collocation_points, i):
    return lagrange_interpolate(collocation_points, np.eye(len(collocation_points))[:, i])
