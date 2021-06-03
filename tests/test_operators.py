from dgpy.spectral import inertial_coords, lgl_points, logical_coords
import unittest

import dgpy.domain as dg_domain
import dgpy.operators as dg_operators
from dgpy.interpolate import lagrange_polynomial

import numpy as np
import numpy.testing as npt
import scipy.integrate


class TestOperators(unittest.TestCase):

    def setUp(self):
        self.eps = np.finfo(float).eps

    def test_mass_matrix(self):
        """Tests that the mass matrix is M_ij = integrate(l_i * l_j, -1, 1)"""

        domain = dg_domain.Domain(
            extents=[(-1, 1)], num_elements=1, num_points=3)
        e = list(domain.elements)[0]
        M = dg_operators.mass_matrix(e, 0, mass_lumping=False)

        def m(x, i, j):
            return lagrange_polynomial(e.collocation_points[0], i)(x) * lagrange_polynomial(e.collocation_points[0], j)(x)
        M_expected = np.zeros((e.num_points[0], e.num_points[0]))
        for i in range(e.num_points[0]):
            for j in range(e.num_points[0]):
                M_expected[i, j] = scipy.integrate.quad(
                    m, -1, 1, args=(i, j))[0]

        npt.assert_allclose(M, M_expected, rtol=100 * self.eps)

    def test_interpolate_1d(self):
        domain = dg_domain.Domain(
            extents=[(-1, 1)], num_elements=1, num_points=3)
        e = list(domain.elements)[0]
        x = e.inertial_coords[0]
        u = x**2 + 2. * x + 3.
        target_points = lgl_points(4)
        u_interp = dg_operators.interpolate_to(u, e, [target_points])
        x_target = inertial_coords([target_points], e.extents)[0]
        u_expected = x_target**2 + 2. * x_target + 3.
        npt.assert_allclose(u_interp, u_expected)

    def test_interpolate_2d(self):
        domain = dg_domain.Domain(
            extents=2 * [(-1, 1)], num_elements=1, num_points=3)
        e = list(domain.elements)[0]
        x, y = e.inertial_coords
        u = x**2 * y**2 + 2. * x + 3. * y + 4.
        e_target = dg_domain.Element(e.extents,
                                     num_points=[4, 4],
                                     quadrature=e.quadrature)
        u_interp = dg_operators.interpolate_to(u, e,
                                               e_target.collocation_points)
        x_target, y_target = e_target.inertial_coords
        u_expected = x_target**2 * y_target**2 + 2. * x_target + 3. * y_target + 4.
        npt.assert_allclose(u_interp, u_expected)

if __name__ == '__main__':
    unittest.main()
