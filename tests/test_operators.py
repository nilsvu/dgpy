import unittest

import dgpy.domain as dg_domain
import dgpy.operators as dg_operators
from dgpy.interpolate import lagrange_polynomial

import numpy as np
import numpy.testing as npt
import scipy.integrate


class TestOperators(unittest.TestCase):

    def setUp(self):
        self.eps = np.finfo(np.float64).eps

    def test_mass_matrix(self):
        """Tests that the mass matrix is M_ij = integrate(l_i * l_j, -1, 1)"""

        domain = dg_domain.Domain(
            extents=[(-1, 1)], num_elements=1, num_points=3)
        e = list(domain.elements)[0]
        M = dg_operators.mass_matrix(e, 0)

        def m(x, i, j):
            return lagrange_polynomial(e.collocation_points[0], i)(x) * lagrange_polynomial(e.collocation_points[0], j)(x)
        M_expected = np.zeros((e.num_points[0], e.num_points[0]))
        for i in range(e.num_points[0]):
            for j in range(e.num_points[0]):
                M_expected[i, j] = scipy.integrate.quad(
                    m, -1, 1, args=(i, j))[0]

        npt.assert_allclose(M, M_expected, rtol=100 * self.eps)


if __name__ == '__main__':
    unittest.main()
