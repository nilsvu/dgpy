import unittest

from scipy.integrate.quadpack import quad

import dgpy.domain as dg_domain

import numpy as np
import numpy.testing as npt


class TestOperators(unittest.TestCase):

    def test_create_domain_1d(self):
        domain = dg_domain.Domain(
            extents=[(0, 2)], num_elements=2, num_points=3)
        self.assertEqual(domain.dim, 1)
        self.assertEqual(len(domain.elements), 2)
        self.assertEqual(domain.get_total_num_points(), 6)

    def test_create_domain_2d(self):
        domain = dg_domain.Domain(
            extents=[(0, 2), (-1, 3)], num_elements=2, num_points=3)
        self.assertEqual(domain.dim, 2)
        self.assertEqual(len(domain.elements), 4)
        self.assertEqual(domain.get_total_num_points(), 36)

    def test_create_domain_3d(self):
        domain = dg_domain.Domain(
            extents=[(0, 2), (-1, 3), (0, 1)], num_elements=2, num_points=3)
        self.assertEqual(domain.dim, 3)
        self.assertEqual(len(domain.elements), 8)
        self.assertEqual(domain.get_total_num_points(), 216)

    def test_set_data(self):
        domain = dg_domain.Domain(
            extents=[(0, 1), (0, 2)], num_elements=2, num_points=3)

        def scalar_field(x, amplitude):
            return amplitude * np.sqrt(x[0]**2 + x[1]**2)

        def vector_field(x, amplitude):
            return amplitude * x

        domain.set_data(scalar_field, 'u', amplitude=2)
        npt.assert_almost_equal(
            domain.indexed_elements[(0, 0)].u[:, 0], [0, 0.5, 1])
        npt.assert_almost_equal(
            domain.indexed_elements[(0, 0)].u[0, :], [0, 1, 2])
        domain.set_data(vector_field, 'v', amplitude=2)
        npt.assert_almost_equal(
            domain.indexed_elements[(0, 0)].v[0, :, 0], [0, 0.5, 1])
        npt.assert_almost_equal(
            domain.indexed_elements[(0, 0)].v[1, 0, :], [0, 1, 2])

        domain.set_data(domain.get_data('u'), 'u2')
        npt.assert_equal(domain.get_data('u2'), domain.get_data('u'))
        domain.set_data(domain.get_data('v'), 'v2', 1)
        npt.assert_equal(domain.get_data('v2'), domain.get_data('v'))

        domain.set_data(domain.get_data(['u', 'v']), ['u3', 'v3'], [0, 1])
        npt.assert_equal(domain.get_data('u3'), domain.get_data('u'))
        npt.assert_equal(domain.get_data('v3'), domain.get_data('v'))

    def test_slice_to_faces_1d(self):
        def poly(coords):
            x = coords[0]
            return x**2 + 2. * x + 3.

        for quadrature in [
                dg_domain.Quadrature.GAUSS_LOBATTO, dg_domain.Quadrature.GAUSS
        ]:
            with self.subTest(quadrature=quadrature):
                domain = dg_domain.Domain(extents=[(0, 2)],
                                          num_elements=2,
                                          num_points=3,
                                          quadrature=quadrature)
                domain.set_data(poly, 'u')
                e = list(domain.elements)[0]
                e.slice_to_faces('u')
                face_left = e.indexed_faces[(0, -1, False)]
                npt.assert_allclose(face_left.u, poly([0.]))
                face_right = e.indexed_faces[(0, 1, False)]
                npt.assert_allclose(face_right.u, poly([1.]))


if __name__ == '__main__':
    unittest.main()
