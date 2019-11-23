import unittest

import dgpy.domain as dg_domain
import dgpy.poisson as dg_poisson
from dgpy.utilities import build_matrix

import numpy as np
import numpy.testing as npt
import os


class TestPoisson(unittest.TestCase):

    def test_primal_operator_properties(self):
        for d in range(1, 3 + 1):
            domain = dg_domain.Domain(
                extents=d * [(0, np.pi)], num_elements=2, num_points=3)
            for scheme in ['primal', 'weak_primal']:
                A = dg_poisson.PoissonOperator(domain, scheme=scheme)
                A_matrix = build_matrix(A)

                npt.assert_allclose(
                    A_matrix, A_matrix.T, err_msg="Operator matrix is not symmetric for {}D {} scheme".format(d, scheme))
                npt.assert_array_less(np.zeros(A.shape[0]), np.linalg.eigvals(
                    A_matrix), err_msg="Operator matrix is not positive definite for {}D {} scheme".format(d, scheme))

    def test_weak_scheme_matches_d4est(self):
        domain = dg_domain.Domain(extents=2 * [(0, np.pi)], num_elements=2, num_points=3)
        # Make sure IP flux uses sigma = C * poly_deg^2 / h (as opposed to num_points^2)
        A_matrix = build_matrix(dg_poisson.PoissonOperator(
            domain, scheme='weak_primal', penalty_parameter=1.5))
        A_matrix_d4est = np.loadtxt(os.path.join(
            os.path.dirname(__file__), 'poisson_matrix_d4est.txt'))
        npt.assert_allclose(A_matrix, A_matrix_d4est)
