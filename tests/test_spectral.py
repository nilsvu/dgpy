import unittest

from dgpy import spectral
import numpy.testing as npt
from numpy import sqrt


class TestSpectral(unittest.TestCase):
    def test_lgl_points(self):
        npt.assert_allclose(spectral.lgl_points(2), [-1, 1])
        npt.assert_allclose(spectral.lgl_points(3), [-1, 0, 1])
        npt.assert_allclose(spectral.lgl_points(4),
                            [-1, -1 / sqrt(5), 1 / sqrt(5), 1])

    def test_lgl_weights(self):
        npt.assert_allclose(spectral.lgl_weights(2), [1, 1])
        npt.assert_allclose(spectral.lgl_weights(3), [1 / 3, 4 / 3, 1 / 3])
        npt.assert_allclose(spectral.lgl_weights(4),
                            [1 / 6, 5 / 6, 5 / 6, 1 / 6])

    def test_lg_points(self):
        npt.assert_allclose(spectral.lg_points(1), [0])
        npt.assert_allclose(spectral.lg_points(2), [-1 / sqrt(3), 1 / sqrt(3)])
        npt.assert_allclose(spectral.lg_points(3),
                            [-sqrt(3 / 5), 0, sqrt(3 / 5)])

    def test_lg_weights(self):
        npt.assert_allclose(spectral.lg_weights(1), [2])
        npt.assert_allclose(spectral.lg_weights(2), [1, 1])
        npt.assert_allclose(spectral.lg_weights(3), [5 / 9, 8 / 9, 5 / 9])
