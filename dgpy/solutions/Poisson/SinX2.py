import numpy as np
from numpy import pi, sin, cos


class SinX2:
    def field(self, coords):
        assert len(coords) == 1
        x, = coords
        return sin(pi * x**2)

    def auxiliary_field(self, coords):
        assert len(coords) == 1
        x, = coords
        return np.array([2. * pi * x * cos(pi * x**2)])

    def source(self, coords):
        assert len(coords) == 1
        x, = coords
        return 4. * pi**2 * x**2 * sin(pi * x**2) - 2. * pi * cos(pi * x**2)
