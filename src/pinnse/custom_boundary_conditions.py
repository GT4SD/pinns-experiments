"""Boundary conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deepxde import gradients as grad
from deepxde.icbcs.boundary_conditions import BC


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(
        self,
        geom,
        component_x,
        on_boundary,
        derivative_order=0,
        component=0,
        periodicity="symmetric",
    ):
        """
        periodicity:
        - symmetric for y[x] = y[-x]
        - antisymmetric for y[x] = -y[-x]
        """
        super(PeriodicBC, self).__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )
        assert periodicity in [
            "symmetric",
            "antisymmetric",
        ], "Given periodicity not understood!"
        self.periodicity = periodicity

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component : self.component + 1]
            yright = outputs[mid:end, self.component : self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        if self.periodicity == "symmetric":
            return yleft - yright  # y[x] = y[-x]
        elif self.periodicity == "antisymmetric":
            return yleft + yright  # y[x] = -y[-x]
