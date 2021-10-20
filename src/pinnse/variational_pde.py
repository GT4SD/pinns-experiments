from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deepxde.data.helper import one_function
from deepxde.data.pde import PDE, TimePDE
from deepxde import config
from deepxde.backend import tf
from deepxde.utils import run_if_all_none


class VPDE(PDE):
    """
    Variational PDE solver:
    only possible for GeometryXTime: due to the lack of uniform_points in other geometries
    """

    def __init__(  # denoted with ($) if existent in PDE
        self,
        geometry,  # ($) geom or geomtime (geomtime handled with TimePDE)
        pde,  # ($) pde to calculate the variational form
        bcs,  # ($) list of bcs
        integration_method="trapezoid",  # method for integration
        # quad_deg,
        # kernel=None,
        num_domain=0,  # ($)
        num_boundary=0,  # ($)
        # train_distribution="Sobol", #($) We should make this uniform
        train_distribution="uniform",
        test_function="sine",
        anchors=None,  # ($)
        solution=None,  # ($)
        num_test=None,  # ($)
    ):
        self.integration_method = integration_method
        if train_distribution != "uniform":
            raise ValueError(
                "Train distribution must be uniform in order to calculate the integral!"
            )

    # def train_points():

    def losses(self, targets, outputs, loss, model):
        f = []
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(model.net.inputs, outputs)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    raise ValueError("Auxiliary variable function not defined.")
                f = self.pde(model.net.inputs, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        # PDE value calculated (f) --> integrate

    # TODO: add new 'train_points': so we define a net like structure --> enable integration
    # We have geom.dim = spatial + temporal for GeometryXTime, so only steal from TimePDE for initial points
    # TODO: create new 'losses' --> also add 'integrate' to easily compute loss
    # I don't think that we need to change train_next_batch since we are changing train_points
    # no need to change test_points since already created with geom.uniform_points
    # no need to change test as well (I think)
    # no need to change resample_train_points, add_anchors, bc_points
    # uniform_points in GeometryXTime creates a grid!
    # net.input in model TrainState, which comes directly from data
