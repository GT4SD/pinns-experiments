from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde import backend as bkd
from deepxde.utils import get_num_args


class CustomPDE(dde.data.PDE):
    def __init__(
        self,
        geometry,
        pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        train_distribution="Sobol",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
        pde_extra_arguments=None,
    ):  # pde_extra_arguments is to be given in dictionary (kwargs) format
        super().__init__(
            geometry,
            pde,
            bcs,
            num_domain,
            num_boundary,
            train_distribution,
            anchors,
            exclusions,
            solution,
            num_test,
            auxiliary_var_function,
        )
        self.pde_extra_arguments = pde_extra_arguments

    def losses(self, targets, outputs, loss, model):
        f = []
        if (self.pde is not None) and (self.pde_extra_arguments is None):
            if get_num_args(self.pde) == 2:
                f = self.pde(model.net.inputs, outputs)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    raise ValueError("Auxiliary variable function not defined.")
                f = self.pde(model.net.inputs, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        elif (self.pde is not None) and (self.pde_extra_arguments is not None):
            f = self.pde(model.net.inputs, outputs, **self.pde_extra_arguments)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * (len(f) + len(self.bcs))
        elif len(loss) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss)
                )
            )

        bcs_start = np.cumsum([0] + self.num_bcs)
        error_f = [fi[bcs_start[-1] :] for fi in f]
        losses = [
            loss[i](bkd.zeros_like(error), error) for i, error in enumerate(error_f)
        ]
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(self.train_x, model.net.inputs, outputs, beg, end)
            losses.append(loss[len(error_f) + i](bkd.zeros_like(error), error))
        return losses


class CustomTimePDE(CustomPDE):
    """Time-dependent PDE solver.

    Args:
        num_initial (int): The number of training points sampled on the initial
            location.
    """

    def __init__(
        self,
        geometryxtime,
        pde,
        ic_bcs,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        train_distribution="Sobol",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
        pde_extra_arguments=None,  # pde_extra_arguments is to be given in dictionary (kwargs) format
    ):
        self.num_initial = num_initial
        super(CustomTimePDE, self).__init__(
            geometryxtime,
            pde,
            ic_bcs,
            num_domain,
            num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
            auxiliary_var_function=auxiliary_var_function,
            pde_extra_arguments=pde_extra_arguments,
        )

    def train_points(self):
        X = super(CustomTimePDE, self).train_points()
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(
                    self.num_initial, random=self.train_distribution
                )
            if self.exclusions is not None:

                def is_not_excluded(x):
                    return not np.any([np.allclose(x, y) for y in self.exclusions])

                tmp = np.array(list(filter(is_not_excluded, tmp)))
            X = np.vstack((tmp, X))
        return X
