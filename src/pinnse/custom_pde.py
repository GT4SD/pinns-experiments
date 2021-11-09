from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
# from deepxde.data.data import Data
from deepxde import backend as bkd
from deepxde import config
from deepxde.utils import get_num_args, run_if_all_none



class CustomPDE(dde.data.PDE):

    def __init__(self,
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
        pde_extra_arguments=None,): # pde_extra_arguments is to be given in dictionary (kwargs) format
        super().__init__(geometry,
            pde,
            bcs,
            num_domain,
            num_boundary,
            train_distribution,
            anchors,
            exclusions,
            solution,
            num_test,
            auxiliary_var_function,)
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
