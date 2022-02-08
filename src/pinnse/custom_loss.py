import tensorflow as tf
import deepxde.backend as bkd
from pinnse.integrate import integrate_1d

# class RadiusBasedLoss:

#     def __init__(self, batch_size: int):
#         self.batch_size = batch_size
    
#     def set_weights(self, weights):
#         assert len(weights) == self.batch_size
#         self.weights = weights

#     def __call__(self, y_true, y_pred):
#         # compute MSE with weights
#         pass

class WeightedLoss:

    def __init__(self, condition):
        self.weights = None
        self.condition = condition

    def set_weights(self, input):
        weights = self.condition(input)
        self.weights = weights

    def __call__(self, y_true, y_pred):
        # compute MSE with weights
        # TODO: generalize for other losses
        assert self.weights != None, "You need to set weights before calculating the loss function!"
        return bkd.reduce_mean(self.weights * bkd.square(y_true - y_pred))


class NormalizationLoss:

    def __init__(self):
        self.input_domain = None # TODO: define integration domain for more accurate calculation
        # TODO: variable normalization constant
        self.normalization_constant = 1
    
    def set_variables(self, output, domain_area, n_points_domain, n_points_boundary):
        """
        Domain area cannot be calculated from geom automatically, due to the problem setup (Cuboid used instead of Sphere)
        """
        self.outputs = output
        self.n_points_boundary = n_points_boundary
        self.volume = domain_area / (n_points_domain - n_points_boundary)

    def __call__(self, y_true, y_pred):
        MSE = bkd.reduce_mean(bkd.square(y_true - y_pred))
        integral = bkd.reduce_mean(self.outputs[self.n_points_boundary:,0:1] * self.volume)
        return MSE + (integral - self.normalization_constant)**2