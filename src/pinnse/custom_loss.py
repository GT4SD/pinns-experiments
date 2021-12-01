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
        self.input_domain = None
    
    def set_domain(self, input):
        self.input_domain = input

    def __call__(self, y_true, y_pred):
        assert self.input_domain != None, "You need to set domain before calculating the loss function!"
        MSE = bkd.reduce_mean(bkd.square(y_true - y_pred))
        # TODO: generalize for arbitrary number of components
        axis1, indexes = tf.unique(self.input_domain[:,0])[0], tf.unique(self.input_domain[:,0])[1]
        sort_indexes = tf.argsort(axis1)
        axis1 = tf.sort(axis1)
        indexes = tf.gather(indexes, sort_indexes)
        axis1 = tf.reshape(axis1, (-1,1))
        y1, y2 = y_pred[:,0:1], y_pred[:,1:2]
        y1, y2 = tf.gather(y1, indexes), tf.gather(y2, indexes)
        integrand = y1**2 + y2**2
        integral = integrate_1d(axis1, integrand)
        print(self.input_domain)
        return MSE + (integral - 1)**2