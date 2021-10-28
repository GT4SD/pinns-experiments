import deepxde.backend as bkd

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
