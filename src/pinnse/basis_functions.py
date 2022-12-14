import numpy as np
import tensorflow as tf
from math import pi


class FourierBasis(object):
    def __init__(
        self,
        max_k,
        input_dimension=1,
        output_dimension=1,
        backend="tensorflow",
        axis="all",
        rescale=True,
    ):
        super().__init__()
        """
        dimension in [1, 2]
        backend in ['tensorflow', 'numpy']
        """
        assert axis in ["all"] + list(np.arange(input_dimension))
        assert input_dimension in [
            1,
            2,
            3,
        ], "Only 1,2 and 3 dimensional Fourier Basis implemented!"
        assert backend in [
            "tensorflow",
            "numpy",
        ], "Only tensorflow and numpy backends are implemented!"
        self.axis = axis
        self.max_k = max_k
        self.dimension = input_dimension
        self.output_dimension = output_dimension
        self.backend = backend
        self.rescale = rescale

    def compute(self, input):
        if self.axis == "all":
            assert self.dimension in [
                1,
                2,
            ], "Only 1,2 dimensional Fourier Basis for all axes implemented!"
            if self.backend == "tensorflow":
                # Rescaling step
                if self.rescale:
                    input = 2 * pi * (input / tf.reduce_max(input))
                ks = tf.range(1, self.max_k + 1, dtype=input.dtype)
                if self.dimension == 1:
                    output = tf.reduce_sum(
                        tf.sin(ks * input) + tf.cos(ks * input), axis=1
                    )
                    output = tf.reshape(output, (-1, 1))  # no weights assigned
                    return tf.repeat(output, self.output_dimension, axis=1)
                elif self.dimension == 2:
                    x, y = input[:, 0:1], input[:, 1:2]
                    c1 = tf.sin(ks * x) * tf.sin(ks * y)
                    c2 = tf.sin(ks * x) * tf.cos(ks * y)
                    c3 = tf.cos(ks * x) * tf.sin(ks * y)
                    c4 = tf.cos(ks * x) * tf.cos(ks * y)
                    output = tf.reduce_sum(c1 + c2 + c3 + c4, axis=1)
                    output = tf.reshape(output, (-1, 1))  # no weights assigned
                    return tf.repeat(output, self.output_dimension, axis=1)
            elif self.backend == "numpy":
                # TODO: implement numpy backend
                return 0
        else:
            if self.backend == "tensorflow":
                # Rescaling step
                if self.rescale:
                    input = 2 * pi * (input / tf.reduce_max(input))
                ks = tf.range(1, self.max_k + 1, dtype=input.dtype)
                if self.dimension == 1:
                    output = tf.reduce_sum(
                        tf.sin(ks * input) + tf.cos(ks * input), axis=1
                    )
                    output = tf.reshape(output, (-1, 1))  # no weights assigned
                    return tf.repeat(output, self.output_dimension, axis=1)
                elif self.dimension != 1:
                    variable = input[:, self.axis : self.axis + 1]
                    output = tf.reduce_sum(
                        tf.sin(ks * variable) + tf.cos(ks * variable), axis=1
                    )
                    output = tf.reshape(output, (-1, 1))  # no weights assigned
                    return tf.repeat(output, self.output_dimension, axis=1)
            elif self.backend == "numpy":
                # TODO: implement numpy backend
                return 0


class LegendreBasis(object):
    def __init__(self, max_n, dimension=1, backend="tensorflow"):
        super().__init__()
        self.max_n = max_n
        self.dimension = dimension
        self.backend = backend

    def _legendrePolynomial(self, input, n):

        # Rescaling step
        input = input / tf.reduce_max(input)

        k = tf.range(n + 1, dtype=input.dtype)
        n_tf = tf.constant(n, dtype=input.dtype)
        combination = tf.exp(tf.math.lgamma(n_tf + 1)) / (
            tf.exp(tf.math.lgamma(k + 1)) * tf.exp(tf.math.lgamma(n_tf - k + 1))
        )
        # factorial is computed using: n! = tf.exp( tf.math.lgamma( n+1 ) )
        elements = combination**2 * (input - 1) ** (n_tf - k) * (input + 1) ** (k)
        polynomial = 1 / (2**n) * tf.reduce_sum(elements, axis=1)
        return polynomial

    def compute(self, input):
        if self.backend == "tensorflow":
            function = 0
            if self.dimension == 1:
                for n in range(1, self.max_n + 1):
                    function += self._legendrePolynomial(input, n)
                function = tf.reshape(function, (-1, 1))
                return function
            elif self.dimension == 2:
                # TODO: implement
                return 0
        elif self.backend == "numpy":
            # TODO: implement
            return 0


class LaguerreBasis(object):
    def __init__(self, max_n, dimension=1, backend="tensorflow"):
        super().__init__()
        self.max_n = max_n
        self.dimension = dimension
        self.backend = backend

    def _laguerrePolynomial(self, input, n):

        # Rescaling step: a fair max_n to use starts at 50
        input = input / tf.reduce_max(input)

        k = tf.range(n + 1, dtype=input.dtype)
        n_tf = tf.constant(n, dtype=input.dtype)
        combination = tf.exp(tf.math.lgamma(n_tf + 1)) / (
            tf.exp(tf.math.lgamma(k + 1)) * tf.exp(tf.math.lgamma(n_tf - k + 1))
        )
        # factorial is computed using: n! = tf.exp( tf.math.lgamma( n+1 ) )
        elements = (
            combination * (-1) ** k * input**k / (tf.exp(tf.math.lgamma(k + 1)))
        )
        polynomial = tf.reduce_sum(elements, axis=1)
        return polynomial

    def compute(self, input):
        if self.backend == "tensorflow":
            function = 0
            if self.dimension == 1:
                for n in range(1, self.max_n + 1):
                    function += self._laguerrePolynomial(input, n)
                function = tf.reshape(function, (-1, 1))
                # Rescaling step
                function = function / tf.reduce_max(function)
                return function
            elif self.dimension == 2:
                # TODO: implement
                return 0
        elif self.backend == "numpy":
            # TODO: implement
            return 0
