from deepxde.nn.tensorflow_compat_v1.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde.nn import regularizers
from deepxde import config
from deepxde.backend import tf
from deepxde.utils import timing

class HNN(NN):
    """Highway neural network
    
    References:
    
    - https://arxiv.org/pdf/1505.00387.pdf

    """

    def __init__(
        self,
        input_size,
        output_size,
        num_neurons,
        num_blocks,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super(HNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_blocks = num_blocks
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.transform_bias_initializer = tf.keras.initializers.Constant(value=-2) # negative initialization for transform bias

    @property
    def inputs(self):
        return self.x

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.y_

    @timing
    def build(self):
        print("Building residual neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.input_size])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        y = self._dense(y, self.num_neurons, activation=self.activation)
        for _ in range(self.num_blocks):
            y = self._highway_block(y)
        self.y = self._dense(y, self.output_size)

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.output_size])
        self.built = True

    def _dense(self, inputs, units, activation=None, use_bias=True, bias_initializer=tf.zeros_initializer()):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer=self.regularizer,
        )

    def _highway_block(self, inputs):
        units = inputs.shape[1]

        x = self._dense(inputs, units, activation=self.activation)
        x = self._dense(x, units)

        T = self._dense(inputs, units, activation=tf.keras.activations.sigmoid,\
             bias_initializer=self.transform_bias_initializer)
        x = x*T + inputs*(1-T)

        return x