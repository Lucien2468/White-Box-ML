import numpy as np
from reversegrad import Tensor
from white_box_ml.nn.activations import *


class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = Tensor(np.random.randn(input_size, output_size) * 0.1)
        self.biases = Tensor(np.zeros((1, output_size)))

    def forward(self, inputs, activation=None):
        self.inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        output = self.inputs @ self.weights + self.biases
        if activation:
            output = activation.forward(output)
        return output
