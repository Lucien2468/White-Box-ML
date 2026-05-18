from white_box_ml.nn.activations import *
from white_box_ml.nn.layer import *


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer, activation=None):
        self.layers.append((layer, activation))

    def forward(self, inputs):
        for layer, activation in self.layers:
            inputs = layer.forward(inputs, activation)
        return inputs
