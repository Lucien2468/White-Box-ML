import numpy as np
from reversegrad import Tensor


class ReLU:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        out = Tensor(np.maximum(0, input.data))

        def _backward():
            input.grad += out.grad * (input.data > 0)

        out._backward = _backward
        out._children = [input]
        self.output = out
        return out

    def backward(self, output_gradient: Tensor) -> Tensor:
        grad_data = output_gradient.data * (self.input.data > 0)
        return Tensor(grad_data)


class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        s = 1 / (1 + np.exp(-input.data))
        out = Tensor(s)

        def _backward():
            input.grad += out.grad * (out.data * (1 - out.data))

        out._backward = _backward
        out._children = [input]
        self.output = out
        return out

    def backward(self, output_gradient: Tensor) -> Tensor:
        sigmoid_derivative = self.output.data * (1 - self.output.data)
        return Tensor(output_gradient.data * sigmoid_derivative)


class Tanh:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        t = np.tanh(input.data)
        out = Tensor(t)

        def _backward():
            input.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        out._children = [input]
        self.output = out
        return out

    def backward(self, output_gradient: Tensor) -> Tensor:
        tanh_derivative = 1 - self.output.data ** 2
        return Tensor(output_gradient.data * tanh_derivative)
