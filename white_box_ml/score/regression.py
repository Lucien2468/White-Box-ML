import numpy as np
from reversegrad import Tensor


class MAE:
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0

    def forward(self, predictions, labels):
        if not isinstance(labels, Tensor):
            labels = Tensor(labels)
        out = Tensor(np.mean(np.abs(predictions.data - labels.data)))

        def _backward():
            grad = np.sign(predictions.data - labels.data) / predictions.data.size
            predictions.grad += out.grad * grad
            labels.grad -= out.grad * grad

        out._backward = _backward
        out._children = [predictions, labels]
        return out


class MSE:
    def __init__(self):
        self.total_loss = 0.0
        self.count = 0

    def forward(self, predictions, labels):
        if not isinstance(labels, Tensor):
            labels = Tensor(labels)
        out = Tensor(np.mean((predictions.data - labels.data) ** 2))

        def _backward():
            grad = 2 * (predictions.data - labels.data) / predictions.data.size
            predictions.grad += out.grad * grad
            labels.grad -= out.grad * grad

        out._backward = _backward
        out._children = [predictions, labels]
        return out
