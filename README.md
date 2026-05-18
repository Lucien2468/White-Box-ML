# White Box ML

White Box ML is a small educational neural network library built from scratch with NumPy.
It is designed to demonstrate how forward propagation, backpropagation, and
gradient descent work without the complexity of a full deep learning framework.

## What This Project Does

- Implements dense layers with learnable weights and biases
- Supports three activation functions: ReLU, Sigmoid, and Tanh
- Uses a simple automatic differentiation engine to compute gradients
- Includes a sequential model container for building layer stacks
- Supports mean squared error (MSE) training for regression-style problems

## Why It Exists

This library is intentionally simple so you can see the math and the execution
flow clearly:

- How values propagate through the model
- How gradients flow backward through the network
- How parameters are updated with gradient descent

It is a learning tool, not a production framework.

## How It Works

White Box ML is built on three core pillars:

1. **Automatic Differentiation (`Tensor`)**: Every operation performed on a `Tensor` object builds a dynamic computation graph. Each `Tensor` stores its data, its gradient, and a `_backward` function that defines how to propagate gradients to its parent tensors.

2. **Neural Network Layers**: Layers like `Dense` perform linear transformations ($Y = XW + B$). These transformations use `Tensor` operations, ensuring they are automatically tracked by the autograd engine.

3. **Backpropagation**: When `loss.backward()` is called, the engine traverses the computation graph in reverse topological order, executing each `_backward` function to populate the `.grad` attributes of all parameters (weights and biases).

4. **Manual Updates**: To keep things transparent, gradients are applied manually in the training loop using simple gradient descent.


```python
import numpy as np
from white_box_ml.nn.sequential import Sequential
from white_box_ml.nn.layer import Dense
from white_box_ml.nn.activations import ReLU, Sigmoid
from autograd import Tensor

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

model = Sequential()
model.add(Dense(2, 4), ReLU())
model.add(Dense(4, 1), Sigmoid())

learning_rate = 0.2
for epoch in range(1000):
    output = model.forward(X)
    y_t = Tensor(y)
    diff = output + y_t * Tensor(-1.0)
    loss = (diff * diff).mean()
    loss.backward()

    for layer, _ in model.layers:
        layer.weights.data -= learning_rate * layer.weights.grad
        layer.biases.data -= learning_rate * np.sum(layer.biases.grad, axis=0, keepdims=True)
        layer.weights.grad = 0
        layer.biases.grad = 0

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.6f}")
```

## Test Results on XOR data

```
Epoch 0, Loss: 0.250000
Epoch 100, Loss: 0.243794
Epoch 200, Loss: 0.191337
Epoch 300, Loss: 0.085639
Epoch 400, Loss: 0.052286
Epoch 500, Loss: 0.035741
Epoch 600, Loss: 0.027104
Epoch 700, Loss: 0.021627
Epoch 800, Loss: 0.017968
Epoch 900, Loss: 0.015282
```


## Project Structure

```
white_box_ml/
├── nn/
│   ├── activations/
│   │   └── activations.py
│   ├── layer/
│   │   └── dense.py
│   └── sequential/
│       └── sequential.py
├── score/
│   └── regression.py
└── __init__.py
```

## Limitations

- CPU only, no GPU support
- No batching or optimizers
- Not intended for large datasets or production workloads
- No advanced layers or training tricks like dropout or batch normalization

## Learning Goals

This project is useful for understanding:

- how neural networks compute outputs
- why activation functions are needed
- how backpropagation computes gradients
- how gradients are used to update weights

## License

MIT License
