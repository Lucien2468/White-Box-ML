from white_box_ml.nn.layer.dense import Dense
from white_box_ml.nn.activations.activations import ReLU
from reversegrad import Tensor
import numpy as np

layer = Dense(3, 2)
x = Tensor(np.random.randn(1, 3))
out = layer.forward(x, activation=ReLU())
loss = out.mean()
loss.backward()
print('ok', out.data.shape, layer.weights.data.shape)
