import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.output = None
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# creating dataset
X, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
dense1.forward(X)

print(dense1.output[:5])
