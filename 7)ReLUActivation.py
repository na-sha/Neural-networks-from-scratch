import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

outputs = [max(0, i) for i in inputs]
print(outputs)

# numpy implementation
npOutputs = np.maximum(0, inputs)
print(npOutputs)


# class of ReLU Activation
class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

