import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
# each input have their own weights but a neuron has only a single bias
bias = 2

# pure python implementation
output = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] +
          inputs[3] * weights[3] + bias)

print(output)

# numpy implementation
npOutput = np.dot(weights, inputs) + bias
print(npOutput)
