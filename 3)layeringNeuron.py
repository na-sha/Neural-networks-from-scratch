import numpy as np

inputs = [1, 2, 3, 2.5]
# neuron connects to different neuron with different biases
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# Output of current layer

# pure python implementation
layerOutputs = []
# For each neuron
for neuronWeights, neuronBias in zip(weights, biases):
    # zeroed output of given neuron
    neuronOutput = 0
    # For each input and weight to the neuron
    for nInput, weight in zip(inputs, neuronWeights):
        # multiply this input by associated weight and add to the neuron's output variable
        neuronOutput += nInput * weight
    # add bias
    neuronOutput += neuronBias
    # put neuron's result to the layer's output list
    layerOutputs.append(neuronOutput)

print(f'pure python implementation: {layerOutputs}')

# numpy implementation
npLayerOutput = np.dot(weights, inputs) + biases
print(f'numpy implementation:\t\t{npLayerOutput}')
