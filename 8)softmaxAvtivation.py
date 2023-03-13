import numpy as np

layerOutputs = [4.8, 1.21, 2.385]

#######################
# python implementation
E = 2.71828182846
expValues = [E**output for output in layerOutputs]
# print(f' exponentiated values: {expValues}')
normValues = [value/sum(expValues) for value in expValues]
print(f'normalised values: {normValues}')
print(f' sum of normalised values: {sum(normValues)}')

######################
# numpy implementation
npExpValues = np.exp(layerOutputs)
# print(f' exponentiated values: {npExpValues}')
npNormValues = expValues / np.sum(expValues)
print(f'normalised values: {npNormValues}')
print(f' sum of normalised values: {sum(npNormValues)}')

# probabilities = npExpValues / np.sum(expValues, axis=1, keepdims=True)
# print(probabilities)
