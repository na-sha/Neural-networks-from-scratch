# generally cross entropy loss is used for softmax activation outputs
import math
import numpy as np

softmaxOutputs = [0.7, 0.1, 0.2]

targetOutput = [1, 0, 0]

# loss = -(math.log(softmaxOutputs[0])*targetOutput[0] +
#          math.log(softmaxOutputs[1])*targetOutput[1] +
#          math.log(softmaxOutputs[2])*targetOutput[2])
# print(loss)

loss = -(math.log(softmaxOutputs[0]))

print(loss)
softmaxOutputs = None
softmaxOutputs1 = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
classTarget = [0, 1, 1]
for targetIndex, distribution in zip(classTarget, softmaxOutputs1):
    print(distribution[targetIndex])

# numpy implementation
npSoftmaxOutputs = np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.5, 0.4],
                             [0.02, 0.9, 0.08]])
print(-np.log(npSoftmaxOutputs[range(len(npSoftmaxOutputs)), classTarget]))
