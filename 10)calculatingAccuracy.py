import numpy as np

softmaxOutputs = np.array([[0.7, 0.2, 0.1],
                           [0.5, 0.1, 0.4],
                           [0.02, 0.9, 0.08]])
classTarget = np.array([0, 1, 1])

prediction = np.argmax(softmaxOutputs, axis=1)

if len(classTarget.shape) == 2:
    classTarget = np.argmax(classTarget, axis=1)

accuracy = np.mean(prediction == classTarget)

print(f'accuracy: {accuracy}')
