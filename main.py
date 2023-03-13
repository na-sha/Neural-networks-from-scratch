import numpy as np
import nnfs
# from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()


class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.output = None
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabjlities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

    def forward(self, output, y):
        pass


class CrossEntropyLoss(Loss):
    def forward(self, yPredicted, yTrue):
        # correctConfidences = None
        samples = len(yPredicted)
        yPredictedClipped = np.clip(yPredicted, 1e-7, 1 - 1e-7)

        if len(yTrue.shape) == 1:
            correctConfidences = yPredictedClipped[
                range(samples),
                yTrue
            ]
        elif len(yTrue.shape) == 2:
            correctConfidences = np.sum(
                yPredictedClipped * yTrue,
                axis=1
            )
        negativeLogLosses = -np.log(correctConfidences)
        return negativeLogLosses


def main():
    # creating dataset
    # X, y = spiral_data(samples=100, classes=3)
    X, y = spiral_data(samples=100, classes=3)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    # plt.show()

    # create a dense layer with 2 input feature and 3 output values
    dense1 = DenseLayer(2, 3)
    # create relu activation to be used with the dense layer
    ReLUActivation = ActivationReLU()
    # create second dense layer with 3 inputs feature and 3 output values
    dense2 = DenseLayer(3, 3)
    # create softmax activation to be used with dense layer
    softmax = ActivationSoftmax()

    # loss function to calculate loss of how wrong our network is
    lossFunction = CrossEntropyLoss()

    lowestLoss = 9999999
    bestDense1Weights = dense1.weights.copy()
    bestDense1Biases = dense1.biases.copy()
    bestDense2Weights = dense2.weights.copy()
    bestDense2Biases = dense2.biases.copy()

    for iteration in range(10000):

        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        # make forward pass of out training data through this layer
        dense1.forward(X)
        # print(f'Dense layer 1 output: {dense1.output[:5]}')

        # the output of the dense1 layer is passed through ReLU activation function
        ReLUActivation.forward(dense1.output)
        # print(f'After Relu Activation: {ReLUActivation.output[:5]}')

        # makes forward pass through second dense layer with input of RelU activation's output
        dense2.forward(ReLUActivation.output)
        # print(f'Dense layer 2 output: {dense2.output}')

        # the output of second dense layer is passed through softmax activation function
        softmax.forward(dense2.output)
        # print(softmax.output[:5])

        loss = lossFunction.calculate(softmax.output, y)
        # print(f' loss {loss}')

        prediction = np.argmax(softmax.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(prediction == y)
        # print(f' accuracy: {accuracy}')

        if loss < lowestLoss:
            print(f'New sets of weight found, iteration: {iteration}, loss: {loss}, accuracy: {accuracy}')
            bestDense1Weights = dense1.weights.copy()
            bestDense1Biases = dense1.biases.copy()
            bestDense2Weights = dense2.weights.copy()
            bestDense2Biases = dense2.biases.copy()
            lowestLoss = loss
        else:
            dense1.weights = bestDense1Weights.copy()
            dense1.biases = bestDense1Biases.copy()
            dense2.weights = bestDense2Weights.copy()
            dense2.biases = bestDense2Biases.copy()


if __name__ == '__main__':
    main()
