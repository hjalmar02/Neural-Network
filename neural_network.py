from math import exp
from mlxtend.data import loadlocal_mnist
from random import randint


def sigmoid(x):
    return 1 / (1 + exp(-x))


def cost(input, target):
    cost = 0
    for i in range(len(input)):
        cost += (input[i] - target[i]) ** 2
    return cost


class Node:

    def __init__(self, bias=None, value=0, tag=None):

        if bias:
            self.bias = bias
        else:
            self.bias = randint(-20, 20) / 10

        self.value = value
        self.tag = tag

    def get_value(self, inputs, weights):

        weighted_value = 0
        for i in range(len(inputs)):
            weighted_value += inputs[i].value * weights[i]
        self.value = sigmoid(weighted_value + self.bias)


class NeuralNetwork:

    def __init__(self, layers, input_shape, output_shape):

        self.layers = []
        # Constructing input layer
        input_layer = []
        for i in range(input_shape):
            input_layer.append(Node())
        self.layers.append(input_layer)

        # Constructing hidden layers
        for i in range(len(layers)):
            layer = []
            for _ in range(layers[i]):
                layer.append(Node())
            self.layers.append(layer)

        # Constructing output layer
        output_layer = []
        for _ in range(output_shape):
            output_layer.append(Node())
        self.layers.append(output_layer)

        # Constructing connections (weights) between layers
        self.weights = []
        for i in range(len(self.layers) - 1):
            layer_weights = []
            for _ in range(len(self.layers[i + 1])):
                node_weights = []
                for _ in range(len(self.layers[i])):
                    node_weights.append(randint(-20, 20) / 10)
                layer_weights.append(node_weights)

            self.weights.append(layer_weights)


    def output(self, input):

        # Assigning input values to input layer
        for i in range(len(input)):
            self.layers[0][i].value = input[i]

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].get_value(self.layers[i - 1], self.weights[i - 1][j])

        return [node.value for node in self.layers[len(self.layers) - 1]]



