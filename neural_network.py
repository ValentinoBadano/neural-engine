from layer import *
from activation_layer import *
from cost_functions import quadratic_error, quadratic_error_derivative

class NeuralNetwork(): 
    def __init__(self, layers):
        self.layers = []
        for l in layers:
            self.addLayer(l)

    def addLayer(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        y = x
        for l in self.layers:
            y = l.propagate(y)
        return y;

    def train(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            for x, y in zip(x_train, y_train):
                output = x

                # prop. hacia adelante
                for l in self.layers:
                    output = l.propagate(output)

                error = quadratic_error_derivative(y, output)

                # prop. hacia atrás
                for l in reversed(self.layers):
                    error = l.retropropagate(error, learning_rate)