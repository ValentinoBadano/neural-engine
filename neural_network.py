from layer import *
from activation_layer import *
from cost_functions import quadratic_error, quadratic_error_derivative
import matplotlib.pyplot as plt 

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

    def train(self, x_train, y_train, epochs, learning_rate, verbose = True, graph = False):
        for i in range(epochs):
            error_promedio = 0
            for x, y in zip(x_train, y_train):
                output = x

                # prop. hacia adelante
                for l in self.layers:
                    output = l.propagate(output)

                error_promedio += quadratic_error(y, output)   # para registro
                error = quadratic_error_derivative(y, output) 

                # prop. hacia atrás
                for l in reversed(self.layers):
                    error = l.retropropagate(error, learning_rate)

            # imprime el error si el verbose está en true        
            if verbose:
                error_promedio /= len(y_train)
                print("Epoch:", i, "Loss:", error_promedio)

            if graph:
                plt.plot(i, error_promedio, 'ro', color='blue', markersize=0.2)
                
        if graph:
            plt.xlabel('Training')
            plt.ylabel('Loss')
            plt.show()