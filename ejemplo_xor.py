from layer import Layer
from activation_layer import ActivationLayer
from neural_network import NeuralNetwork
from activation_functions import tanh, tanh_derivative
import numpy as np

# datos XOR
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# definimos el modelo
nn = NeuralNetwork([Layer(2, 3), 
        ActivationLayer(tanh, tanh_derivative),
        Layer(3, 1),
        ActivationLayer(tanh, tanh_derivative)])

# hacemos una predicción inicial
y_pred = nn.predict(x_train)
print(y_pred)

# entrenamos al modelo
nn.train(x_train, y_train, epochs=400, learning_rate=0.1, graph=True)

# predicción final
y_pred = nn.predict(x_train)
print(y_pred)
