from layer import *
from activation_layer import *
from neural_network import NeuralNetwork
from cost_functions import quadratic_error, quadratic_error_derivative
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative
import matplotlib.pyplot as plt 

""" 
TODO:
Crear ejemplos de datos
Implementar descenso de gradiente minibatch 
Crear funciones de preprocesado de datos
X Unificar todo en una clase Network
 """

# ejemplo de datos
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


nn = NeuralNetwork([Layer(2, 3), 
        ActivationLayer(tanh, tanh_derivative),
        Layer(3, 3),
        ActivationLayer(tanh, tanh_derivative),
        Layer(3, 1),
        ActivationLayer(tanh, tanh_derivative)])

y_pred = nn.predict(x_train)
print(y_pred)

nn.train(x_train, y_train, epochs=1000, learning_rate=0.1)

y_pred = nn.predict(x_train)
print(y_pred)
