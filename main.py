from layer import *
from activation_layer import *
from cost_functions import quadratic_error, quadratic_error_derivative
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative
import matplotlib.pyplot as plt 

""" 
TODO:
Crear ejemplos de datos
implementar descenso de gradiente minibatch 
 """

# ejemplo de datos
val_x = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
val_y = np.array([[[0]], [[1]], [[1]], [[0]]])


red = [Layer(2, 3), 
        ActivationLayer(tanh, tanh_derivative),
        Layer(3, 1),
        ActivationLayer(tanh, tanh_derivative)]


# ejemplo de predicción
for x, y in zip(val_x, val_y):
    salida = x

    for capa in red:
        salida = capa.propagate(salida)
    print(x, salida)


#          algoritmo de entrenamiento

epochs = 500 # número de veces que se "entrenará" al modelo

for i in range(epochs):
    for x, y in zip(val_x, val_y):
        salida = x

        # prop. hacia adelante
        for capa in red:
            salida = capa.propagate(salida)

        plt.plot(i, quadratic_error(salida, y), 'ro', color='blue', markersize=0.2)
        error = quadratic_error_derivative(y, salida)

        # prop. hacia atrás
        for capa in reversed(red):
            error = capa.retropropagate(error, 0.1)


print("\nLuego del entrenamiento:")
for x, y in zip(val_x, val_y):
    salida = x

    for capa in red:
        salida = capa.propagate(salida)
    print(x, salida)

plt.xlabel('Training')
plt.ylabel('Loss')
plt.show()