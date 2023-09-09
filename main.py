from layer import *
from activation_layer import *
from cost_functions import quadratic_error_derivative
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative

""" 
TODO:
Crear ejemplos de datos
Crear función de coste y su derivada
Crear capas de activación (sigmoide, tanh?)
Entrenar, entrenar, entrenar...
crear algoritmo de entrenamiento mediante rp
 """

# ejemplo con AND
val_x = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
val_y = np.array([[[0]], [[0]], [[0]], [[1]]])


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

epochs = 1000 # número de veces que se "entrenará" al modelo

for i in range(epochs):
    for x, y in zip(val_x, val_y):
        salida = x

        # prop. hacia adelante
        for capa in red:
            salida = capa.propagate(salida)

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