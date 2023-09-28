from layer import *
from activation_layer import *
from neural_network import NeuralNetwork
from cost_functions import quadratic_error, quadratic_error_derivative
from activation_functions import sigmoid, sigmoid_derivative, tanh, tanh_derivative
import matplotlib.pyplot as plt 

from tensorflow import keras

""" 
TODO:
Implementar descenso de gradiente minibatch 
Implementar grafico de activaciones
Crear funciones de preprocesado de datos
 """

def to_categorically(y):
    # función que va a pasar un número a un vector para cada elemento del array
    processed_array = []
    for item in y:
        v = [0,0,0,0,0,0,0,0,0,0]
        v[item] = 1
        processed_array.append(v)
    return processed_array

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# preprocesado de datos de entrenamiento

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255


y_train = to_categorically(y_train)
y_test = to_categorically(y_test)

# creamos el modelo
red = NeuralNetwork([Layer(784, 100), 
        ActivationLayer(tanh, tanh_derivative),
        Layer(100, 50),
        ActivationLayer(tanh, tanh_derivative),
        Layer(50, 10),
        ActivationLayer(tanh, tanh_derivative)])

# entrenamos el modelo con las primeras 2000 muestras para ahorrar tiempo
red.train(x_train[:2000], y_train[:2000], epochs=20, learning_rate=0.3)

# para realizar una prueba elegimos una muestra al azar
random = np.random.randint(0, 20000)
random_sample= x_test[random]

# graficamos el número de mnist con plt
pixels = random_sample.reshape((28, 28))
plt.imshow(pixels, cmap='gray')

# predecimos su valor con el modelo
p = red.predict(random_sample)
prediccion = str(np.argmax(p))
string_prediccion = "Estoy seguro de que esto es un " + prediccion
plt.xlabel(string_prediccion)
plt.show()





