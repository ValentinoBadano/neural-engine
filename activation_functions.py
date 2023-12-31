import numpy as np 

# algunas funciones de activación para experimentar en el modelo

# función de activación Sigmoide
def sigmoid(x):
	x = np.clip(x, -500, 500)     # evita el overflow
	return 1 / (1 + np.exp(-x))

# derivada de la función de activación Sigmoide
def sigmoid_derivative(x):
    x = np.clip(x, -1000, 1000)	  # evita el overflow
    return x * (1 - x)

# función de activación ReLU (unidad lineal rectificada)
def relu(x):
    return np.maximum(0, x)

# derivada de la función de activación ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# función de activación tanh
def tanh(x):
    return np.tanh(x)

# derivada de la función de activación tanh
def tanh_derivative(x):
    return 1 - np.square(np.tanh(x))

# función de activación Softmax (para capas de salida en clasificación multiclase)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # evita el overflow
    return exp_x / np.sum(exp_x, axis=0)