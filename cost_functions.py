import numpy as np

# Función de error cuadrático 
def quadratic_error(x, y):
	error = np.square(y - x)
	return abs(error)

# Derivada del error cuadrático 
def quadratic_error_derivative(x, y):
	error = 2*(y - x)
	return error / y.size