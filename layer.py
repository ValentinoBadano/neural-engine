import numpy as np

class Layer(object):
	
	def __init__(self, inputs, outputs):
		# Dado el tamaño de la capa inicializa los pesos y los bíases
		self.input = None
		self.output = None

		# las siguientes matrices son aleatorias con números del intervalo (-1, 1)
		self.w = np.random.rand(outputs, inputs) - 0.5 # matriz de pesos
		self.b = np.random.rand(outputs, 1) - 0.5 # matriz fila de bíases
		

	def propagate(self, input):
		# Dada una entrada calcula y retorna la salida Y = W*X + B
		self.input = input
		self.output = np.dot(self.w, input) + self.b
		return self.output
	
	def retropropagation(self, de_dy, learning_rate):
		# Dado el error en la salida ajusta los pesos y los bíases
		# de_dx = np.dot(de_dy, self.w.T)
		de_dw = np.dot(self.input.T, de_dy)

		self.w -= learning_rate * de_dw
		self.b -= learning_rate * de_dy

	def printValues(self):
		print("W =", self.w, "b=", self.b)