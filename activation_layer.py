class ActivationLayer(object):
	
	def __init__(self, activation_function, activation_function_derivative):
		# Dado el tamaño de la capa inicializa los pesos y los bíases
		self.activation_function = activation_function
		self.activation_function_derivative = activation_function_derivative

	def propagate(self, input):
		self.input = input
		self.output = self.activation_function(input)
		return self.output 

	def retropropagate(self, error, lrate):
		de_dx = self.activation_function_derivative(self.input) * error
		return de_dx