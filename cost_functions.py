def quadratic_error(x, y):
	error = (y - x)**2
	return abs(error)

def quadratic_error_derivative(x, y):
	error = 2*(y - x)
	return error / y.size