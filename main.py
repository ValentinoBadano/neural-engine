from layer import *

""" 
TODO:
Crear ejemplos de datos
Crear función de coste y su derivada
Crear capas de activación (sigmoide, tanh?)
Entrenar, entrenar, entrenar...
crear algoritmo de entrenamiento mediante rp
 """

val_x = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
val_y = np.reshape([[0],[0],[0],[1]], (4,1,1))

red = [Layer(2, 3),
        Layer(3, 1)]

#capas[0].printValues()
#capas[1].printValues()
#print("Salida: ",capas[0].propagate(val_x[0][0]))
for x, y in zip(val_x, val_y):
    salida = x

    for capa in red:
        salida = capa.propagate(salida)
        
print("1 ^ 1 =", salida)