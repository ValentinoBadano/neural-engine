# neural-engine

En busca de entender realmente cómo funcionan las redes neuronales y el aprendizaje automático, decidí crear mi propio framework en Python.
El objetivo de este proyecto es proporcionar una plataforma para aprender sobre el funcionamiento interno de las redes neuronales.

## Notebook explicativo

[Notebook en Colab](https://colab.research.google.com/drive/1rnHMuf3mroqp_uqQY8qtnsTlC13-13tP?usp=sharing)

## Uso
Es muy sencillo crear y entrenar un modelo:
```py
modelo = NeuralNetwork([Layer(100, 20), 
        ActivationLayer(tanh, tanh_derivative),
        Layer(20, 1),
        ActivationLayer(tanh, tanh_derivative)])

modelo.train(x_train, y_train, epochs=400, learning_rate=0.1)

```

## Requisitos

Se necesita tener instalado Python 3.6 o superior.
Quise mantenerlo simple, y por lo tanto, para utilizar este framework solo se necesita tener instaladas las siguientes librerías:

* NumPy
* Matplotlib

## Contribuciones
Se aceptan contribuciones de cualquier tipo. Si tienes alguna sugerencia o mejora, no dudes en abrir una issue o enviar un pull request.

## Licencia
Este proyecto está licenciado bajo la licencia MIT. Esta licencia es de código abierto y permite que cualquiera utilice, modifique y distribuya el código libremente.
