import tensorflow as tf
import numpy as np

def func(x):
    return x[0]**2 + x[1]*x[0] - tf.math.sqrt(x[2])

def func_out(input_shape):
    return (input_shape[0], 1, )

x = tf.keras.Input(shape = (3, 1), name = 'x')
func_layer = tf.keras.layers.Lambda(func, output_shape = func_out)(x)
model = tf.keras.Model(inputs = x, outputs = func_layer, name = 'func_model')

a = tf.constant(np.array([[1], [2], [3]]), dtype=tf.float32)
print(model(a))
