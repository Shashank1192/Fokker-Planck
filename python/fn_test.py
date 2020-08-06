import tensorflow as tf
import numpy as np
import der_test as dt
import utility as ut
import diff as df


def func(x):
    return x[:, 0]**2 + x[:, 1]*x[:, 0] - tf.math.sqrt(x[:, 2])

def func_out(input_shape):
    return (input_shape[0], 1, )

x = tf.keras.Input(shape = (3, ), name = 'x')
func_layer = tf.keras.layers.Lambda(func, output_shape = func_out)(x)
model_ = tf.keras.Model(inputs = x, outputs = func_layer, name = 'func_model')
@ut.tester
@tf.function
def model__hess(x):
    return tf.reshape(tf.hessians(model_(x), x), (2,2,3,3))
a = tf.constant([[1,2,3], [4, 5, 6]], dtype=tf.float64)
print(tf.shape(a))
print( a[:, 0]**2 + a[:, 1]*a[:, 0] )
dt.comp_grad(model_, a)
print(dt.grad(model_, a))
model__hess(a)

#print(df.grad(func, a))
#print(df.hess(func, a))
print(0.5**2*df.partial(func, a, 2, 2))
