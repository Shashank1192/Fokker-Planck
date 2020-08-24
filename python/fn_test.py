import tensorflow as tf
import autograd.numpy as np
import der_test as dt
import utility as ut
import diff as df
#import eqn1 as eq
import autograd as ag

def func(x):
    return x[:, 0]**2 #+ x[:, 1]*x[:, 0] - tf.math.sqrt(x[:, 2])

def func_out(input_shape):
    return (input_shape[0], 1, )

x = tf.keras.Input(shape = (1, ), name = 'x', dtype = tf.float32)
func_layer = tf.keras.layers.Lambda(func, output_shape = func_out)(x)
model_ = tf.keras.Model(inputs = x, outputs = func_layer, name = 'func_model')
a = tf.constant([[1], [4]], dtype=tf.float32)
"""
@ut.tester
@tf.function
def model__hess(x):
    return tf.reshape(tf.hessians(model_(x), x), (2,2,3,3))


#dt.comp_grad(model_, a)
#print(dt.grad(model_, a))
#model__hess(a)
#print(gmodel([1]))
print(model_(a))
print(df.grad(func, a))
print(df.hess(func, a))
print(df.partial(model_, a, 0))
#print(eq.diff_op(model_, a))
"""


b = tf.constant([[5], [9]], dtype=tf.float32)
print(a*b)
