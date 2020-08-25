# Implentation of an equation using eqn module
import tensorflow as tf
import eqn
import solve as sl


def diff_op(func, input, a=0.3, b=0.5, sigma=0.1):
    num_x = tf.shape(input)[0]
    dim = tf.shape(input)[1]
    with tf.GradientTape() as tape2:
        tape2.watch(input)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(input)
            fx = func(input)
            c = tf.reshape(a*input - b*tf.math.pow(input, 3), (num_x, dim))
            c_fx = tf.keras.layers.Multiply()([c, fx])
            dc_fx = tape1.gradient(c_fx, input)
            d_fx = tape1.gradient(fx, input)
        d2_fx = tape2.gradient(d_fx, input)
    return  -dc_fx + 0.5 * sigma**2 * d2_fx

def init_cond(input):
    return input
