import tensorflow as tf
import numpy as np

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
h = tf.zeros_like(a)
h_col = tf.ones_like(a[:, 0])
a[:, 0] + h_col
print(a[:, 0] + h_col)
tf.tensor_scatter_nd_update(a, [[0, 0], [0, 1]], a[:, 0] + h_col)
print(a)



print(tf.shape(a))

def grad(func, input):
    dx = 1e-6
    dim = tf.shape(input)[1]
    gradients = []
    for t in input:
        x = tf.reshape(t, (1, dim))
        partials = []
        for i in range(dim):
            h = np.zeros(dim)
            h[i] = dx
            h = tf.constant(h, dtype = tf.float64)
            partials.append((func(x + h) - func(x - h))/(2*dx))
        gradients.append(tf.concat(partials, 0))
    return tf.stack(gradients)


def hess(func, input):
    dx, dy = 1e-4, 1e-4
    dim = tf.shape(input)[1]
    hessians = []
    for t in input:
        x = tf.reshape(t, (1, dim))
        hessian = []
        for i in range(dim):
            hx = np.zeros(dim)
            hx[i] = dx
            hx = tf.constant(hx, dtype = tf.float64)
            row_partials = []
            for j in range(dim):
                hy = np.zeros(dim)
                hy[j] = dy
                hy = tf.constant(hy, dtype = tf.float64)
                left = (func(x + hx + hy) - func(x - hx + hy))/(2*dx)
                right = (func(x + hx - hy) - func(x - hx - hy))/(2*dx)
                row_partials.append((left-right)/(2*dy))
            #print("row {}".format(row_partials))
            hessian.append(tf.concat(row_partials, 0))
        hessians.append(tf.stack(hessian))
    return tf.stack(hessians)

def mixed_partial(func, input, i, j):
    dx, dy = 1e-4, 1e-4
    dim = tf.shape(input)[1]
    partials = []
    hx = np.zeros(dim)
    hx[i] = dx
    hx = tf.constant(hx, dtype = tf.float64)
    hy = np.zeros(dim)
    hy[j] = dy
    hy = tf.constant(hy, dtype = tf.float64)
    for t in input:
        x = tf.reshape(t, (1, dim))
        left = (func(x + hx + hy) - func(x - hx + hy))/(2*dx)
        right = (func(x + hx - hy) - func(x - hx - hy))/(2*dx)
        partials.append((left-right)/(2*dy))
        #print("row {}".format(row_partials))
    return tf.stack(partials)
