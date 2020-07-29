import tensorflow as tf
import solve
import utility as ut


a = tf.constant([[1, 2, 3], [4,5,6]], dtype=tf.float32)
model = solve.dgm_model(2, 5, 3)

@tf.function
def loss(x):
    f = lambda x_: tf.reduce_sum(tf.math.square(tf.math.square(x_)), axis = 0)
    return tf.vectorized_map(f, x)

@ut.tester
@tf.function
def comp_single_hess(f, x):
    return tf.hessians(f(x), x)

#@tf.function
def comp_hess(f, x):
    hess = []
    for v in x:
        hess.append(comp_single_hess(f, [v]))
    return hess

@ut.tester
@tf.function
def comp_grad(f, x):
    return tf.gradients(f(x), x)

@ut.tester
@tf.function
def comp_partial(f, x, i):
    return tf.gradients(tf.gradients(f(x), x), x)

@ut.tester
@tf.function
def model_hess(x):
    return tf.hessians(model(x), x)

"""
comp_partial(loss, a, 0)
comp_grad(loss, a)
comp_single_hess(loss, [a[0]])
hess = []
for v in a:
    hess.append(comp_single_hess(loss, [v]))
print("hessian is :")
print(hess)

comp_hess(loss, a)
comp_grad(model, a)
model_hess(a)
comp_partial(model, a, 0)
"""
