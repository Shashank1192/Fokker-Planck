import tensorflow as tf
import solve
tf.get_logger().setLevel('ERROR')
space_dim = 3

model = solve.dgm_model(space_dim, 5, 3)

a = tf.constant([[1, 2, 3], [4,5,6]], dtype=tf.float32)
for vec in a:
    print(vec)
@tf.function
def loss(x):
    f = lambda x_: tf.reduce_sum(tf.math.square(tf.math.square(x_)), axis = 0)
    return tf.vectorized_map(f, x)

@tf.function
def comp_single_hess(f, x):
    return tf.hessians(f(x), x)

#@tf.function
def comp_hess(f, x):
    hess = []
    for v in x:
        hess.append(comp_single_hess(f, [v]))
    return hess


@tf.function
def comp_grad(f, x):
    return tf.gradients(f(x), x)

#"""
dy_dx = comp_grad(loss, a)
print(dy_dx)
d2y_d2x = comp_single_hess(loss, [a[0]])

print(d2y_d2x)
hess = []
for v in a:
    hess.append(comp_single_hess(loss, [v]))
print("hessian is :")
print(hess)
print("other hessian is")
print(comp_hess(loss, a))
#"""

model = solve.dgm_model(1, 5, 3)
a = tf.constant([[1, 2]], dtype = tf.float32)
print(model(a))
print(comp_grad(model, a))

@tf.function
def model_hess(x):
    return tf.hessians(model(x), x)

print(model_hess([a]))
