# Automatic differentiation

from eqn1 import diff_op, test_func
import solve as sl
import tensorflow as tf

tf.keras.backend.set_floatx('float64')
model = sl.dgm_model(dim=1, num_nodes=50, num_hidden_layers=4)
inp =  tf.constant([[1.0], [4.0]], type=tf.float64)
print(diff_op(test_func, inp))

"""
with tf.autodiff.ForwardAccumulator(primals = inp, tangents=tf.constant([[1.0], [1.0]], dtype=tf.float64) ) as acc:
    g = diff_op(model, inp)
print(acc.jvp(g))
"""
