import tensorflow as tf

class MultiplyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiplyLayer, self).__init__()

    def call(self, input_1, input_2, training=False):
        x = tf.math.multiply(input_1, input_2)

        return x

# Input variable with value of 2
input_x = tf.Variable(2)
# Input variable with value of 21
input_y = tf.Variable(21)

# Creating layer object:
mylayer = MultiplyLayer()
# Passing input to the layer
output = mylayer(input_x, input_y, training=False)
# Printing output, it should be something like:
# tf.Tensor(42, shape=(), dtype=int32)
# So, 2*21 = 42
print(output)

"""
x = tf.Variable([[1, 2], [3, 4]])
s_1 = tf.keras.layers.Dense(units = node_count, activation="relu")(x)
tf.print(s_1)
z = dgm_layer(1, x, s_1, s_1, node_count = node_count)
print(z)
"""
