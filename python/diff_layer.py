"""
A test for creating a differentiation layer
"""
import tensorflow as tf
import solve

# Make a custom layer that needs to be differentiated
class PowerLayer(tf.keras.layers.Layer):
    def __init__(self, power, dtype = tf.float64):
        super(PowerLayer, self).__init__(dtype = dtype)
        self.power = power

    def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.math.pow(inputs, self.power)


# Test our custom layer
pl = PowerLayer(3)
test_input = tf.constant([[1.0], [4.0]], dtype=tf.float64)
print(pl(test_input))


# Make a differentiation layer
class DiffLayer(tf.keras.layers.Layer):
    def __init__(self, func_layer, dtype = tf.float64):
        super(DiffLayer, self).__init__(dtype = dtype)
        self.func = func_layer
        self.inputs = None

    def call(self, inputs): # Defines the computation from inputs to outputs
        self.inputs = inputs
        with tf.GradientTape() as tape:
            tape.watch(self.inputs)
            fx = self.func(self.inputs)
            df = tape.gradient(fx, self.inputs)
        return df

# Test our differentiation layer
dl = DiffLayer(pl)
print(dl(test_input))


# Make a more sophisticated differential operator
def operator(func, input, a=0.3, b=0.5, sigma=0.1):
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


# Make a more sophisticated differentiation layer
class DiffOpLayer(tf.keras.layers.Layer):
    def __init__(self, diff_op, func_layer, dtype = tf.float64):
        super(DiffOpLayer, self).__init__(dtype = dtype)
        self.func = func_layer
        self.diff_op = diff_op
        self.inputs = None


    def call(self, inputs): # Defines the computation from inputs to outputs
        self.inputs = inputs
        return self.diff_op(self.func, self.inputs)

# Test our sophisticated differentiation layer
dol = DiffOpLayer(operator, pl)
print(dol(test_input))


"""
Can you train a model with such layers?
"""

# Create a DGM model with the last layerperforming differentiation
dim = 1
num_nodes = 50
num_hidden_layers = 3
x = tf.keras.Input(shape = [None, dim, ], name = 'x')
s_1 = tf.keras.layers.Dense(units = num_nodes, activation = 'tanh', name = 's_1')(x)
s_l = s_1
for l in range(1, num_hidden_layers):
    s_l = solve.dgm_layer(l = l, x = x, s_1 = s_1, s_l = s_l, num_nodes = num_nodes)
f_x = tf.keras.layers.Dense(units = 1, activation = None, name = 'f_x')(s_l)
f_x_r = solve.ReshapeLayer((tf.shape(x)[0], dim))(f_x)
model = tf.keras.Model(inputs = x, outputs = f_x_r, name = 'DGM_diff_test')
tf.keras.utils.plot_model(model, "../images/{}.png".format(model.name), show_shapes=True)
