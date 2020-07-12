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
# Create 3 layers
x = tf.keras.Input(shape = (2, ))
layer1 = tf.keras.layers.Dense(2, activation="relu", name="layer1")(x)
layer2 = tf.keras.layers.Dense(3, activation="relu", name="layer2")(layer1)
layer3 = tf.keras.layers.Dense(4, name="layer3")(layer2)
z1 = tf.keras.layers.Dense(units = 5, activation = 'tanh')(x)

# Call layers on a test input
print(z1(tf.Variable([[1,3], [4,5]])))

class DGM_Layer(kl.Layer):
    """
    Implementation of the Deep Galerkin Layer
    """

    def __init__(self, units, space_dim, **kwargs):
        super(DGM_Layer, self).__init__(**kwargs)
        self.units = units
        self.st_dim = space_dim + 1


    def build(self, input_shape):
        self.U_z = self.add_weight(name = 'U_z', shape = (self.units, self.st_dim), initializer = 'normal', trainable = True)
        self.W_z = self.add_weight(name = 'W_z', shape = (self.units, self.units), initializer = 'normal', trainable = True)
        self.b_z = self.add_weight(name = 'b_z', shape = (self.units, 1), initializer = 'normal', trainable = True)
        super(DGM_Layer, self).build(input_shape)

    def call(self, inputs):
        x, S_1, S_l = inputs
        print('S_l shape, W_z shape = {}'.format(S_l.shape, self.W_z.shape))
        Z =kl.Dot(axes = 1)([self.U_z, x])#, self.b_z]))
        return Z



def dgm_layer(x, s1, sl, units, space_dim, **kwargs):


node_count = 50

x = ks.Input(shape = (2, ))
s1 = kl.Dense(units = node_count, activation = 'tanh')
z1 = DGM_Layer(units = node_count, space_dim = 1)
print("x-Shape, S-shape, z_shape  = {} {} ()".format(x.shape, s1.shape, z1.shape))

model = ks.Model(inputs = x, outputs = z1)
x_data = [[3, 2]]#np.random.normal(size=(1, 10, 10, 1))
print(s1(x_data))
"""
model = ks.models.Sequential()
model.add(kl.Input(shape = (None, 2, 1)))
model.add(kl.Dense(units = 32, activation = 'tanh'))
model.add(DGM_Layer(32, 1))
model.add(kl.Dense(8, activation = 'softmax'))
model.build(input_shape=(None, 2, 1))
"""
model.summary()
