import keras as ks
import keras.layers as kl
import keras.activations as ka
import tensorflow as tf
"""
def dgm_layer(input, s1_layer, sl_layer, dimension, node_count):
    z_ = ks.layers.Dense(units = node_count, activation = None)(sl_layer)
    z_x = ks.layers.Dense(units = node_count, activation = None, use_bias = False)(input)
    z = ks.activations.tanh(ks.layers.Add()([z_, z_x]))
    g_ = ks.layers.Dense(units = node_count, activation = None)(s1_layer)
    g_x = ks.layers.Dense(units = node_count, activation = None, use_bias = False)(input)
    g = ks.activations.tanh(ks.layers.Add()([g_, g_x]))
    r_ = ks.layers.Dense(units = node_count, activation = None)(sl_layer)
    r_x = ks.layers.Dense(units = node_count, activation = None, use_bias = False)(input)
    r = ks.activations.tanh(ks.layers.Add()([r_, r_x]))
    h_dot = ks.layers.Multiply()([sl_layer, r])
    h_ = ks.layers.Dense(units = node_count, activation = None)(h_dot)
    h_x = ks.layers.Dense(units = node_count, activation = None, use_bias = False)(input)
    h = ks.activations.tanh(ks.layers.Add()([h_, h_x]))
    print(g.shape)


x = ks.layers.Input(shape = (2,))
s1 = ks.layers.Dense(units = 50, activation = 'tanh')(x)
dgm_layer(x, s1, s1, 2, 50)
"""

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
        kl.Dot(axes = (1,1))([self.W_z, S_l])
        Z = ka.tanh(tf.matmul(self.U_z, x) +  self.b_z)#ka.tanh(kl.Add()([kl.Dot(axes = 1)([self.U_z, x]), kl.Dot(axes = 1)([self.W_z, S_l]), self.b_z]))
        return Z


node_count = 50

x = ks.Input(shape = (2, ))
s1 = kl.Dense(units = node_count, activation = 'tanh')(x)
z1 = DGM_Layer(units = node_count, space_dim = 1)([x, s1, s1])
print("x-Shape, S-shape, z_shape  = {} {} ()".format(x.shape, s1.shape, z1.shape))

model = ks.Model(inputs = x, outputs = z1)

print('model oshape:'.format(model.output_shape))
"""
model = ks.models.Sequential()
model.add(kl.Input(shape = (None, 2, 1)))
model.add(kl.Dense(units = 32, activation = 'tanh'))
model.add(DGM_Layer(32, 1))
model.add(kl.Dense(8, activation = 'softmax'))
model.build(input_shape=(None, 2, 1))
"""
model.summary()
