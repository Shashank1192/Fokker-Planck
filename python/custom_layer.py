import tensorflow as tf

class DGMLayer(tf.keras.layers.Layer):
    """
    Description: Class for implementing a DGM layer
    """
    def __init__(self, S_l_layer, input_dim, num_nodes, activation, dtype=tf.float64):
        super(DGMLayer, self).__init__(dtype=dtype)
        self.S_l = S_l_layer
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.activation = activation

    def build(self, input_shape):
        self.U_z_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_z_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_z_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_g_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_g_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_g_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_r_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_r_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_r_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_h_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_h_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_h_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)

    def call(self, input):
        S_l = self.S_l(input)
        Z_l = self.activation(tf.matmul(input, self.U_z_l) + tf.matmul(S_l, self.W_z_l) + self.b_z_l)
        G_l = self.activation(tf.matmul(input, self.U_g_l) + tf.matmul(S_l, self.W_g_l) + self.b_g_l)
        R_l = self.activation(tf.matmul(input, self.U_r_l) + tf.matmul(S_l, self.W_r_l) + self.b_r_l)
        H_l = self.activation(tf.matmul(input, self.U_h_l) + tf.matmul(tf.multiply(S_l, R_l), self.W_h_l) + self.b_h_l)
        return tf.multiply(tf.ones_like(G_l) - G_l, H_l) + tf.multiply(Z_l, S_l)



class DGMModel(tf.keras.models.Model):
    """
    Description: Class for implementing the DGM architechture
    """
    def __init__(self, input_dim, num_nodes, num_dgm_layers, activation, dtype=tf.float64):
        super(DGMModel, self).__init__(dtype=dtype)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.num_dgm_layers = num_dgm_layers
        self.activation = activation
        self.S_layers = tf.keras.layers.Dense(units=self.num_nodes, activation=self.activation, name='S_1', dtype=dtype)
        for i in range(num_dgm_layers):
            self.S_layers = DGMLayer(S_l_layer=self.S_layers, input_dim=self.input_dim, num_nodes=self.num_nodes, activation=self.activation, dtype=dtype)
        self.f_layer = tf.keras.layers.Dense(units=1, use_bias=True, activation=None, name='f_layer', dtype=dtype)

    def call(self, input):
         x = self.S_layers(input)
         return self.f_layer(x)

num_nodes = 50

"""
S_1 = tf.keras.layers.Dense(units = num_nodes, activation = 'tanh', name = 's_1', dtype=tf.float64)
dgm = DGMLayer(S_1, 2, 50, tf.keras.activations.tanh)
print(dgm(x))
"""

model = DGMModel(2, 50, 3, tf.keras.activations.tanh)
print(model(x))
print(model.summary())
