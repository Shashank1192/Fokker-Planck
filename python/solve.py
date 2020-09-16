import tensorflow as tf
import numpy as np

# creates layer S_(l+1) from S_l, x and S_1 according to the DGM paper  ---->  https://arxiv.org/abs/1708.07469
def dgm_layer(l, x, s_1, s_l, num_nodes):
    z_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'z_x_' + str(l))(x)
    z_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'z_s_' + str(l))(s_l)
    z = tf.keras.layers.Activation(tf.keras.activations.tanh)(tf.keras.layers.Add(name = 'z_' + str(l))([z_x, z_s]))
    g_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'g_x_' + str(l))(x)
    g_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'g_s_' + str(l))(s_1)
    g = tf.keras.layers.Activation(tf.keras.activations.tanh)(tf.keras.layers.Add(name = 'g_' + str(l))([g_x, g_s]))
    r_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'r_x_' + str(l))(x)
    r_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'r_s_' + str(l))(s_l)
    r = tf.keras.layers.Activation(tf.keras.activations.tanh)(tf.keras.layers.Add(name = 'r_' + str(l))([r_x, r_s]))
    h_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'h_x_' + str(l))(x)
    h_sr = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False)(tf.keras.layers.Multiply(name = 'h_sr_' + str(l))([s_l, r]))
    h = tf.keras.layers.Activation(tf.keras.activations.tanh)(tf.keras.layers.Add(name = 'h' + str(l))([h_x, h_sr]))
    one__g = tf.keras.layers.Subtract(name = 'one__g_' + str(l))([tf.ones_like(g), g])
    one__gh = tf.keras.layers.Multiply(name = 'one__gh_' + str(l))([one__g, h])
    zs = tf.keras.layers.Multiply(name = 'zs' + str(l))([z, s_l])
    return tf.keras.layers.Add(name = 's_' + str(l+1))([one__gh, zs])

class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, out_shape, dtype = tf.float64):
        super(ReshapeLayer, self).__init__(dtype = dtype)
        self.out_shape = out_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.out_shape)

# creates the full DGM architechture as a tensorflow model
def dgm_model(dim, num_nodes, num_hidden_layers, name = "FP_solver"):
    x = tf.keras.Input(shape = [None, dim, ], name = 'x')
    s_1 = tf.keras.layers.Dense(units = num_nodes, activation = 'tanh', name = 's_1')(x)
    s_l = s_1
    for l in range(1, num_hidden_layers):
        s_l = dgm_layer(l = l, x = x, s_1 = s_1, s_l = s_l, num_nodes = num_nodes)
    f_x = tf.keras.layers.Dense(units = 1, activation = None, name = 'f_x')(s_l)
    f_x_r = ReshapeLayer((tf.shape(x)[0], dim))(f_x)
    model = tf.keras.Model(inputs = x, outputs = f_x_r, name = name)
    tf.keras.utils.plot_model(model, "../images/{}.png".format(model.name), show_shapes=True)
    return model

class DGMSolver(object):
    """
    Implements a Python object that solves quasi-linear parabolic PDEs using DGM architechture
    """
    def __init__(self, eq, num_nodes, num_hidden_layers, name = "DGMSolver", graph = True, graph_path = None):
        # assign attributes
        self.eq = eq
        self.dim = eq.dim
        self.num_nodes = num_nodes
        self.num_hidden_layers = num_hidden_layers
        self.name = name

        # build the DGM arcitechture
        self.x = tf.keras.Input(shape = [None, self.dim, ], name = 'x')
        s_1 = tf.keras.layers.Dense(units = num_nodes, activation = 'tanh', name = 's_1')(self.x)
        s_l = s_1
        for l in range(1, num_hidden_layers):
            s_l = dgm_layer(l = l, x = self.x, s_1 = s_1, s_l = s_l, num_nodes = num_nodes)
        f_x = tf.keras.layers.Dense(units = 1, activation = None, name = 'f_x')(s_l)
        f_x_r = ReshapeLayer((tf.shape(self.x)[0], self.dim))(f_x)
        self.model = tf.keras.Model(inputs = self.x, outputs = f_x_r, name = name)

        # plot the computational graph of the neural network
        if graph:
            if graph_path is None:
                graph_path = "../images/{}.png".format(self.model.name)
            tf.keras.utils.plot_model(self.model, graph_path, show_shapes=True)

        # complie the model with custom loss function
        def custom_loss(loss, model, input):
            def loss_(y_true, y_pred):
                return tf.py_function(loss, [model, input], Tout = tf.float32)
            return loss_

        self.model.compile(loss = custom_loss(self.eq.loss, self.model, self.x), optimizer='adam')
        self.x = tf.constant([[1.0], [4.0]], dtype=tf.float32)
        print(self.model.loss(tf.constant([[1.0], [4.0]], dtype=tf.float32), tf.constant([[1.0], [4.0]], dtype=tf.float32)))

    def synthesize_data(self, num_samples = int(1e4)):
        inputs = self.eq.domain_sampler(num_samples = num_samples)
        outputs = np.zeros((num_samples, 1))
        return inputs, outputs

    def solve(self, num_samples = int(1e4)):
        inputs, outputs = self.synthesize_data(num_samples)
        print(self.model.evaluate(inputs, outputs))
        self.model.fit(inputs, outputs, epochs=2)


#model = dgm_model(dim = 2, num_nodes = 50, num_hidden_layers = 4)
#model.summary()
