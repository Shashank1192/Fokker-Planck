import tensorflow as tf
import eqn

# creates layer S_(l+1) from S_l, x and S_1 according to the DGM paper  ---->  https://arxiv.org/abs/1708.07469
def dgm_layer(l, x, s_1, s_l, num_nodes):
    z_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'z_x_' + str(l))(x)
    z_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'z_s_' + str(l))(s_l)
    z = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'z_' + str(l))([z_x, z_s]))
    g_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'g_x_' + str(l))(x)
    g_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'g_s_' + str(l))(s_1)
    g = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'g_' + str(l))([g_x, g_s]))
    r_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'r_x_' + str(l))(x)
    r_s = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False, name = 'r_s_' + str(l))(s_l)
    r = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'r_' + str(l))([r_x, r_s]))
    h_x = tf.keras.layers.Dense(units = num_nodes, activation = None, name = 'h_x_' + str(l))(x)
    h_sr = tf.keras.layers.Dense(units = num_nodes, activation = None, use_bias = False)(tf.keras.layers.Multiply(name = 'h_sr_' + str(l))([s_l, r]))
    h = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'h' + str(l))([h_x, h_sr]))
    one__g = tf.keras.layers.Subtract(name = 'one__g_' + str(l))([tf.ones_like(g), g])
    one__gh = tf.keras.layers.Multiply(name = 'one__gh_' + str(l))([one__g, h])
    zs = tf.keras.layers.Multiply(name = 'zs' + str(l))([z, s_l])
    return tf.keras.layers.Add(name = 's_' + str(l+1))([one__gh, zs])

# creates the full DGM architechture as a tensorflow model
def dgm_model(dim, num_nodes, num_hidden_layers, name = "FP_solver"):
    x = tf.keras.Input(shape = [None, dim, ], name = 'x')
    s_1 = tf.keras.layers.Dense(units = num_nodes, activation = 'tanh', name = 's_1')(x)
    s_l = s_1
    for l in range(1, num_hidden_layers):
        s_l = dgm_layer(l = l, x = x, s_1 = s_1, s_l = s_l, num_nodes = num_nodes)
    f_x = tf.reshape(tf.keras.layers.Dense(units = 1, activation = None, name = 'f_x')(s_l), shape = (tf.shape(x)[0], dim))
    model = tf.keras.Model(inputs = x, outputs = f_x, name = name)
    tf.keras.utils.plot_model(model, "../images/{}.png".format(model.name), show_shapes=True)
    return model

class DGMSolver(object):
    """
    Implements a Python object that solves quasi-linear parabolic PDEs using DGM architechture
    """
    def __init__(self, eqn, dim, num_nodes, num_hidden_layers, name = "DGMSolver"):
        self.eqn = eqn
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_hidden_layers = num_hidden_layers
        self.name = name
#model = dgm_model(dim = 2, num_nodes = 50, num_hidden_layers = 4)
#model.summary()
