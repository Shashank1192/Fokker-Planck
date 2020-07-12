import keras as ks
import keras.layers as kl
import keras.activations as ka
import tensorflow as tf

def dgm_layer(l, x, s_1, s_l, node_count):
    z_x = tf.keras.layers.Dense(units = node_count, activation = None, name = 'z_x_' + str(l))(x)
    z_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False, name = 'z_s_' + str(l))(s_l)
    z = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'z_' + str(l))([z_x, z_s]))
    g_x = tf.keras.layers.Dense(units = node_count, activation = None, name = 'g_x_' + str(l))(x)
    g_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False, name = 'g_s_' + str(l))(s_1)
    g = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'g_' + str(l))([g_x, g_s]))
    r_x = tf.keras.layers.Dense(units = node_count, activation = None, name = 'r_x_' + str(l))(x)
    r_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False, name = 'r_s_' + str(l))(s_l)
    r = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'r_' + str(l))([r_x, r_s]))
    h_x = tf.keras.layers.Dense(units = node_count, activation = None, name = 'h_x_' + str(l))(x)
    h_sr = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False)(tf.keras.layers.Multiply(name = 'h_sr_' + str(l))([s_l, r]))
    h = tf.keras.activations.tanh(tf.keras.layers.Add(name = 'h' + str(l))([h_x, h_sr]))
    one__g = tf.keras.layers.Subtract(name = 'one__g_' + str(l))([tf.ones_like(g), g])
    one__gh = tf.keras.layers.Multiply(name = 'one__gh_' + str(l))([one__g, h])
    zs = tf.keras.layers.Multiply(name = 'zs' + str(l))([z, s_l])
    return tf.keras.layers.Add(name = 's_' + str(l+1))([one__gh, zs])

def dgm_model(space_dim, node_count, hidden_layer_count, name = "Fokker-Planck solver"):
    x = tf.keras.Input(shape = (space_dim + 1, 1), name = 'x')
    s_1 = tf.keras.layers.Dense(units = node_count, activation = 'tanh', name = 's_1')(x)
    s_l = s_1
    for l in range(1, hidden_layer_count):
        s_l = dgm_layer(l = l, x = x, s_1 = s_1, s_l = s_l, node_count = node_count)
    f_x = tf.keras.layers.Dense(units = 1, activation = None, name = 'f_x')(s_l)
    model = tf.keras.Model(inputs = x, outputs = f_x, name = name)
    tf.keras.utils.plot_model(model, "../images/{}.png".format(model.name), show_shapes=True)
    return model


model = dgm_model(space_dim = 2, node_count = 50, hidden_layer_count = 4)
model.summary()
