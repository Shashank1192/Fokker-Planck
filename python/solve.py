import keras as ks
import keras.layers as kl
import keras.activations as ka
import tensorflow as tf

def dgm_layer(x, s1, sl, node_count):
    z_x = tf.keras.layers.Dense(units = node_count, activation = None, name = 'my_dense')(x)
    z_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False)(sl)
    z = tf.keras.activations.tanh(tf.keras.layers.Add()([z_x, z_s]))
    g_x = tf.keras.layers.Dense(units = node_count, activation = None)(x)
    g_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False)(s1)
    g = tf.keras.activations.tanh(tf.keras.layers.Add()([g_x, g_s]))
    r_x = tf.keras.layers.Dense(units = node_count, activation = None)(x)
    r_s = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False)(sl)
    r = tf.keras.activations.tanh(tf.keras.layers.Add()([r_x, r_s]))
    h_x = tf.keras.layers.Dense(units = node_count, activation = None)(x)
    h_sr = tf.keras.layers.Dense(units = node_count, activation = None, use_bias = False)(tf.keras.layers.Multiply()([sl, r]))
    h = tf.keras.activations.tanh(tf.keras.layers.Add()([h_x, h_sr]))
    _g = tf.keras.layers.Subtract()([tf.ones_like(g), g])
    _gh = tf.keras.layers.Multiply()([_g, h])
    zs = tf.keras.layers.Multiply()([z, sl])
    return tf.keras.layers.Add()([_gh, zs])


node_count, space_dim = 7, 1
x = tf.Variable([[1, 2], [3, 4]])
s1 = tf.keras.layers.Dense(units = node_count, activation="relu")(x)
tf.print(s1)
z = dgm_layer(x, s1, s1, node_count = node_count)
print(z)

x = tf.keras.Input(shape = (space_dim + 1, 1))
print("shape o f x = {}".format(x.shape))
s1 = tf.keras.layers.Dense(units = node_count, activation="relu")(x)
s2 = dgm_layer(x, s1, s1, node_count)
model = tf.keras.Model(inputs = x, outputs = s2, name="Fokker-Planck solver")
model.summary()
tf.keras.utils.plot_model(model, "../images/{}.png".format(model.name), show_shapes=True)
