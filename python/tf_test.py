import tensorflow as tf
import solve

space_dim = 3

model = solve.dgm_model(space_dim, 5, 3)

a = tf.constant([1, 2, 3], dtype=tf.float32)


def loss(x_):
    return tf.reduce_sum(tf.math.square(x_), axis = 0)

@tf.function
def comp_hess(f, x):
    #f = tf.reduce_sum(tf.math.square(x), axis = 0)
    hes = tf.hessians(f(x), x)
    return hes


d2y_d2x = comp_hess(loss,a)
print(d2y_d2x)




x = tf.constant([[1,2,3,4]], dtype = 'float32')
with tf.GradientTape() as tape:
    tape.watch(x)
    preds = model(x)

grads = tape.gradient(preds, x)
print(grads)

def get_my_hessian(f, x):
    with tf.GradientTape(persistent=True) as hess_tape:
        hess_tape.watch(x)
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = f(x)
        grad = grad_tape.gradient(y, x)
        grad_grads = [hess_tape.gradient(g, x) for g in grad]
    hess_rows = [gg[tf.newaxis, ...] for gg in grad_grads]
    hessian = tf.concat(hess_rows, axis=0)
    return hessian


def calc_hessian_diag(f, x):
    """
    Calculates the diagonal entries of the Hessian of the function f
    (which maps rank-1 tensors to scalars) at coordinates x (rank-1
    tensors).

    Let k be the number of points in x, and n be the dimensionality of
    each point. For each point k, the function returns

      (d^2f/dx_1^2, d^2f/dx_2^2, ..., d^2f/dx_n^2) .

    Inputs:
      f (function): Takes a shape-(k,n) tensor and outputs a
          shape-(k,) tensor.
      x (tf.Tensor): The points at which to evaluate the Laplacian
          of f. Shape = (k,n).

    Outputs:
      A tensor containing the diagonal entries of the Hessian of f at
      points x. Shape = (k,n).
    """
    # Use the unstacking and re-stacking trick, which comes
    # from https://github.com/xuzhiqin1990/laplacian/
    with tf.GradientTape(persistent=True) as g1:
        # Turn x into a list of n tensors of shape (k,)
        x_unstacked = tf.unstack(x, axis=1)
        g1.watch(x_unstacked)

        with tf.GradientTape() as g2:
            # Re-stack x before passing it into f
            x_stacked = tf.stack(x_unstacked, axis=1) # shape = (k,n)
            g2.watch(x_stacked)
            f_x = f(x_stacked) # shape = (k,)

        # Calculate gradient of f with respect to x
        df_dx = g2.gradient(f_x, x_stacked) # shape = (k,n)
        # Turn df/dx into a list of n tensors of shape (k,)
        df_dx_unstacked = tf.unstack(df_dx, axis=1)

    # Calculate 2nd derivatives
    d2f_dx2 = []
    for df_dxi in df_dx_unstacked:
        for xj in x_unstacked:
        # Take 2nd derivative of each dimension separately:
        #   d/dx_i (df/dx_i)
            d2f_dx2.append(g1.gradient(df_dxi, xj))

    # Stack 2nd derivates
    d2f_dx2_stacked = tf.stack(d2f_dx2, axis=1) # shape = (k,n)

    return d2f_dx2_stacked

#print(loss.numpy())
d2f_dx2 = get_my_hessian(loss, x)
print(d2f_dx2)
