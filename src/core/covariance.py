import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def one_to_one_dist(x, y):
    """
    calculate the square euclidean distance between two vectors (or scalars)

    x: n-dimensional vector or scalar (x and y must have the same shape)
    y: n-dimensional vector or scalar
    """
    return jnp.sum((x - y)**2.0)

@jax.jit
def cdist(X,Y):
    """
    calculate the square euclidean distance between each pair of the two collections of inputs

    X: m_X by n array of m_X observations in an n-dimensional space
    Y: m_Y by n array of m_Y observations in an n-dimensional space
    """
    one_to_multi = jax.vmap(one_to_one_dist, (0,None),0)
    multi_to_multi = jax.vmap(one_to_multi, (None,0),1)
    return multi_to_multi(X,Y)

def cov_matrix_emulator(input_xy_1, input_theta_1, input_xy_2, input_theta_2, stdev, length_xy, length_theta):
    """
    squared exponential covariance kernel with diagonal noise term
    
    stdev:  stdandard deviation
    length_xy: correlation length for xy (controllable variable)
    length_theta: correlation length for theta (uncontrollable variable)
    """
    # divide by correlation length
    input_xy_1 /= length_xy
    input_xy_2 /= length_xy
    input_theta_1 /= length_theta
    input_theta_2 /= length_theta

    # calculate the covariance matrix
    dist = cdist(input_xy_1, input_xy_2)
    dist += cdist(input_theta_1, input_theta_2)

    cov_matrix = stdev**2 * jnp.exp(-dist)
    
    return cov_matrix
