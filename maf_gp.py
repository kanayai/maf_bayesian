
import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

jax.config.update("jax_enable_x64", True)

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

    # dist = jnp.zeros((input_xy_1.shape[0], input_xy_2.shape[0]))
    # for i in range(input_xy_1.shape[1]):
    #     dist += (input_xy_1[:,i][:,None] - input_xy_2[:,i])**2

    # for i in range(input_theta_1.shape[1]):
    #     dist += (input_theta_1[:,i][:,None] - input_theta_2[:,i])**2

    cov_matrix = stdev**2 * jnp.exp(-dist)
    
    # cov_matrix = jnp.zeros((input_xy_1.shape[0], input_xy_2.shape[0]))
    # for i in range(input_xy_2.shape[0]
    #     jnp.sum((input_xy_1 - input_xy_2[i,:])**2, axis=1)
    #     dist_theta = jnp.sum((input_theta_1 - input_theta_2[i,:])**2, axis=1)
    #     cov_matrix = cov_matrix.at[:,i].set(stdev**2 * jnp.exp( - dist_xy - dist_theta))
    
    return cov_matrix

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





def model(input_xy_exp, input_xy_sim, input_theta_sim, data_exp, data_sim, add_bias_E1=False, add_bias_alpha=False):
    """
    input_xy_exp: experimental loads and angles
    input_xy_sim: simulation loads and angles
    input_theta_sim: simulation thetas (FE parameters)
    data_exp: experimental extensions
    data_sim: simulation extensions
    """

    num_exp = len(input_xy_exp)
    
    # priors of theta
    #161
    #E1 = numpyro.sample("E_1", dist.Normal(148800., 2000.))
    #E2 = numpyro.sample("E_2", dist.Normal(9190., 100.))
    #v12 = numpyro.sample("v_12", dist.Normal(0.34, 0.01))
    #v23 = numpyro.sample("v_23", dist.Normal(0.44, 0.01))
    #G12 = numpyro.sample("G_12", dist.Normal(5060., 70.))
    
    E1 = numpyro.sample("E_1", dist.Normal(161000., 2000.))
    E2 = numpyro.sample("E_2", dist.Normal(11380., 100.))
    v12 = numpyro.sample("v_12", dist.Normal(0.32, 0.01))
    v23 = numpyro.sample("v_23", dist.Normal(0.43, 0.01))
    G12 = numpyro.sample("G_12", dist.Normal(5170., 70.))

    theta = jnp.array([E1, E2, v12, v23, G12])

    # priors of bias
    if add_bias_E1:
        # sigma_b = numpyro.sample("sigma_b", dist.Exponential(0.001))
        bias = []
        for i in range(num_exp):
            bias.append(numpyro.sample("b_"+str(i+1)+"_E1", dist.Normal(0, 1000)))   
    
    if add_bias_alpha:
        # sigma_b = numpyro.sample("sigma_b", dist.Exponential(1/np.deg2rad(1))) 
        bias = []
        for i in range(num_exp):
            bias.append(numpyro.sample("b_"+str(i+1)+"_alpha", dist.Normal(0, np.deg2rad(1))))

    # priors of hyper-parameters
    mean_emulator = numpyro.sample("mu_emulator", dist.Normal(0, 0.01))
    
    stdev_emulator = numpyro.sample("sigma_emulator", dist.Exponential(20.))
    # length_load = numpyro.sample("lambda_P", dist.Gamma(5, 1))
    # length_alpha = numpyro.sample("lambda_alpha", dist.Gamma(jnp.pi/2, 1))
    # length_E1 = numpyro.sample("lambda_E1", dist.Gamma(148800*0.5, 1))
    # length_E2 = numpyro.sample("lambda_E2", dist.Gamma(9190*0.5, 1))
    # length_v12 = numpyro.sample("lambda_v12", dist.Gamma(1, 2))
    # length_v23 = numpyro.sample("lambda_v23", dist.Gamma(1, 2))
    # length_G12 = numpyro.sample("lambda_G12", dist.Gamma(5060*0.5, 1))

    length_load = numpyro.sample("lambda_P", dist.LogNormal(1.5, 0.5))
    length_alpha = numpyro.sample("lambda_alpha", dist.LogNormal(0.34, 0.5))
    length_E1 = numpyro.sample("lambda_E1", dist.LogNormal(11., 0.5))
    length_E2 = numpyro.sample("lambda_E2", dist.LogNormal(8.3, 0.5))
    length_v12 = numpyro.sample("lambda_v12", dist.LogNormal(-0.80, 0.5))
    length_v23 = numpyro.sample("lambda_v23", dist.LogNormal(-0.80, 0.5))
    length_G12 = numpyro.sample("lambda_G12", dist.LogNormal(7.7, 0.5))
    
    # hyper_parameter_emulator = jnp.array([stdev_emulator, length_load, length_alpha, 
    #                 length_E1, length_E2, length_v12, length_v23, length_G12])
    length_xy = jnp.array([length_load, length_alpha])
    length_theta = jnp.array([length_E1, length_E2, length_v12, length_v23, length_G12])

    stdev_measure = numpyro.sample("sigma_measure", dist.Exponential(100.))

    data_size_exp = [i.shape[0] for i in input_xy_exp]
    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        
        input_theta_exp = []
        for i in range(num_exp):
            # theta_b = theta
            theta_b = theta.at[0].add(bias[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i],1)))
        input_theta = jnp.concatenate( [*input_theta_exp, input_theta_sim], axis=0 ) 
        
    if add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append( jnp.array(input_xy_exp[i]) + 
                                  jnp.concatenate((jnp.zeros((data_size_exp[i],1)), bias[i]*jnp.ones((data_size_exp[i],1))), axis=1) )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)

        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    if not add_bias_E1 and not add_bias_alpha: 
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    # 
    data = jnp.concatenate((*data_exp, data_sim), axis=0)
    
    # compute covariance matrix
    cov_matrix = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    
    # add measurement noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:,0]
    diag_line = jnp.concatenate([(stdev_measure**2 * loads_exp), jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure

    # cov_matrix_measure = jnp.diag((stdev_measure * input_xy_exp[:,0])**2)
    # cov_matrix = cov_matrix.at[0:input_xy_exp.shape[0], 0:input_xy_exp.shape[0]].add(cov_matrix_measure)
    
    jitter = jnp.diag( 1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter
    
    mean_vector = mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])
    
    # sample data according to the standard gaussian process formula
    numpyro.sample(
        "data",
        dist.MultivariateNormal(loc=mean_vector, covariance_matrix=cov_matrix),
        obs=data,
    )

def model_n(input_xy_exp, input_xy_sim, input_theta_sim, data_exp, data_sim, direction='h', add_bias_E1=False, add_bias_alpha=False):
    """
    model with reparameterization

    input_xy_exp: experimental loads and angles
    input_xy_sim: simulation loads and angles
    input_theta_sim: simulation thetas (FE parameters)
    data_exp: experimental extensions
    data_sim: simulation extensions
    """

    num_exp = len(input_xy_exp)
    
    # priors of theta
    cdf_normal = dist.Normal().cdf
    
    E1_n = numpyro.sample("E_1_n", dist.Normal())
    #E1 = 148800. + 2000. * E1_n
    E1 = 161000. + 2000. * E1_n
    numpyro.deterministic('E_1', E1)
    
    E2_n = numpyro.sample("E_2_n", dist.Normal())
    #E2 = 9190. + 100. * E2_n
    E2 = 11380. + 100. * E2_n
    numpyro.deterministic('E_2', E2)

    v12_n = numpyro.sample("v_12_n", dist.Normal())
    #v12 = 0.34 + 0.01 * v12_n
    v12 = 0.32 + 0.01 * v12_n
    numpyro.deterministic('v_12', v12)
    
    v23_n = numpyro.sample("v_23_n", dist.Normal())
    v23 = 0.43 + 0.01 * v23_n
    #v23 = 0.44 + 0.01 * v23_n
    numpyro.deterministic("v_23", v23)
    
    G12_n = numpyro.sample("G_12_n", dist.Normal())
    #G12 = 5060. + 70. * G12_n
    G12 = 5170. + 70. * G12_n
    numpyro.deterministic("G_12", G12)

    theta = jnp.array([E1, E2, v12, v23, G12])

    # priors of bias
    if add_bias_E1:
        sigma_b_E1 = numpyro.sample("sigma_b_E1", dist.Exponential(0.0001))
        bias_E1 = []
        for i in range(num_exp):
            b_E1_n = numpyro.sample("b_"+str(i+1)+"_E1_n", dist.Normal())
        #   b_E1 = 0. + 1000. * b_E1_n
            b_E1 = 0. + sigma_b_E1 * b_E1_n
            numpyro.deterministic("b_"+str(i+1)+"_E1", b_E1)
            bias_E1.append(b_E1)
    
    if add_bias_alpha:
        sigma_b_alpha = numpyro.sample("sigma_b_alpha", dist.Exponential(1/np.deg2rad(5))) 
        bias_alpha = []
        for i in range(num_exp):
            b_alpha_n = numpyro.sample("b_"+str(i+1)+"_alpha_n", dist.Normal())
            #b_alpha = 0. + np.deg2rad(1) * b_alpha_n
            b_alpha = 0. + sigma_b_alpha * b_alpha_n
            numpyro.deterministic("b_"+str(i+1)+"_alpha", b_alpha)
            bias_alpha.append(b_alpha)

    # priors of hyper-parameters
    mean_emulator_n = numpyro.sample("mu_emulator_n", dist.Normal())
    mean_emulator = 0. + 0.01 * mean_emulator_n
    numpyro.deterministic("mu_emulator", mean_emulator)

    stdev_emulator_n = numpyro.sample("sigma_emulator_n", dist.Normal())
    stdev_emulator = dist.Exponential(20.).icdf(cdf_normal(stdev_emulator_n))
    numpyro.deterministic("sigma_emulator", stdev_emulator)
    
    length_load_n = numpyro.sample("lambda_P_n", dist.Normal())
    # length_load = dist.Gamma(5., 1.).icdf(cdf_normal(length_load_n))
    length_load = jnp.exp(1.5 + 0.5*length_load_n)
    numpyro.deterministic("lambda_P", length_load)
    
    length_alpha_n = numpyro.sample("lambda_alpha_n", dist.Normal())
    # length_alpha = dist.Gamma(jnp.pi/2, 1.).icdf(cdf_normal(length_alpha_n))
    length_alpha = jnp.exp(0.34 + 0.5 * length_alpha_n)
    numpyro.deterministic("lambda_alpha", length_alpha)

    length_E1_n = numpyro.sample("lambda_E1_n", dist.Normal())
    # length_E1 = dist.Gamma(148800*0.5, 1.).icdf(cdf_normal(length_E1_n))
    length_E1 = jnp.exp(11. + 0.5 * length_E1_n)
    numpyro.deterministic("lambda_E1", length_E1)
                          
    length_E2_n = numpyro.sample("lambda_E2_n", dist.Normal())
    # length_E2 = dist.Gamma(9190*0.5, 1.).icdf(cdf_normal(length_E2_n))
    length_E2 = jnp.exp(8.3 + 0.5 * length_E2_n)
    numpyro.deterministic("lambda_E2", length_E2)
    
    length_v12_n = numpyro.sample("lambda_v12_n", dist.Normal())
    # length_v12 = dist.Gamma(1., 2.).icdf(cdf_normal(length_v12_n))
    length_v12 = jnp.exp(-0.80 + 0.5 * length_v12_n)
    numpyro.deterministic("lambda_v12", length_v12)

    length_v23_n = numpyro.sample("lambda_v23_n", dist.Normal())
    # length_v23 = dist.Gamma(1., 2.).icdf(cdf_normal(length_v23_n))
    length_v23 = jnp.exp(-0.80 + 0.5 * length_v23_n)
    numpyro.deterministic("lambda_v23", length_v23)

    length_G12_n = numpyro.sample("lambda_G12_n", dist.Normal())
    # length_G12 = dist.Gamma(5060*0.5, 1.).icdf(cdf_normal(length_G12_n))
    length_G12 = jnp.exp(7.7 + 0.5 * length_G12_n)
    numpyro.deterministic("lambda_G12", length_G12)

    # hyper_parameter_emulator = jnp.array([stdev_emulator, length_load, length_alpha, 
    #                 length_E1, length_E2, length_v12, length_v23, length_G12])
    length_xy = jnp.array([length_load, length_alpha])
    length_theta = jnp.array([length_E1, length_E2, length_v12, length_v23, length_G12])

    stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
    stdev_measure = dist.Exponential(100.).icdf(cdf_normal(stdev_measure_n))
    stdev_measure = numpyro.deterministic("sigma_measure", stdev_measure)

    data_size_exp = [i.shape[0] for i in input_xy_exp]
    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        
        input_theta_exp = []
        for i in range(num_exp):
            # theta_b = theta
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i],1)))
        input_theta = jnp.concatenate( [*input_theta_exp, input_theta_sim], axis=0 ) 
        
    if add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append( jnp.array(input_xy_exp[i]) + 
                                  jnp.concatenate((jnp.zeros((data_size_exp[i],1)), bias_alpha[i]*jnp.ones((data_size_exp[i],1))), axis=1) )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        
        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    if not add_bias_E1 and not add_bias_alpha: 
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    # 
    data = jnp.concatenate((*data_exp, data_sim), axis=0)
    
    # compute covariance matrix
    cov_matrix = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    
    # add measurement noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:,0]
    diag_line = jnp.concatenate([(stdev_measure**2 * loads_exp), jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure

    # cov_matrix_measure = jnp.diag((stdev_measure * input_xy_exp[:,0])**2)
    # cov_matrix = cov_matrix.at[0:input_xy_exp.shape[0], 0:input_xy_exp.shape[0]].add(cov_matrix_measure)
    
    jitter = jnp.diag( 1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter

    if direction == 'h':
        mean_vector = mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])
    elif direction == 'v':
        mean_vector = mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])
    
    # sample data according to the standard gaussian process formula
    numpyro.sample(
        "data",
        dist.MultivariateNormal(loc=mean_vector, covariance_matrix=cov_matrix),
        obs=data,
    )

def model_n_hv(input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, add_bias_E1=False, add_bias_alpha=False):
    """
    model with reparameterization with both horizontal (shear extension) and vertical (noraml extension) data

    input_xy_exp: experimental loads and angles
    input_xy_sim: simulation loads and angles
    input_theta_sim: simulation thetas (FE parameters)
    data_exp_(h,v   ): horizontal or vertical experimental extensions
    data_sim_(h,v): horizontal or vertical simulation extensions
    """

    num_exp = len(input_xy_exp)
    
    # priors of theta
    cdf_normal = dist.Normal().cdf
    
    E1_n = numpyro.sample("E_1_n", dist.Normal())
    #E1 = 148800. + 2000. * E1_n
    #E1 = 161000. + 4000. * E1_n
    E1 = 165000. + 6050. * E1_n
    numpyro.deterministic('E_1', E1)
    
    E2_n = numpyro.sample("E_2_n", dist.Normal())
    #E2 = 9190. + 200. * E2_n
    #E2 = 11380. + 200. * E2_n
    E2 = 11500. + 250. * E2_n
    numpyro.deterministic('E_2', E2)

    v12_n = numpyro.sample("v_12_n", dist.Normal())
    #v12 = 0.34 + 0.01 * v12_n
    #v12 = 0.32 + 0.01 * v12_n
    v12 = 0.36 + 0.005 * v12_n
    numpyro.deterministic('v_12', v12)
    
    v23_n = numpyro.sample("v_23_n", dist.Normal())
    #v23 = 0.44 + 0.01 * v23_n
    #v23 = 0.43 + 0.01 * v23_n
    v23 = 0.41 + 0.01 * v23_n
    numpyro.deterministic("v_23", v23)
    
    G12_n = numpyro.sample("G_12_n", dist.Normal())
    #G12 = 5060. + 70. * G12_n
    #G12 = 5170. + 70. * G12_n
    G12 = 5000. + 80. * G12_n
    numpyro.deterministic("G_12", G12)

    theta = jnp.array([E1, E2, v12, v23, G12])

    # priors of bias
    if add_bias_E1:
        sigma_b_E1 = numpyro.sample("sigma_b_E1", dist.Exponential(0.0001))
        bias_E1 = []
        for i in range(num_exp):
            b_E1_n = numpyro.sample("b_"+str(i+1)+"_E1_n", dist.Normal())
            #b_E1 = 0. + 1000. * b_E1_n
            #b_E1 = 0. + 1500. * b_E1_n
            b_E1 = 0. + sigma_b_E1 * b_E1_n
            numpyro.deterministic("b_"+str(i+1)+"_E1", b_E1)
            bias_E1.append(b_E1)
    
    if add_bias_alpha:
        sigma_b_alpha = numpyro.sample("sigma_b_alpha", dist.Exponential(1/np.deg2rad(10))) 
        bias_alpha = []
        for i in range(num_exp):
            b_alpha_n = numpyro.sample("b_"+str(i+1)+"_alpha_n", dist.Normal())
            #b_alpha = 0. + np.deg2rad(1) * b_alpha_n
            b_alpha = 0. + sigma_b_alpha * b_alpha_n
            numpyro.deterministic("b_"+str(i+1)+"_alpha", b_alpha)
            bias_alpha.append(b_alpha)

    # priors of hyper-parameters
    mean_emulator_n = numpyro.sample("mu_emulator_n", dist.Normal())
    mean_emulator = 0. + 0.01 * mean_emulator_n
    numpyro.deterministic("mu_emulator", mean_emulator)

    stdev_emulator_n = numpyro.sample("sigma_emulator_n", dist.Normal())
    stdev_emulator = dist.Exponential(20.).icdf(cdf_normal(stdev_emulator_n))
    numpyro.deterministic("sigma_emulator", stdev_emulator)
    
    length_load_n = numpyro.sample("lambda_P_n", dist.Normal())
    # length_load = dist.Gamma(5., 1.).icdf(cdf_normal(length_load_n))
    length_load = jnp.exp(1.5 + 0.5*length_load_n)
    numpyro.deterministic("lambda_P", length_load)
    
    length_alpha_n = numpyro.sample("lambda_alpha_n", dist.Normal())
    # length_alpha = dist.Gamma(jnp.pi/2, 1.).icdf(cdf_normal(length_alpha_n))
    length_alpha = jnp.exp(0.34 + 0.5 * length_alpha_n)
    numpyro.deterministic("lambda_alpha", length_alpha)

    length_E1_n = numpyro.sample("lambda_E1_n", dist.Normal())
    # length_E1 = dist.Gamma(148800*0.5, 1.).icdf(cdf_normal(length_E1_n))
    length_E1 = jnp.exp(11. + 0.5 * length_E1_n)
    numpyro.deterministic("lambda_E1", length_E1)
                          
    length_E2_n = numpyro.sample("lambda_E2_n", dist.Normal())
    # length_E2 = dist.Gamma(9190*0.5, 1.).icdf(cdf_normal(length_E2_n))
    length_E2 = jnp.exp(8.3 + 0.5 * length_E2_n)
    numpyro.deterministic("lambda_E2", length_E2)
    
    length_v12_n = numpyro.sample("lambda_v12_n", dist.Normal())
    # length_v12 = dist.Gamma(1., 2.).icdf(cdf_normal(length_v12_n))
    length_v12 = jnp.exp(-0.80 + 0.5 * length_v12_n)
    numpyro.deterministic("lambda_v12", length_v12)

    length_v23_n = numpyro.sample("lambda_v23_n", dist.Normal())
    # length_v23 = dist.Gamma(1., 2.).icdf(cdf_normal(length_v23_n))
    length_v23 = jnp.exp(-0.80 + 0.5 * length_v23_n)
    numpyro.deterministic("lambda_v23", length_v23)

    length_G12_n = numpyro.sample("lambda_G12_n", dist.Normal())
    # length_G12 = dist.Gamma(5060*0.5, 1.).icdf(cdf_normal(length_G12_n))
    length_G12 = jnp.exp(7.7 + 0.5 * length_G12_n)
    numpyro.deterministic("lambda_G12", length_G12)

    # hyper_parameter_emulator = jnp.array([stdev_emulator, length_load, length_alpha, 
    #                 length_E1, length_E2, length_v12, length_v23, length_G12])
    length_xy = jnp.array([length_load, length_alpha])
    length_theta = jnp.array([length_E1, length_E2, length_v12, length_v23, length_G12])

    stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
    stdev_measure = dist.Exponential(100.).icdf(cdf_normal(stdev_measure_n))
    stdev_measure = numpyro.deterministic("sigma_measure", stdev_measure)

    data_size_exp = [i.shape[0] for i in input_xy_exp]
    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        
        input_theta_exp = []
        for i in range(num_exp):
            # theta_b = theta
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i],1)))
        input_theta = jnp.concatenate( [*input_theta_exp, input_theta_sim], axis=0 ) 
        
    if add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append( jnp.array(input_xy_exp[i]) + 
                                  jnp.concatenate((jnp.zeros((data_size_exp[i],1)), bias_alpha[i]*jnp.ones((data_size_exp[i],1))), axis=1) )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        
        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    if not add_bias_E1 and not add_bias_alpha: 
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        # theta = jnp.array(theta)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    # 
    data_h = jnp.concatenate((*data_exp_h, data_sim_h), axis=0)
    data_v = jnp.concatenate((*data_exp_v, data_sim_v), axis=0)
    
    # compute covariance matrix
    cov_matrix = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    
    # add measurement noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:,0]
    diag_line = jnp.concatenate([(stdev_measure**2 * loads_exp), jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure

    # cov_matrix_measure = jnp.diag((stdev_measure * input_xy_exp[:,0])**2)
    # cov_matrix = cov_matrix.at[0:input_xy_exp.shape[0], 0:input_xy_exp.shape[0]].add(cov_matrix_measure)
    
    jitter = jnp.diag( 1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter

    mean_vector_h = mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])
    mean_vector_v = mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])
    
    # sample data according to the standard gaussian process formula
    numpyro.sample(
        "data_h",
        dist.MultivariateNormal(loc=mean_vector_h, covariance_matrix=cov_matrix),
        obs=data_h,
    )

    numpyro.sample(
        "data_v",
        dist.MultivariateNormal(loc=mean_vector_v, covariance_matrix=cov_matrix),
        obs=data_v,
    )

def run_inference_hv(model, rng_key, input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, add_bias_E1=False, add_bias_alpha=False, num_warmup=1000, num_samples=1000):
    """
    function to run the inference with both horizontal and vertical data 
    """
    # start = time.time()
    init_strategy = init_to_median(num_samples=30)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=2,
        thinning=2,
        progress_bar=True,
    )
    mcmc.run(rng_key, input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, add_bias_E1, add_bias_alpha)
    mcmc.print_summary()
    # print("\nMCMC elapsed time:", time.time() - start)
    return mcmc


def posterior_predict(rng_key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, data_exp, data_sim, test_xy, test_theta, 
            mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, direction='h'):
   
    # compute kernels between train and test data, etc.
    input_xy = jnp.concatenate((input_xy_exp, input_xy_sim, test_xy[0][None,:]), axis=0)
    # input_theta_exp = jnp.tile(test_theta, (input_xy_exp.shape[0],1))
    input_theta = jnp.concatenate((input_theta_exp, input_theta_sim, test_theta[None,:]), axis=0)

    # covariance matrix for data
    cov_matrix_data = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta) 
    diag_line = jnp.concatenate([(stdev_measure**2 * input_xy_exp[:,0]), jnp.zeros(input_xy_sim.shape[0]+1)])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix_data += cov_matrix_measure
    cov_matrix_data += jnp.diag(jnp.ones(cov_matrix_data.shape[0]) * 1e-10)

    # covariance matrix for the interpolation points
    test_theta_p = jnp.tile(test_theta, (test_xy.shape[0],1))
    cov_matrix_interp = cov_matrix_emulator(test_xy, test_theta_p, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)

    # covariance matrix between data and interpolation points
    cov_matrix_data_interp = cov_matrix_emulator(input_xy, input_theta, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)
    # cov_matrix_interp_data = cov_matrix_emulator(test_xy, test_theta_p, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    cov_matrix_interp_data = cov_matrix_data_interp.T
    
    # cov_matrix_data_inv = jnp.linalg.inv(cov_matrix_data)
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ (cov_matrix_data_inv @ (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))
    # linear mean function
    if direction == 'h':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.sin(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])))
    elif direction == 'v':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.cos(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])))

    # constant mean function
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))

    # cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.matmul(cov_matrix_data_inv, cov_matrix_data_interp))
    cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.linalg.solve(cov_matrix_data, cov_matrix_data_interp))
    stdev_post_emulator = jnp.sqrt(jnp.clip(jnp.diag(cov_post_emulator), a_min=0.0))
    
    L = jnp.linalg.cholesky(cov_post_emulator + jnp.diag(jnp.ones(test_xy.shape[0]) * 1e-10))
    white_noise = random.normal(rng_key, (test_xy.shape[0],))
    sample_post = L @ white_noise + mean_post_emulator
    # we return both the mean function, standard deviation and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_post_emulator, stdev_post_emulator, sample_post

def prior_predict(rng_key, input_xy_sim, input_theta_sim, data_sim, test_xy, test_theta, 
            mean_emulator, stdev_emulator, length_xy, length_theta, direction='h'):
   
    # compute kernels between train and test data, etc.
    input_xy = jnp.concatenate((input_xy_sim, test_xy[0][None,:]), axis=0)
    # input_theta_exp = jnp.tile(test_theta, (input_xy_exp.shape[0],1))
    input_theta = jnp.concatenate((input_theta_sim, test_theta[None,:]), axis=0)

    # covariance matrix for data
    cov_matrix_data = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta) 
    cov_matrix_data += jnp.diag(jnp.ones(cov_matrix_data.shape[0]) * 1e-10)

    # covariance matrix for the interpolation points
    test_theta_p = jnp.tile(test_theta, (test_xy.shape[0],1))
    cov_matrix_interp = cov_matrix_emulator(test_xy, test_theta_p, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)

    # covariance matrix between data and interpolation points
    cov_matrix_data_interp = cov_matrix_emulator(input_xy, input_theta, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)
    # cov_matrix_interp_data = cov_matrix_emulator(test_xy, test_theta_p, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    cov_matrix_interp_data = cov_matrix_data_interp.T
    
    # cov_matrix_data_inv = jnp.linalg.inv(cov_matrix_data)
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ (cov_matrix_data_inv @ (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))
    # linear mean function
    if direction == 'h':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.sin(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_sim, np.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])))
    elif direction == 'v':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.cos(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_sim, np.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])))

    # constant mean function 
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_sim, np.zeros(1)), axis=0) - mean_emulator))

    # cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.matmul(cov_matrix_data_inv, cov_matrix_data_interp))
    cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.linalg.solve(cov_matrix_data, cov_matrix_data_interp))
    stdev_post_emulator = jnp.sqrt(jnp.clip(jnp.diag(cov_post_emulator), a_min=0.0))
    
    L = jnp.linalg.cholesky(cov_post_emulator + jnp.diag(jnp.ones(test_xy.shape[0]) * 1e-10))
    white_noise = random.normal(rng_key, (test_xy.shape[0],))
    sample_post = L @ white_noise + mean_post_emulator
    # we return both the mean function, standard deviation and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_post_emulator, stdev_post_emulator, sample_post



    