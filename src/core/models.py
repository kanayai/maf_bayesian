import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from .covariance import cov_matrix_emulator

def get_priors_from_config(config, num_exp):
    """
    Helper to extract and sample priors based on configuration.
    Returns dictionaries of sampled values.
    """
    priors = config["priors"]
    bias_flags = config["bias"]
    
    # --- Physical Parameters (Theta) ---
    theta_vals = []
    # Order matters: E1, E2, v12, v23, G12
    param_names = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    
    if "n" in config["model_type"]: # Reparameterized models
        reparam_cfg = priors["theta"]["reparam"]
        for name in param_names:
            n_sample = numpyro.sample(f"{name}_n", dist.Normal())
            val = reparam_cfg[name]["mean"] + reparam_cfg[name]["scale"] * n_sample
            numpyro.deterministic(name, val)
            theta_vals.append(val)
    else: # Standard models
        std_cfg = priors["theta"]["standard"]
        for name in param_names:
            val = numpyro.sample(name, std_cfg[name])
            theta_vals.append(val)
            
    theta = jnp.array(theta_vals)

    # --- Bias ---
    bias_E1 = []
    bias_alpha = []
    
    if bias_flags["add_bias_E1"]:
        sigma_b_E1 = numpyro.sample("sigma_b_E1", priors["bias_priors"]["sigma_b_E1"])
        for i in range(num_exp):
            b_E1_n = numpyro.sample(f"b_{i+1}_E1_n", dist.Normal())
            b_E1 = 0. + sigma_b_E1 * b_E1_n
            numpyro.deterministic(f"b_{i+1}_E1", b_E1)
            bias_E1.append(b_E1)
            
    if bias_flags["add_bias_alpha"]:
        sigma_b_alpha = numpyro.sample("sigma_b_alpha", priors["bias_priors"]["sigma_b_alpha"])
        for i in range(num_exp):
            b_alpha_n = numpyro.sample(f"b_{i+1}_alpha_n", dist.Normal())
            b_alpha = 0. + sigma_b_alpha * b_alpha_n
            numpyro.deterministic(f"b_{i+1}_alpha", b_alpha)
            bias_alpha.append(b_alpha)

    # --- Hyperparameters ---
    hyper = priors["hyper"]
    cdf_normal = dist.Normal().cdf

    # Emulator Mean
    mean_emulator_n = numpyro.sample("mu_emulator_n", dist.Normal())
    mean_emulator = hyper["mu_emulator"]["mean"] + hyper["mu_emulator"]["scale"] * mean_emulator_n
    numpyro.deterministic("mu_emulator", mean_emulator)

    # Emulator Stdev
    stdev_emulator_n = numpyro.sample("sigma_emulator_n", dist.Normal())
    stdev_emulator = hyper["sigma_emulator"]["target_dist"].icdf(cdf_normal(stdev_emulator_n))
    numpyro.deterministic("sigma_emulator", stdev_emulator)

    # Length Scales
    ls_cfg = hyper["length_scales"]
    ls_names = ["lambda_P", "lambda_alpha", "lambda_E1", "lambda_E2", "lambda_v12", "lambda_v23", "lambda_G12"]
    ls_vals = {}
    
    for name in ls_names:
        n_sample = numpyro.sample(f"{name}_n", dist.Normal())
        # Log-normal like reparameterization: val = exp(mean + scale * n)
        val = jnp.exp(ls_cfg[name]["mean"] + ls_cfg[name]["scale"] * n_sample)
        numpyro.deterministic(name, val)
        ls_vals[name] = val

    length_xy = jnp.array([ls_vals["lambda_P"], ls_vals["lambda_alpha"]])
    length_theta = jnp.array([ls_vals["lambda_E1"], ls_vals["lambda_E2"], ls_vals["lambda_v12"], ls_vals["lambda_v23"], ls_vals["lambda_G12"]])

    # Measurement Noise
    stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
    stdev_measure = hyper["sigma_measure"]["target_dist"].icdf(cdf_normal(stdev_measure_n))
    numpyro.deterministic("sigma_measure", stdev_measure)

    return theta, bias_E1, bias_alpha, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure


def model_n_hv(input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, config):
    """
    Model with reparameterization with both horizontal and vertical data.
    Uses config for priors and settings.
    """
    num_exp = len(input_xy_exp)
    add_bias_E1 = config["bias"]["add_bias_E1"]
    add_bias_alpha = config["bias"]["add_bias_alpha"]

    # Get all priors
    theta, bias_E1, bias_alpha, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure = get_priors_from_config(config, num_exp)

    # Prepare inputs based on bias
    data_size_exp = [i.shape[0] for i in input_xy_exp]
    
    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta_exp = []
        for i in range(num_exp):
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i],1)))
        input_theta = jnp.concatenate( [*input_theta_exp, input_theta_sim], axis=0 ) 
        
    elif add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append( jnp.array(input_xy_exp[i]) + 
                                  jnp.concatenate((jnp.zeros((data_size_exp[i],1)), bias_alpha[i]*jnp.ones((data_size_exp[i],1))), axis=1) )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    else: # No bias
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta = jnp.concatenate( [jnp.tile( theta, (sum(data_size_exp),1) ), input_theta_sim], axis=0 ) 
        
    # Prepare Data
    data_h = jnp.concatenate((*data_exp_h, data_sim_h), axis=0)
    data_v = jnp.concatenate((*data_exp_v, data_sim_v), axis=0)
    
    # Compute Covariance Matrix
    cov_matrix = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    
    # Add Measurement Noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:,0]
    diag_line = jnp.concatenate([(stdev_measure**2 * loads_exp), jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure
    
    # Jitter
    jitter = jnp.diag( 1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter

    # Mean Vectors
    mean_vector_h = mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])
    mean_vector_v = mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])
    
    # Sample Data
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

def posterior_predict(rng_key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, data_exp, data_sim, test_xy, test_theta, 
            mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, direction='h'):
   
    # compute kernels between train and test data, etc.
    input_xy = jnp.concatenate((input_xy_exp, input_xy_sim, test_xy[0][None,:]), axis=0)
    # input_theta_exp = jnp.tile(test_theta, (input_xy_exp.shape[0],1))
    input_theta = jnp.concatenate((input_theta_exp, input_theta_sim, test_theta[None,:]), axis=0)

    # covariance matrix for data
    cov_matrix_data = cov_matrix_emulator(input_xy, input_theta, input_xy, input_theta, stdev_emulator, length_xy, length_theta) 
    # Cast shape to int to avoid TracerIntegerConversionError
    zeros_padding = jnp.zeros(int(input_xy_sim.shape[0]) + 1)
    diag_line = jnp.concatenate([(stdev_measure**2 * input_xy_exp[:,0]), zeros_padding])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix_data += cov_matrix_measure
    cov_matrix_data += jnp.diag(jnp.ones(cov_matrix_data.shape[0]) * 1e-10)

    # covariance matrix for the interpolation points
    # Cast shape to int
    test_theta_p = jnp.tile(test_theta, (int(test_xy.shape[0]), 1))
    cov_matrix_interp = cov_matrix_emulator(test_xy, test_theta_p, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)

    # covariance matrix between data and interpolation points
    cov_matrix_data_interp = cov_matrix_emulator(input_xy, input_theta, test_xy, test_theta_p, stdev_emulator, length_xy, length_theta)
    # cov_matrix_interp_data = cov_matrix_emulator(test_xy, test_theta_p, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    cov_matrix_interp_data = cov_matrix_data_interp.T
    
    # cov_matrix_data_inv = jnp.linalg.inv(cov_matrix_data)
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ (cov_matrix_data_inv @ (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))
    # linear mean function
    if direction == 'h':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.sin(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, jnp.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.sin(input_xy[:,1])))
    elif direction == 'v':
        mean_post_emulator = mean_emulator * test_xy[:,0] * jnp.cos(test_xy[:,1]) + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, jnp.zeros(1)), axis=0) - mean_emulator * input_xy[:,0] * jnp.cos(input_xy[:,1])))

    # constant mean function
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))

    # cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.matmul(cov_matrix_data_inv, cov_matrix_data_interp))
    cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.linalg.solve(cov_matrix_data, cov_matrix_data_interp))
    stdev_post_emulator = jnp.sqrt(jnp.clip(jnp.diag(cov_post_emulator), a_min=0.0))
    
    L = jnp.linalg.cholesky(cov_post_emulator + jnp.diag(jnp.ones(test_xy.shape[0]) * 1e-10))
    # Need random here, so we need to import it or pass it
    # The function signature has rng_key
    import jax.random as random
    white_noise = random.normal(rng_key, (test_xy.shape[0],))
    sample_post = L @ white_noise + mean_post_emulator
    # we return both the mean function, standard deviation and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_post_emulator, stdev_post_emulator, sample_post
