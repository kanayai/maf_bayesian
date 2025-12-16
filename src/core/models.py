import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from .covariance import cov_matrix_emulator


def get_priors_from_config(config, num_exp):
    """
    Helper to extract and sample priors based on configuration.
    Returns dictionaries of sampled values.
    """
    priors = config["priors"]
    priors = config["priors"]
    bias_flags = config["bias"]
    model_type = config.get("model_type", "model_n")

    # --- Physical Parameters (Theta) ---
    theta_vals = []
    # Order matters: E1, E2, v12, v23, G12
    param_names = ["E_1", "E_2", "v_12", "v_23", "G_12"]

    if "n" not in model_type and "hierarchical" not in model_type:
        raise ValueError(
            "Only reparameterized models (model_type='model_n' or 'model_n_hv' or 'model_hierarchical') are supported."
        )

    # Reparameterized models
    reparam_cfg = priors["theta"]["reparam"]
    for name in param_names:
        n_sample = numpyro.sample(f"{name}_n", dist.Normal())
        val = reparam_cfg[name]["mean"] + reparam_cfg[name]["scale"] * n_sample
        numpyro.deterministic(name, val)
        theta_vals.append(val)

    theta = jnp.array(theta_vals)

    # --- Bias ---
    bias_E1 = []
    bias_alpha = []

    if bias_flags["add_bias_E1"]:
        sigma_b_E1 = numpyro.sample("sigma_b_E1", priors["bias_priors"]["sigma_b_E1"])
        for i in range(num_exp):
            b_E1_n = numpyro.sample(f"b_{i + 1}_E1_n", dist.Normal())
            b_E1 = 0.0 + sigma_b_E1 * b_E1_n
            numpyro.deterministic(f"b_{i + 1}_E1", b_E1)
            bias_E1.append(b_E1)

    if bias_flags["add_bias_alpha"]:
        sigma_b_alpha = numpyro.sample(
            "sigma_b_alpha", priors["bias_priors"]["sigma_b_alpha"]
        )
        for i in range(num_exp):
            b_alpha_n = numpyro.sample(f"b_{i + 1}_alpha_n", dist.Normal())
            b_alpha = 0.0 + sigma_b_alpha * b_alpha_n
            numpyro.deterministic(f"b_{i + 1}_alpha", b_alpha)
            bias_alpha.append(b_alpha)

    # --- Hyperparameters ---
    hyper = priors["hyper"]
    cdf_normal = dist.Normal().cdf

    # Emulator Mean
    mean_emulator_n = numpyro.sample("mu_emulator_n", dist.Normal())
    mu_cfg = hyper["mu_emulator"]
    if "log_mean" in mu_cfg:
        # LogNormal reparameterization: val = exp(log_mean + log_scale * n)
        mean_emulator = jnp.exp(mu_cfg["log_mean"] + mu_cfg["log_scale"] * mean_emulator_n)
    else:
        # Normal reparameterization: val = mean + scale * n
        mean_emulator = mu_cfg["mean"] + mu_cfg["scale"] * mean_emulator_n
    numpyro.deterministic("mu_emulator", mean_emulator)

    # Emulator Stdev
    stdev_emulator_n = numpyro.sample("sigma_emulator_n", dist.Normal())
    sig_em_cfg = hyper["sigma_emulator"]
    if "log_mean" in sig_em_cfg:
        # LogNormal reparameterization: val = exp(log_mean + log_scale * n)
        stdev_emulator = jnp.exp(sig_em_cfg["log_mean"] + sig_em_cfg["log_scale"] * stdev_emulator_n)
    else:
        # icdf transform for target_dist
        stdev_emulator = sig_em_cfg["target_dist"].icdf(cdf_normal(stdev_emulator_n))
    numpyro.deterministic("sigma_emulator", stdev_emulator)

    # Measurement Noise - sample only parameters relevant to chosen noise model
    noise_model = config["data"].get("noise_model", "proportional")

    stdev_measure = 0.0
    stdev_measure_base = 0.0
    stdev_constant = 0.0

    if noise_model == "constant":
        # Constant model: only sample sigma_constant
        stdev_constant_n = numpyro.sample("sigma_constant_n", dist.Normal())
        sig_const_cfg = hyper["sigma_constant"]
        if "log_mean" in sig_const_cfg:
             stdev_constant = jnp.exp(sig_const_cfg["log_mean"] + sig_const_cfg["log_scale"] * stdev_constant_n)
        else:
            stdev_constant = sig_const_cfg["target_dist"].icdf(
                cdf_normal(stdev_constant_n)
            )
        numpyro.deterministic("sigma_constant", stdev_constant)
    elif noise_model == "additive":
        # Additive model: sample sigma_measure and sigma_measure_base
        stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
        sig_meas_cfg = hyper["sigma_measure"]
        if "log_mean" in sig_meas_cfg:
            stdev_measure = jnp.exp(sig_meas_cfg["log_mean"] + sig_meas_cfg["log_scale"] * stdev_measure_n)
        else:
            stdev_measure = sig_meas_cfg["target_dist"].icdf(cdf_normal(stdev_measure_n))
        numpyro.deterministic("sigma_measure", stdev_measure)

        stdev_measure_base_n = numpyro.sample("sigma_measure_base_n", dist.Normal())
        sig_base_cfg = hyper["sigma_measure_base"]
        if "log_mean" in sig_base_cfg:
            stdev_measure_base = jnp.exp(sig_base_cfg["log_mean"] + sig_base_cfg["log_scale"] * stdev_measure_base_n)
        else:
            stdev_measure_base = sig_base_cfg["target_dist"].icdf(
                cdf_normal(stdev_measure_base_n)
            )
        numpyro.deterministic("sigma_measure_base", stdev_measure_base)
    elif noise_model == "prop_quad":
        # Quadratic Proportional model: sample sigma_measure
        stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
        sig_meas_cfg = hyper["sigma_measure"]
        if "log_mean" in sig_meas_cfg:
             stdev_measure = jnp.exp(sig_meas_cfg["log_mean"] + sig_meas_cfg["log_scale"] * stdev_measure_n)
        else:
            stdev_measure = sig_meas_cfg["target_dist"].icdf(cdf_normal(stdev_measure_n))
        numpyro.deterministic("sigma_measure", stdev_measure)
    else:
        # Proportional model (default): only sample sigma_measure
        stdev_measure_n = numpyro.sample("sigma_measure_n", dist.Normal())
        sig_meas_cfg = hyper["sigma_measure"]
        if "log_mean" in sig_meas_cfg:
            stdev_measure = jnp.exp(sig_meas_cfg["log_mean"] + sig_meas_cfg["log_scale"] * stdev_measure_n)
        else:
            stdev_measure = sig_meas_cfg["target_dist"].icdf(cdf_normal(stdev_measure_n))
        numpyro.deterministic("sigma_measure", stdev_measure)

    # Length Scales
    ls_cfg = hyper["length_scales"]
    ls_names = [
        "lambda_P",
        "lambda_alpha",
        "lambda_E1",
        "lambda_E2",
        "lambda_v12",
        "lambda_v23",
        "lambda_G12",
    ]
    ls_vals = {}

    for name in ls_names:
        n_sample = numpyro.sample(f"{name}_n", dist.Normal())
        # LogNormal reparameterization: val = exp(log_mean + log_scale * n)
        val = jnp.exp(ls_cfg[name]["log_mean"] + ls_cfg[name]["log_scale"] * n_sample)
        numpyro.deterministic(name, val)
        ls_vals[name] = val

    length_xy = jnp.array([ls_vals["lambda_P"], ls_vals["lambda_alpha"]])
    
    # Latent Variance (for hierarchical model)
    sigma_latent = 0.0
    if "sigma_latent" in hyper:
        sl_n = numpyro.sample("sigma_latent_n", dist.Normal())
        sl_cfg = hyper["sigma_latent"]
        if "log_mean" in sl_cfg:
             sigma_latent = jnp.exp(sl_cfg["log_mean"] + sl_cfg["log_scale"] * sl_n)
        elif "target_dist" in sl_cfg:
             sigma_latent = sl_cfg["target_dist"].icdf(cdf_normal(sl_n))
        numpyro.deterministic("sigma_latent", sigma_latent)
    length_theta = jnp.array(
        [
            ls_vals["lambda_E1"],
            ls_vals["lambda_E2"],
            ls_vals["lambda_v12"],
            ls_vals["lambda_v23"],
            ls_vals["lambda_G12"],
        ]
    )

    return (
        theta,
        bias_E1,
        bias_alpha,
        mean_emulator,
        stdev_emulator,
        length_xy,
        length_theta,
        stdev_measure,
        stdev_measure_base,
        stdev_constant,
        sigma_latent,
    )


def model_n(
    input_xy_exp,
    input_xy_sim,
    input_theta_sim,
    data_exp,
    data_sim,
    config,
):
    """
    Model with reparameterization for single direction (horizontal or vertical).
    Uses config for priors and settings.
    """
    num_exp = len(input_xy_exp)
    add_bias_E1 = config["bias"]["add_bias_E1"]
    add_bias_alpha = config["bias"]["add_bias_alpha"]
    direction = config["data"].get("direction", "h") # Default to horizontal if not specified

    # Get all priors
    (
        theta,
        bias_E1,
        bias_alpha,
        mean_emulator,
        stdev_emulator,
        length_xy,
        length_theta,
        stdev_measure,
        stdev_measure_base,
        stdev_constant,
        _, # sigma_latent unused
    ) = get_priors_from_config(config, num_exp)

    # Prepare inputs based on bias
    data_size_exp = [i.shape[0] for i in input_xy_exp]

    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta_exp = []
        for i in range(num_exp):
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i], 1)))
        input_theta = jnp.concatenate([*input_theta_exp, input_theta_sim], axis=0)

    elif add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append(
                jnp.array(input_xy_exp[i])
                + jnp.concatenate(
                    (
                        jnp.zeros((data_size_exp[i], 1)),
                        bias_alpha[i] * jnp.ones((data_size_exp[i], 1)),
                    ),
                    axis=1,
                )
            )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        input_theta = jnp.concatenate(
            [jnp.tile(theta, (sum(data_size_exp), 1)), input_theta_sim], axis=0
        )

    else:  # No bias
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta = jnp.concatenate(
            [jnp.tile(theta, (sum(data_size_exp), 1)), input_theta_sim], axis=0
        )

    # Prepare Data
    data = jnp.concatenate((*data_exp, data_sim), axis=0)

    # Compute Covariance Matrix
    cov_matrix = cov_matrix_emulator(
        input_xy,
        input_theta,
        input_xy,
        input_theta,
        stdev_emulator,
        length_xy,
        length_theta,
    )

    # Add Measurement Noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:, 0]

    # Noise model selection
    noise_model = config["data"].get("noise_model", "proportional")

    if noise_model == "additive":
        # sigma^2 = sigma_measure^2 * P + sigma_base^2
        noise_diag = stdev_measure**2 * loads_exp + stdev_measure_base**2
    elif noise_model == "constant":
        # sigma^2 = sigma_constant^2 (constant variance)
        num_exp_points = len(loads_exp)
        noise_diag = stdev_constant**2 * jnp.ones(num_exp_points)
    elif noise_model == "prop_quad":
        # sigma^2 = sigma_measure^2 * P^2
        noise_diag = stdev_measure**2 * loads_exp**2
    else:
        # proportional: sigma^2 = sigma_measure^2 * P
        noise_diag = stdev_measure**2 * loads_exp

    diag_line = jnp.concatenate([noise_diag, jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure

    # Jitter
    jitter = jnp.diag(1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter

    # Mean Vectors
    if direction == "h":
        mean_vector = mean_emulator * input_xy[:, 0] * jnp.sin(input_xy[:, 1])
    else: # direction == "v"
        mean_vector = mean_emulator * input_xy[:, 0] * jnp.cos(input_xy[:, 1])

    # Sample Data
    numpyro.sample(
        "data", # Single output variable for single direction model
        dist.MultivariateNormal(loc=mean_vector, covariance_matrix=cov_matrix),
        obs=data,
    )


def model_n_hv(
    input_xy_exp,
    input_xy_sim,
    input_theta_sim,
    data_exp_h,
    data_exp_v,
    data_sim_h,
    data_sim_v,
    config,
):
    """
    Model with reparameterization with both horizontal and vertical data.
    Uses config for priors and settings.
    """
    num_exp = len(input_xy_exp)
    add_bias_E1 = config["bias"]["add_bias_E1"]
    add_bias_alpha = config["bias"]["add_bias_alpha"]

    # Get all priors
    (
        theta,
        bias_E1,
        bias_alpha,
        mean_emulator,
        stdev_emulator,
        length_xy,
        length_theta,
        stdev_measure,
        stdev_measure_base,
        stdev_constant,
        _, # sigma_latent unused
    ) = get_priors_from_config(config, num_exp)

    # Prepare inputs based on bias
    data_size_exp = [i.shape[0] for i in input_xy_exp]

    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta_exp = []
        for i in range(num_exp):
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i], 1)))
        input_theta = jnp.concatenate([*input_theta_exp, input_theta_sim], axis=0)

    elif add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append(
                jnp.array(input_xy_exp[i])
                + jnp.concatenate(
                    (
                        jnp.zeros((data_size_exp[i], 1)),
                        bias_alpha[i] * jnp.ones((data_size_exp[i], 1)),
                    ),
                    axis=1,
                )
            )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        input_theta = jnp.concatenate(
            [jnp.tile(theta, (sum(data_size_exp), 1)), input_theta_sim], axis=0
        )

    else:  # No bias
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta = jnp.concatenate(
            [jnp.tile(theta, (sum(data_size_exp), 1)), input_theta_sim], axis=0
        )

    # Prepare Data
    data_h = jnp.concatenate((*data_exp_h, data_sim_h), axis=0)
    data_v = jnp.concatenate((*data_exp_v, data_sim_v), axis=0)

    # Compute Covariance Matrix
    cov_matrix = cov_matrix_emulator(
        input_xy,
        input_theta,
        input_xy,
        input_theta,
        stdev_emulator,
        length_xy,
        length_theta,
    )

    # Add Measurement Noise
    loads_exp = jnp.concatenate(input_xy_exp, axis=0)[:, 0]

    # Noise model selection
    noise_model = config["data"].get("noise_model", "proportional")

    if noise_model == "additive":
        # sigma^2 = sigma_measure^2 * P + sigma_base^2
        noise_diag = stdev_measure**2 * loads_exp + stdev_measure_base**2
    elif noise_model == "constant":
        # sigma^2 = sigma_constant^2 (constant variance)
        num_exp_points = len(loads_exp)
        noise_diag = stdev_constant**2 * jnp.ones(num_exp_points)
    elif noise_model == "prop_quad":
        # sigma^2 = sigma_measure^2 * P^2
        noise_diag = stdev_measure**2 * loads_exp**2
    else:
        # proportional: sigma^2 = sigma_measure^2 * P
        noise_diag = stdev_measure**2 * loads_exp

    diag_line = jnp.concatenate([noise_diag, jnp.zeros(input_xy_sim.shape[0])])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix += cov_matrix_measure

    # Jitter
    jitter = jnp.diag(1e-6 * jnp.ones(input_xy.shape[0]))
    cov_matrix += jitter

    # Mean Vectors
    mean_vector_h = mean_emulator * input_xy[:, 0] * jnp.sin(input_xy[:, 1])
    mean_vector_v = mean_emulator * input_xy[:, 0] * jnp.cos(input_xy[:, 1])

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



 
def model_n_hierarchical(
    input_xy_exp, # List of arrays (N, 2)
    input_xy_sim,
    input_theta_sim,
    data_exp_h_raw, # List of arrays (N, 3) - Raw multi-sensor data
    data_exp_v_raw, # List of arrays (N, 3)
    data_sim_h,
    data_sim_v,
    config,
):
    """
    Hierarchical model where each specimen has a latent random effect,
    and all 3 sensor positions are independent given the latent effect.
    """
    num_exp = len(input_xy_exp)
    add_bias_E1 = config["bias"]["add_bias_E1"]
    add_bias_alpha = config["bias"]["add_bias_alpha"]

    # Get all priors
    (
        theta,
        bias_E1,
        bias_alpha,
        mean_emulator,
        stdev_emulator,
        length_xy,
        length_theta,
        stdev_measure,
        stdev_measure_base,
        stdev_constant,
        sigma_latent,
    ) = get_priors_from_config(config, num_exp)

    # Prepare inputs based on bias (SAME AS model_n_hv)
    data_size_exp = [i.shape[0] for i in input_xy_exp]

    if add_bias_E1:
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
        input_theta_exp = []
        for i in range(num_exp):
            theta_b = theta.at[0].add(bias_E1[i])
            input_theta_exp.append(jnp.tile(theta_b, (data_size_exp[i], 1)))
        input_theta = jnp.concatenate([*input_theta_exp, input_theta_sim], axis=0)

    elif add_bias_alpha:
        input_xy_exp_b = []
        for i in range(num_exp):
            input_xy_exp_b.append(
                jnp.array(input_xy_exp[i])
                + jnp.concatenate(
                    (
                        jnp.zeros((data_size_exp[i], 1)),
                        bias_alpha[i] * jnp.ones((data_size_exp[i], 1)),
                    ),
                    axis=1,
                )
            )
        input_xy = jnp.concatenate((*input_xy_exp_b, input_xy_sim), axis=0)
        input_theta = jnp.concatenate(
            [jnp.tile(theta, (sum(data_size_exp), 1)), input_theta_sim], axis=0
        )

    else:  # No bias
        input_xy = jnp.concatenate((*input_xy_exp, input_xy_sim), axis=0)
    # --- GP SETUP ---
    
    # 1. Sample Latent Variables (Explicitly) - One per experiment
    # epsilon_i ~ N(0, sigma_latent)
    # We use a non-centered parameterization if possible, or just standard centered.
    # Centered: sample epsilon directly.
    # Non-centered: sample eps_raw ~ N(0,1), epsilon = eps_raw * sigma_latent
    # Let's use centered for simplicity as sigma_latent is sampled.
    epsilon = numpyro.sample("epsilon", dist.Normal(0.0, sigma_latent), sample_shape=(num_exp,))
    
    # Expand epsilon to match observations
    # For each experiment i, we have N_i points * 3 sensors = 3*N_i obs.
    # We need to constructing a vector of epsilons matching the flattened data.
    epsilon_expanded_list = []
    
    # Rebuild Inputs for 3x Expansion
    input_xy_exp_expanded_list = []
    input_theta_exp_expanded_list = []
    
    for i in range(num_exp):
        n_i = data_size_exp[i]
        
        # Epsilon expansion:
        # epsilon[i] repeated 3 * N_i times
        epsilon_expanded_list.append(jnp.full((3 * n_i,), epsilon[i]))

        # Base input xy for this exp
        xy_base = input_xy_exp[i] # (N_i, 2)
        
        # Handle alpha bias
        if add_bias_alpha:
             xy_base = xy_base + jnp.concatenate(
                     (jnp.zeros((n_i, 1)), bias_alpha[i] * jnp.ones((n_i, 1))),
                     axis=1
                 )
        
        # Expand xy 3x
        xy_expanded = jnp.repeat(xy_base, 3, axis=0) # (3*N_i, 2)
        input_xy_exp_expanded_list.append(xy_expanded)
        
        # Handle theta bias
        if add_bias_E1:
             theta_use = theta.at[0].add(bias_E1[i])
        else:
             theta_use = theta
             
        # Expand theta 3x
        th_expanded = jnp.tile(theta_use, (3 * n_i, 1))
        input_theta_exp_expanded_list.append(th_expanded)
            
    epsilon_vector = jnp.concatenate(epsilon_expanded_list)
    
    input_xy_exp_expanded = jnp.concatenate(input_xy_exp_expanded_list, axis=0)
    input_theta_exp_final = jnp.concatenate(input_theta_exp_expanded_list, axis=0)
         
    # Combine with Sim
    input_xy_final = jnp.concatenate((input_xy_exp_expanded, input_xy_sim), axis=0)
    input_theta_final = jnp.concatenate((input_theta_exp_final, input_theta_sim), axis=0)
    
    # Compute Covariance Matrix for ALL points
    cov_matrix = cov_matrix_emulator(
        input_xy_final,
        input_theta_final,
        input_xy_final,
        input_theta_final,
        stdev_emulator,
        length_xy,
        length_theta,
    )
    
    # Add Jitter
    cov_matrix += jnp.diag(1e-6 * jnp.ones(input_xy_final.shape[0]))
    
    # Noise Variance Diagonal
    # Exp part
    loads_exp = input_xy_exp_expanded[:, 0]
    
    # Determine noise model diag
    noise_diag_exp = jnp.where(
        stdev_constant > 0,
        stdev_constant**2 * jnp.ones(loads_exp.shape[0]),
        jnp.where(
             stdev_measure_base > 0,
             stdev_measure**2 * loads_exp + stdev_measure_base**2,
             jnp.where(
                 config["data"].get("noise_model") == "prop_quad",
                 stdev_measure**2 * loads_exp**2,
                 stdev_measure**2 * loads_exp
             )
        )
    )
    
    noise_diags = [noise_diag_exp]
    
    # Sim noise (jitter/small)
    noise_diags.append(jnp.zeros(input_xy_sim.shape[0]))
    
    noise_diag_vec = jnp.concatenate(noise_diags)
    cov_matrix += jnp.diag(noise_diag_vec)
    
    # PREPARE DATA VECTOR
    # Flatten raw data (N, 3) -> (3N,)
    data_h_list = []
    data_v_list = []
    
    for i in range(num_exp):
        data_h_list.append(data_exp_h_raw[i].flatten()) 
        data_v_list.append(data_exp_v_raw[i].flatten())
        
    data_h_list.append(data_sim_h)
    data_v_list.append(data_sim_v)
    
    data_h_final = jnp.concatenate(data_h_list)
    data_v_final = jnp.concatenate(data_v_list)
    
    # MEAN VECTOR
    # Add epsilon to mean vector (for experimental part only)
    # Sim part has epsilon=0 (or distinct?)
    # Generally latent var is for specimen. Sim is "ideal" specimen? or epsilon=0.
    # Usually Sim is mean behavior, so epsilon=0 is appropriate.
    
    # Construct epsilon vector for full data (Exp + Sim)
    epsilon_final = jnp.concatenate([epsilon_vector, jnp.zeros(input_xy_sim.shape[0])])
    
    # Mean = GP_Mean + Epsilon
    base_mean_h = mean_emulator * input_xy_final[:, 0] * jnp.sin(input_xy_final[:, 1])
    base_mean_v = mean_emulator * input_xy_final[:, 0] * jnp.cos(input_xy_final[:, 1])
    
    mean_vector_h = base_mean_h + epsilon_final
    mean_vector_v = base_mean_v + epsilon_final # Epsilon affects both directions same way?
    # Specimen offset affects the material response.
    # Ideally epsilon is on the material property or output?
    # If it's "random offset", likely additive on output.
    # Simplest: additive on output.
    
    
    # SAMPLE
    numpyro.sample(
        "data_h",
        dist.MultivariateNormal(loc=mean_vector_h, covariance_matrix=cov_matrix),
        obs=data_h_final,
    )

    numpyro.sample(
        "data_v",
        dist.MultivariateNormal(loc=mean_vector_v, covariance_matrix=cov_matrix),
        obs=data_v_final,
    )

def posterior_predict(
    rng_key,
    input_xy_exp,
    input_xy_sim,
    input_theta_exp,
    input_theta_sim,
    data_exp,
    data_sim,
    test_xy,
    test_theta,
    mean_emulator,
    stdev_emulator,
    length_xy,
    length_theta,
    stdev_measure=0.0, 
    stdev_measure_base=0.0,
    stdev_constant=0.0,
    sigma_latent=0.0, 
    experiment_ids=None,
    epsilon=None, # New arg: (N_exp,) or None
    noise_model="proportional", 
    direction="h",
):
    # Handle Hierarchical Expansion
    # Only if experiment_ids provided
    is_hierarchical = (experiment_ids is not None)
    
    # Note: sigma_latent must be provided if is_hierarchical is True, else default 0.0 is used.
    
    if is_hierarchical:
        # Hierarchical mode
        # Expand Exp Inputs: (N, 2) -> (3N, 2)
        input_xy_exp_expanded = jnp.repeat(input_xy_exp, 3, axis=0)
        input_theta_exp_expanded = jnp.repeat(input_theta_exp, 3, axis=0)
        
        # Expand Epsilon if provided
        # epsilon matches max(experiment_ids) + 1?
        # experiment_ids corresponds to input_xy_exp (N,)
        # We need to map experiment_ids expanded to epsilon values.
        # eps_expanded = epsilon[exp_ids_expanded]
        
        if epsilon is not None:
             exp_ids_expanded = jnp.repeat(experiment_ids, 3, axis=0)
             epsilon_expanded = epsilon[exp_ids_expanded]
             # Note: indices in experiment_ids must be valid for epsilon array
        
        # Combine with Sim
        # test_xy might be (M, 2)
        M_test = test_xy.shape[0]
        t_theta_expanded = jnp.tile(test_theta, (M_test, 1))

        input_xy = jnp.concatenate(
            (input_xy_exp_expanded, input_xy_sim, test_xy), axis=0
        )
        input_theta = jnp.concatenate(
            (input_theta_exp_expanded, input_theta_sim, t_theta_expanded), axis=0
        )
    else:
        # Standard mode (Legacy)
        M_test = test_xy.shape[0]
        t_theta_expanded = jnp.tile(test_theta, (M_test, 1))
        
        input_xy = jnp.concatenate(
            (input_xy_exp, input_xy_sim, test_xy), axis=0
        )
        input_theta = jnp.concatenate(
            (input_theta_exp, input_theta_sim, t_theta_expanded), axis=0
        )

    # covariance matrix for data
    cov_matrix_data = cov_matrix_emulator(
        input_xy,
        input_theta,
        input_xy,
        input_theta,
        stdev_emulator,
        length_xy,
        length_theta,
    )
    # Cast shape to int to avoid TracerIntegerConversionError
    zeros_padding = jnp.zeros(int(input_xy_sim.shape[0]) + M_test)
    
    # Latent Variance (Block Diagonal) REMOVED
    # Model now handles latent effect in Mean Function via epsilon.
    
    # Noise Variance
    # Loads for noise calc
    if experiment_ids is not None:
        loads_exp = input_xy_exp_expanded[:, 0]
    else:
        loads_exp = input_xy_exp[:, 0]

    # Noise model selection logic...
    def get_noise_diag(loads):
        var_const = stdev_constant**2 * jnp.ones_like(loads)
        var_add = stdev_measure**2 * loads + stdev_measure_base**2
        var_prop = stdev_measure**2 * loads
        var_prop_quad = stdev_measure**2 * loads**2
        
        if noise_model == "constant":
            return var_const
        elif noise_model == "additive":
            return var_add
        elif noise_model == "prop_quad":
            return var_prop_quad
        else:
            return var_prop

    noise_diag = get_noise_diag(loads_exp)

    diag_line = jnp.concatenate([noise_diag, zeros_padding])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix_data += cov_matrix_measure
    cov_matrix_data += jnp.diag(jnp.ones(cov_matrix_data.shape[0]) * 1e-10)

    # --- SIMULATION (Mean + Draw) ---
    L = jnp.linalg.cholesky(cov_matrix_data)

    # Mean Vector
    # Original GP Mean
    if direction == "h":
        mu_vec = mean_emulator * input_xy[:, 0] * jnp.sin(input_xy[:, 1])
    else:
        mu_vec = mean_emulator * input_xy[:, 0] * jnp.cos(input_xy[:, 1])
        
    # Add Epsilon (Latent Offset) to Mean
    if is_hierarchical and epsilon is not None:
         # epsilon_expanded: (3N_exp,)
         # sim + test part: 0
         eps_padding = jnp.zeros(int(input_xy_sim.shape[0]) + M_test)
         epsilon_vec = jnp.concatenate([epsilon_expanded, eps_padding])
         mu_vec = mu_vec + epsilon_vec
         
    # Generate Sample
    # draw ~ N(0, I)
    rng_key, key_u, key_pred = jax.random.split(rng_key, 3)
    u = jax.random.normal(key_u, shape=(input_xy.shape[0],))
    f_sample = mu_vec + jnp.dot(L, u)

    # Return prediction for test points
    # The last M_test points are test points
    sample_test = f_sample[-M_test:]
    
    # Compute Mean Prediction (ignoring sample noise) for return?
    # Usually we just return the sample. The wrapper computes mean over samples.
    # But function signature returns (pred_mean, pred_std, sample).
    # We must calculate conditional mean/std if analyzing GP uncertainty.
    # Using simple joint sample IS correct for `sample_test`.
    # But `pred_mean` and `pred_std`?
    # If we only used Cholesky joint sampling, we don't naturally get the *conditional* mean directly unless we do the solve.
    # The joint sample is equivalent to posterior sample.
    # We can skip computing explicit mean/std if only samples are used?
    # `analyze.py` uses `means, stds_em, f_samples`.
    # If we skip, we break contract.
    # We MUST compute conditional mean/std.
    
    # ... Implementation of Conditional Logic ... (Copying from previous thought but adapted for M points)
    # y_all includes loaded data.
    # K_dd = cov...[:-M, :-M]
    # k_dt = cov...[:-M, -M:] (Matrix)
    # k_tt = cov...[-M:, -M:] (Matrix)
    
    # Concatenate Data
    # analyze.py passes flattened data in both cases (or we should ensure it)
    if is_hierarchical:
         y_exp = data_exp.flatten()
    else:
         y_exp = data_exp # Already flattened by caller?
         if y_exp.ndim > 1:
             y_exp = y_exp.flatten()
    
    y_all = jnp.concatenate([y_exp, data_sim])
    
    # Indices
    num_data = y_all.shape[0]
    
    # Slice Covariance
    K_dd = cov_matrix_data[:num_data, :num_data]
    K_dt = cov_matrix_data[:num_data, num_data:] # (N_data, M_test)
    K_tt = cov_matrix_data[num_data:, num_data:] # (M_test, M_test)
    
    # Slice Mean
    mu_d = mu_vec[:num_data]
    mu_t = mu_vec[num_data:]
    
    # Cholesky of Data Block (already computed? L includes test points. 
    # Use L[:num_data, :num_data] corresponds to K_dd?
    # Yes, Cholesky is lower triangular. Block (1,1) depends only on Block (1,1) of K.
    L_dd = L[:num_data, :num_data] 
    
    # Mean
    # alpha = K_dd^-1 (y - mu_d)
    # We use L_dd to solve.
    # solve(L_dd, y-mu) -> tmp
    # solve(L_dd.T, tmp) -> alpha
    
    diff = y_all - mu_d
    tmp = jax.scipy.linalg.solve_triangular(L_dd, diff, lower=True)
    alpha_vec = jax.scipy.linalg.solve_triangular(L_dd.T, tmp, lower=False)
    
    pred_mean = mu_t + jnp.dot(K_dt.T, alpha_vec)
    
    # Variance
    # v = L_dd^-1 K_dt
    v_mat = jax.scipy.linalg.solve_triangular(L_dd, K_dt, lower=True)
    # pred_cov = K_tt - v.T v
    pred_cov = K_tt - jnp.dot(v_mat.T, v_mat)
    pred_var = jnp.diag(pred_cov) # We only need diagonal for std
    pred_std = jnp.sqrt(jnp.maximum(pred_var, 1e-10))
    
    # Sample (using joint sample we already computed is fine/better)
    # OR reconstruct:
    # pred_sample = pred_mean + pred_std * N(0,1)? (NO, ignores correlations between test points)
    # Correct would be MVN sample using pred_cov.
    # But `f_sample[-M:]` IS a valid sample from the joint, effectively posterior sample.
    sample_test = f_sample[-M_test:]
    
    return pred_mean, pred_std, sample_test


def gp_predict_pure(
    train_xy,
    train_theta,
    train_y,
    test_xy,
    test_theta,
    mean_emulator,
    stdev_emulator,
    length_xy,
    length_theta,
    stdev_measure,
    stdev_measure_base=0.0,
    stdev_constant=0.0,
    noise_model="proportional",
    direction="h",
    jitter=1e-6,
    rng_key=None,
    noiseless_training=False,
):
    """
    Pure JAX GP prediction (mean, sigma, sample) conditioned on training data.
    If rng_key is provided, samples from the GP posterior (latent function f).
    Returns (mu, sigma, sample).
    """
    # 1. Compute Kernels
    # K_ff (Training-Training)
    K_ff = cov_matrix_emulator(
        train_xy, train_theta,
        train_xy, train_theta,
        stdev_emulator, length_xy, length_theta
    )
    
    # Add Noise to Diagonal of K_ff
    # Replicate noise logic (simplified for JAX pure function)
    # Note: stdev_* are scalars here (sampled values)
    
    
    if not noiseless_training:
        print("DEBUG: gp_predict_pure: ADDING NOISE")
        # Noise Variance Diagonal
        loads_train = train_xy[:, 0]
        
        noise_diag = jnp.where(
            stdev_constant > 0,
            stdev_constant**2 * jnp.ones_like(loads_train),
            jnp.where(
                stdev_measure_base > 0,
                stdev_measure**2 * loads_train + stdev_measure_base**2,
                jnp.where(
                    noise_model == "prop_quad",
                    stdev_measure**2 * loads_train**2,
                    stdev_measure**2 * loads_train # Proportional default
                )
            )
        )
        K_ff = K_ff + jnp.diag(noise_diag)
    else:
        print("DEBUG: gp_predict_pure: NOISELESS TRAINING")

    # Add constant jitter
    K_ff = K_ff + jnp.eye(K_ff.shape[0]) * jitter
    
    # K_star_f (Test-Training)
    K_star_f = cov_matrix_emulator(
        test_xy, test_theta,
        train_xy, train_theta,
        stdev_emulator, length_xy, length_theta
    )
    
    # K_star_star (Test-Test)
    K_star_star = cov_matrix_emulator(
        test_xy, test_theta,
        test_xy, test_theta,
        stdev_emulator, length_xy, length_theta
    )
    
    # 2. Compute Mean Function (Physics Surrogate)
    if direction == "h":
        mu_train = mean_emulator * train_xy[:, 0] * jnp.sin(train_xy[:, 1])
        mu_test = mean_emulator * test_xy[:, 0] * jnp.sin(test_xy[:, 1])
    else:
        mu_train = mean_emulator * train_xy[:, 0] * jnp.cos(train_xy[:, 1])
        mu_test = mean_emulator * test_xy[:, 0] * jnp.cos(test_xy[:, 1])
        
    # 3. Solve (Cholesky preferred for stability)
    L = jax.scipy.linalg.cholesky(K_ff, lower=True)
    alpha = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, train_y - mu_train, lower=True))
    
    # Predictive Mean
    f_mean = mu_test + K_star_f @ alpha
    
    # Predictive Covariance
    v = jax.scipy.linalg.solve_triangular(L, K_star_f.T, lower=True)
    f_cov = K_star_star - v.T @ v
    
    # Predictive Std (diagonal)
    # Predictive Std (diagonal)
    f_sigma = jnp.sqrt(jnp.clip(jnp.diag(f_cov), a_min=0.0))
    
    f_sample = None
    if rng_key is not None:
        L_cov = jax.scipy.linalg.cholesky(f_cov + jnp.eye(f_cov.shape[0]) * 1e-10, lower=True)
        white_noise = random.normal(rng_key, (f_cov.shape[0],))
        f_sample = f_mean + L_cov @ white_noise

    return f_mean, f_sigma, f_sample


def sample_prior_predictive_curves(
    num_samples,
    rng_key,
    config,
    sim_data,
    test_loads,
    test_angles
):
    """
    Generates simulated curves by sampling parameters from priors and conditioning ONLY on simulation data.
    
    sim_data: dict with 'input_xy_sim', 'input_theta_sim', 'data_sim_h', 'data_sim_v'
    """
    from numpyro.infer import Predictive

    # 1. Define Priors-Only Model
    def prior_model():
        # Just grab priors, dummy num_exp=0 as we don't have experiment scaling bias here usually
        # But get_priors_from_config expects output for bias arrays.
        # We pass num_exp=1 to satisfy structure, though unused if bias off.
        return get_priors_from_config(config, num_exp=1)
        
    # 2. Sample Parameters
    predictive = Predictive(prior_model, num_samples=num_samples)
    samples = predictive(rng_key)
    
    # Split keys for GP sampling
    rng_key_gp, _ = random.split(rng_key)
    gp_keys = random.split(rng_key_gp, num_samples)
    
    # 3. Prepare GP Data (Constant across samples)
    train_xy = sim_data['input_xy_sim']
    train_theta = sim_data['input_theta_sim']
    
    # Support both directions? Let's assume user wants same as config or default 'h'
    direction = config['data'].get('direction', 'h')
    train_y = sim_data['data_sim_h'] if direction == 'h' else sim_data['data_sim_v']
    
    noise_model = config['data'].get('noise_model', 'proportional')
    
    # 4. Prepare Test Grid (Mesh to List)
    # Ensure test_loads/angles are flattened or grid
    # Let's assume test_loads (N_L,) and test_angles (N_A,) -> grid (N_L*N_A, 2)
    # IF they are 1D arrays.
    
    # We want to produce curves, usually Load vs Ext for specific angles.
    # Let's assume input test_loads and test_angles form the query points directly.
    # If they are separate arrays, we meshgrid them? 
    # Usually we plot Load-Ext curves at fixed angles. 
    # Let's assume the user passes fully formed test_xy grid/list (N_test, 2)
    pass # Documentation placeholder
    
    # Creating grid internally for robustness if lists passed
    if test_loads.ndim == 1 and test_angles.ndim == 1 and len(test_angles) < len(test_loads):
         # Typical case: few angles, many loads. Create blocks.
         test_xy_list = []
         test_angles_flat = [] # To track which angle correspond to which point
         for ang in test_angles:
             for load in test_loads:
                 test_xy_list.append([load, ang])
                 test_angles_flat.append(ang)
         test_xy = jnp.array(test_xy_list)
    else:
         # Assume inputs are matched
         test_xy = jnp.stack([test_loads, test_angles], axis=1)

    # 5. Vectorized Prediction function
    def single_sample_prediction(sample_idx):
        # Extract scalar params for this sample
        s = {k: v[sample_idx] for k, v in samples.items()}
        
        # Reconstruct length_xy / length_theta arrays
        # (This duplicates logic from get_priors but manual reconstruction needed as they are not single keys in samples)
        # Actually samples contains 'lambda_P', etc.
        
        l_xy = jnp.array([s["lambda_P"], s["lambda_alpha"]])
        l_theta = jnp.array([
            s["lambda_E1"], s["lambda_E2"], s["lambda_v12"], 
            s["lambda_v23"], s["lambda_G12"]
        ])
        
        # Physical theta for Test points
        # For prior prediction (simulation only), what theta do we use for test points?
        # A: The SAMPLED values (E1, E2...). We are asking:
        # "If the material really had THESE properties (E1*, E2*...), what would the curve look like?"
        # The simulation data (with FIXED theta_sim) constrains the GP.
        
        theta_val = jnp.array([s["E_1"], s["E_2"], s["v_12"], s["v_23"], s["G_12"]])
        test_theta_tiled = jnp.tile(theta_val, (test_xy.shape[0], 1))
        
        # Augment Training Data with pinned (0,0) point
        # Pinning the start of the curve (Load=0 -> Ext=0) for the current sampled material (theta_val)
        # We need to pin ALL points where Load=0 (start of each angle's curve)
        
        # Identify "zero load" points in test_xy (Column 0 is Load)
        zero_load_indices = jnp.where(jnp.isclose(test_xy[:, 0], 0.0, atol=1e-5))[0]
        
        if len(zero_load_indices) > 0:
            pin_xy = test_xy[zero_load_indices] # (N_pinned, 2)
            pin_theta = jnp.tile(theta_val, (len(zero_load_indices), 1))
            pin_y = jnp.zeros(len(zero_load_indices))
            
            train_xy_aug = jnp.concatenate([train_xy, pin_xy], axis=0)
            train_theta_aug = jnp.concatenate([train_theta, pin_theta], axis=0)
            train_y_aug = jnp.concatenate([train_y, pin_y], axis=0)
        else:
            # Fallback if no zero load found (unlikely given linspace(0,...))
            train_xy_aug = train_xy
            train_theta_aug = train_theta
            train_y_aug = train_y
        
        # Extract hyperparameters
        mu_em = s["mu_emulator"]
        sig_em = s["sigma_emulator"]
        sig_meas = s.get("sigma_measure", 0.0)
        sig_base = s.get("sigma_measure_base", 0.0)
        sig_const = s.get("sigma_constant", 0.0)
        
        mean, sigma, sample = gp_predict_pure(
            train_xy=train_xy_aug,
            train_theta=train_theta_aug,
            train_y=train_y_aug,
            test_xy=test_xy,
            test_theta=test_theta_tiled,
            mean_emulator=mu_em,
            stdev_emulator=sig_em,
            length_xy=l_xy,
            length_theta=l_theta,
            stdev_measure=sig_meas,
            stdev_measure_base=sig_base,
            stdev_constant=sig_const,
            noise_model=noise_model,
            direction=direction,
            rng_key=gp_keys[sample_idx],
            noiseless_training=True  # Simulation data is assumed noiseless
        )
        return mean, sigma, sample
        
    # Vmap over samples
    # returns (num_samples, num_test_points)
    means, sigmas, samples_path = jax.vmap(single_sample_prediction)(jnp.arange(num_samples))
    
    return means, sigmas, samples_path, test_xy
