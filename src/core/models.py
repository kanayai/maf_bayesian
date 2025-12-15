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
    bias_flags = config["bias"]

    # --- Physical Parameters (Theta) ---
    theta_vals = []
    # Order matters: E1, E2, v12, v23, G12
    param_names = ["E_1", "E_2", "v_12", "v_23", "G_12"]

    if "n" not in config["model_type"]:
        raise ValueError(
            "Only reparameterized models (model_type='model_n' or 'model_n_hv') are supported."
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
        stdev_constant = hyper["sigma_constant"]["target_dist"].icdf(
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
        stdev_measure_base = hyper["sigma_measure_base"]["target_dist"].icdf(
            cdf_normal(stdev_measure_base_n)
        )
        numpyro.deterministic("sigma_measure_base", stdev_measure_base)
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
    stdev_measure,
    stdev_measure_base=0.0,
    stdev_constant=0.0,
    direction="h",
):
    # compute kernels between train and test data, etc.
    input_xy = jnp.concatenate(
        (input_xy_exp, input_xy_sim, test_xy[0][None, :]), axis=0
    )
    # input_theta_exp = jnp.tile(test_theta, (input_xy_exp.shape[0],1))
    input_theta = jnp.concatenate(
        (input_theta_exp, input_theta_sim, test_theta[None, :]), axis=0
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
    zeros_padding = jnp.zeros(int(input_xy_sim.shape[0]) + 1)

    # Noise variance - handle all three models using jnp.where (JAX-compatible)
    # Use nested where to check: constant > 0 -> constant, else (base > 0 -> additive, else proportional)
    noise_diag = jnp.where(
        stdev_constant > 0,
        stdev_constant**2 * jnp.ones(input_xy_exp.shape[0]),  # Constant model
        jnp.where(
            stdev_measure_base > 0,
            stdev_measure**2 * input_xy_exp[:, 0]
            + stdev_measure_base**2,  # Additive model
            stdev_measure**2 * input_xy_exp[:, 0],  # Proportional model
        ),
    )

    diag_line = jnp.concatenate([noise_diag, zeros_padding])
    cov_matrix_measure = jnp.diag(diag_line)
    cov_matrix_data += cov_matrix_measure
    cov_matrix_data += jnp.diag(jnp.ones(cov_matrix_data.shape[0]) * 1e-10)

    # covariance matrix for the interpolation points
    # Cast shape to int
    test_theta_p = jnp.tile(test_theta, (int(test_xy.shape[0]), 1))
    cov_matrix_interp = cov_matrix_emulator(
        test_xy,
        test_theta_p,
        test_xy,
        test_theta_p,
        stdev_emulator,
        length_xy,
        length_theta,
    )

    # covariance matrix between data and interpolation points
    cov_matrix_data_interp = cov_matrix_emulator(
        input_xy,
        input_theta,
        test_xy,
        test_theta_p,
        stdev_emulator,
        length_xy,
        length_theta,
    )
    # cov_matrix_interp_data = cov_matrix_emulator(test_xy, test_theta_p, input_xy, input_theta, stdev_emulator, length_xy, length_theta)
    cov_matrix_interp_data = cov_matrix_data_interp.T

    # cov_matrix_data_inv = jnp.linalg.inv(cov_matrix_data)
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ (cov_matrix_data_inv @ (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))
    # linear mean function
    if direction == "h":
        mean_post_emulator = mean_emulator * test_xy[:, 0] * jnp.sin(
            test_xy[:, 1]
        ) + cov_matrix_interp_data @ jnp.linalg.solve(
            cov_matrix_data,
            (
                jnp.concatenate((data_exp, data_sim, jnp.zeros(1)), axis=0)
                - mean_emulator * input_xy[:, 0] * jnp.sin(input_xy[:, 1])
            ),
        )
    elif direction == "v":
        mean_post_emulator = mean_emulator * test_xy[:, 0] * jnp.cos(
            test_xy[:, 1]
        ) + cov_matrix_interp_data @ jnp.linalg.solve(
            cov_matrix_data,
            (
                jnp.concatenate((data_exp, data_sim, jnp.zeros(1)), axis=0)
                - mean_emulator * input_xy[:, 0] * jnp.cos(input_xy[:, 1])
            ),
        )

    # constant mean function
    # mean_post_emulator = mean_emulator + cov_matrix_interp_data @ jnp.linalg.solve(cov_matrix_data, (jnp.concatenate((data_exp, data_sim, np.zeros(1)), axis=0) - mean_emulator))

    # cov_post_emulator = cov_matrix_interp - jnp.matmul(cov_matrix_interp_data, jnp.matmul(cov_matrix_data_inv, cov_matrix_data_interp))
    cov_post_emulator = cov_matrix_interp - jnp.matmul(
        cov_matrix_interp_data,
        jnp.linalg.solve(cov_matrix_data, cov_matrix_data_interp),
    )
    stdev_post_emulator = jnp.sqrt(jnp.clip(jnp.diag(cov_post_emulator), a_min=0.0))

    L = jnp.linalg.cholesky(
        cov_post_emulator + jnp.diag(jnp.ones(test_xy.shape[0]) * 1e-10)
    )
    # Need random here, so we need to import it or pass it
    # The function signature has rng_key
    import jax.random as random

    white_noise = random.normal(rng_key, (test_xy.shape[0],))
    sample_post = L @ white_noise + mean_post_emulator
    # we return both the mean function, standard deviation and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_post_emulator, stdev_post_emulator, sample_post


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
                stdev_measure**2 * loads_train # Proportional default
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
