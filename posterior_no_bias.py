
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random
# import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from pathlib import Path
import arviz as az
from tqdm import tqdm
from maf_gp import model_n_hv, posterior_predict

jax.config.update("jax_enable_x64", True)
matplotlib.rcParams["axes.formatter.limits"] = (-4,4)
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = 'sans-serif'

# path of file of saved posterior samples (need to change for the file of saved posterior samples)
# Automatic file detection
results_dir = Path("results_mcmc")
pattern = "no_bias_hv_*_MAF_linear.nc"
files = list(results_dir.glob(pattern))
if not files:
    # Fallback to check for .h5 if no .nc found (though we prefer .nc now)
    pattern_h5 = "no_bias_hv_*_MAF_linear.h5"
    files = list(results_dir.glob(pattern_h5))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} or {pattern_h5} found in {results_dir}")

# Sort by modification time (newest first)
latest_file = max(files, key=lambda f: f.stat().st_mtime)
file_path = latest_file
print(f"Using latest posterior samples file: {file_path}")
idata = az.from_netcdf(file_path)

# path of data folder (need to change for horizontal prediction or vertical prediction)
data_path = Path("./data_extension_v")
# data for which loading angles to use (need to change according to requirement)
angles = [45, 90, 135] 
# loading angle need to predict
angle_value = 45
# figure save format, dpi
fig_format = 'jpeg'
dpi = 300
# folder to save figures etc.
if len(angles) == 3:
    fig_path = Path("figures_no_bias")
    suffix = "hv"
else:
    fig_path = Path("figures_no_bias_leave_out")
    suffix = "hv"
    for i in angles:
        suffix = suffix + '_' + str(i)

fig_path.mkdir(exist_ok=True)

# direction
direction = data_path.stem[-1]
print(f"Direction: {direction}")


# experimental data
input_xy_exp_h = []
data_exp_h = []
input_xy_exp_v = []
data_exp_v = []

# Load Horizontal Data
data_path_h = Path("data_extension_h")
for file_load_angle, file_ext in zip( sorted(data_path_h.glob("input_load_angle_exp_*")),
                sorted(data_path_h.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if (np.abs(np.rad2deg(load_angle[0,1]) - np.array(angles)) < 1e-6).any():
        input_xy_exp_h.append(load_angle)
        data_exp_h.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1))

# Load Vertical Data
data_path_v = Path("data_extension_v")
for file_load_angle, file_ext in zip( sorted(data_path_v.glob("input_load_angle_exp_*")),
                sorted(data_path_v.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if (np.abs(np.rad2deg(load_angle[0,1]) - np.array(angles)) < 1e-6).any():
        input_xy_exp_v.append(load_angle)
        data_exp_v.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1))

if len(input_xy_exp_h) > 0: 
    input_xy_exp = jnp.concatenate(input_xy_exp_h, axis=0)
    # Note: model_n_hv expects lists of arrays for data_exp if we want to match how it was likely trained
    # But let's check if we need to concatenate them. 
    # Looking at MAF_gp_hv.py, it passes data_exp_h_mean which is a LIST of arrays.
    # So we should keep them as lists.
    
# simulation data
input_xy_sim = jnp.array(np.loadtxt(data_path_h / "input_load_angle_sim.txt", delimiter=","))
input_theta_sim = jnp.array(np.loadtxt(data_path_h / "input_theta_sim.txt", delimiter=","))
data_sim_h = jnp.array(np.loadtxt(data_path_h / "data_extension_sim.txt", delimiter=",").mean(axis=1))
data_sim_v = jnp.array(np.loadtxt(data_path_v / "data_extension_sim.txt", delimiter=",").mean(axis=1))

# experiment date for the prediction loading angle (plotting)
# We use the 'direction' variable to decide which one to plot
input_xy_exp_plt = []
data_exp_plt = []
data_path_plt = data_path_v if direction == 'v' else data_path_h

for file_load_angle, file_ext in zip( sorted(data_path_plt.glob("input_load_angle_exp_*")),
                             sorted(data_path_plt.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if np.abs(np.rad2deg(load_angle[0,1]) - angle_value) < 1e-6:
        input_xy_exp_plt.append(load_angle)
        data_exp_plt.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1)) 

# plot posterior distribution of parameters
# samples = mcmc.get_samples()
samples = {}
for key in idata.posterior.data_vars:
    data = idata.posterior[key].values
    # Flatten the first two dimensions (chain, draw) to match NumPyro's get_samples() output structure
    flat_data = data.reshape(-1, *data.shape[2:])
    samples[key] = jnp.array(flat_data)
    
# plot posterior distribution of parameters
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for i, ax in enumerate(axes.flatten()):
    if i < len(samples):
        key = list(samples.keys())[i]
        sns.histplot(samples[key], ax=ax, kde=True)
        ax.set_title(key)
plt.tight_layout()
plt.savefig(fig_path.joinpath("inference_theta_linear.svg"))
print('Updating "inference_theta_linear.svg" in "./figures_no_bias/" ... Done.')


# predict prior
rng_key = random.PRNGKey(0)
rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)

# prediction points
samples_load = jnp.linspace(0, 10, 100)
test_xy = []
for i in range(len(samples_load)):
    test_xy.append(jnp.array([samples_load[i], jnp.deg2rad(angle_value)]))
test_xy = jnp.stack(test_xy)

# Prior prediction
print("Running prior prediction...")
prior_predictive = Predictive(model_n_hv, num_samples=500)
# Pass both horizontal and vertical data
prior_samples = prior_predictive(rng_key_prior, input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v)

# Extract prior parameters
prior_mean_emulator = prior_samples["mu_emulator"]
prior_stdev_emulator = prior_samples["sigma_emulator"]
prior_stdev_measure = prior_samples["sigma_measure"]
prior_length_xy = jnp.stack([prior_samples["lambda_P"], prior_samples["lambda_alpha"]], axis=1)
prior_length_theta = jnp.stack([prior_samples["lambda_E1"], prior_samples["lambda_E2"], 
                                prior_samples["lambda_v12"], prior_samples["lambda_v23"], 
                                prior_samples["lambda_G12"]], axis=1)
prior_theta = jnp.stack([prior_samples["E_1"], prior_samples["E_2"], 
                         prior_samples["v_12"], prior_samples["v_23"], 
                         prior_samples["G_12"]], axis=1)

# Vectorized prediction function
def predict_batch(rng_key, theta, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure):
    def single_predict(key, t, me, se, lx, lt, sm):
        input_theta_exp = jnp.tile(t, (input_xy_exp.shape[0], 1))
        # For posterior_predict, we still use the single direction data/model logic
        # We need to decide which 'data_exp' and 'data_sim' to pass based on 'direction'
        data_exp_target = jnp.concatenate(data_exp_v, axis=0) if direction == 'v' else jnp.concatenate(data_exp_h, axis=0)
        data_sim_target = data_sim_v if direction == 'v' else data_sim_h
        
        return posterior_predict(key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, 
                                 data_exp_target, data_sim_target, test_xy, t, me, se, lx, lt, sm, direction=direction)[2]
    
    return vmap(single_predict)(random.split(rng_key, theta.shape[0]), theta, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure)

prior_predictions = predict_batch(rng_key_prior, prior_theta, prior_mean_emulator, prior_stdev_emulator, prior_length_xy, prior_length_theta, prior_stdev_measure)

mean_prediction_prior = jnp.mean(prior_predictions, axis=0)
percentiles_prior = jnp.percentile(prior_predictions, jnp.array([5.0, 95.0]), axis=0)

# make plots
fig_prior, ax_prior = plt.subplots(figsize=(5,5), constrained_layout=True)
ax_prior.fill_betweenx(samples_load, percentiles_prior[0, :], percentiles_prior[1, :], color="lightblue", label="90% interval")
ax_prior.plot(mean_prediction_prior, samples_load, "blue", ls="solid", lw=2.0, label="Mean prediction")
sz=2
for i in range(len(input_xy_exp_plt)):
    ax_prior.plot(data_exp_plt[i], input_xy_exp_plt[i][:,0], "o", markersize=sz, label='Data '+str(i+1))
ax_prior.set(xlabel="Extension [mm]", ylabel="Load [kN]", title="Prior prediction ($" + str(angle_value) + r"^\circ$)")
ax_prior.legend(fontsize=10)
if angle_value == 45: ax_prior.set_xlim(-0.005, 0.09)
if angle_value == 90: ax_prior.set_xlim(-0.02, 0.125)
if angle_value == 135: ax_prior.set_xlim(-0.055, 0.09)
fig_prior.savefig(fig_path.joinpath("prior_prediction_" + str(angle_value) + "deg_linear_" + suffix + "." + fig_format), dpi=dpi, transparent=True)


# Posterior prediction
print("Running posterior prediction...")
# Extract posterior parameters from 'samples' dict
post_mean_emulator = samples["mu_emulator"]
post_stdev_emulator = samples["sigma_emulator"]
post_stdev_measure = samples["sigma_measure"]
post_length_xy = jnp.stack([samples["lambda_P"], samples["lambda_alpha"]], axis=1)
post_length_theta = jnp.stack([samples["lambda_E1"], samples["lambda_E2"], 
                               samples["lambda_v12"], samples["lambda_v23"], 
                               samples["lambda_G12"]], axis=1)
post_theta = jnp.stack([samples["E_1"], samples["E_2"], 
                        samples["v_12"], samples["v_23"], 
                        samples["G_12"]], axis=1)

# Use a subset of samples if too many (e.g. 500)
num_post_samples = min(500, post_theta.shape[0])
# Ensure we have enough samples, if not use all
if post_theta.shape[0] > 0:
    indices = np.random.choice(post_theta.shape[0], num_post_samples, replace=False)
else:
    indices = []

predictions_post = []
if len(indices) > 0:
    # Slice the parameters for the selected indices
    post_mean_emulator_sel = post_mean_emulator[indices]
    post_stdev_emulator_sel = post_stdev_emulator[indices]
    post_stdev_measure_sel = post_stdev_measure[indices]
    post_length_xy_sel = post_length_xy[indices]
    post_length_theta_sel = post_length_theta[indices]
    post_theta_sel = post_theta[indices]
    
    predictions_post = predict_batch(rng_key_post, post_theta_sel, post_mean_emulator_sel, post_stdev_emulator_sel, 
                                     post_length_xy_sel, post_length_theta_sel, post_stdev_measure_sel)

if len(predictions_post) > 0:
    mean_prediction_post = jnp.mean(predictions_post, axis=0)
    percentiles_post = jnp.percentile(predictions_post, jnp.array([5.0, 95.0]), axis=0)
else:
    # Fallback if no samples (shouldn't happen if MCMC ran)
    mean_prediction_post = jnp.zeros_like(samples_load)
    percentiles_post = jnp.zeros((2, len(samples_load)))

# make plots
fig_post, ax_post = plt.subplots(figsize=(5,5), constrained_layout=True)
ax_post.fill_betweenx(samples_load, percentiles_post[0, :], percentiles_post[1, :], color="lightblue", label="90% interval")
ax_post.plot(mean_prediction_post, samples_load, "blue", ls="solid", lw=2.0, label="Posterior mean")
sz=2
for i in range(len(input_xy_exp_plt)):
    ax_post.plot(data_exp_plt[i], input_xy_exp_plt[i][:,0], "o", markersize=sz, label='Data '+str(i+1))
ax_post.set(xlabel="Extension [mm]", ylabel="Load [kN]", title="Posterior prediction ($" + str(angle_value) + r"^\circ$)")
ax_post.legend(fontsize=10)
if angle_value == 45: ax_post.set_xlim(-0.005, 0.09)
if angle_value == 90: ax_post.set_xlim(-0.02, 0.125)
if angle_value == 135: ax_post.set_xlim(-0.055, 0.09)
fig_post.savefig(fig_path.joinpath("post_prediction_"+str(angle_value)+"deg_linear_" + suffix + "." + fig_format), dpi=dpi, transparent=True)

# plot prior and posterior together
fig_all, ax_all = plt.subplots(figsize=(5,5), constrained_layout=True)
ax_all.fill_betweenx(samples_load, percentiles_prior[0, :], percentiles_prior[1, :], alpha=0.75, color="lightgreen", label='Prior 95% interval')
ax_all.plot(mean_prediction_prior, samples_load, c="green", ls="dashed", lw=1., label='Prior mean')
ax_all.fill_betweenx(samples_load, percentiles_post[0, :], percentiles_post[1, :], alpha=1, color="lightblue", label="Posterior 95% interval")
ax_all.plot(mean_prediction_post, samples_load, c="blue", ls="solid", lw=0.5, label="Posterior mean")
sz=2
for i in range(len(input_xy_exp_plt)):
    ax_all.plot(data_exp_plt[i], input_xy_exp_plt[i][:,0], "o", markersize=sz, label='Data '+str(i+1))
ax_all.set(xlabel="Extension [mm]", ylabel="Load [kN]", title="Predictions ($" + str(angle_value) + r"^\circ$)")
ax_all.legend(fontsize=10, loc="lower right")
if angle_value == 45: ax_all.set_xlim(-0.005, 0.09)
if angle_value == 90: ax_all.set_xlim(-0.02, 0.125)
if angle_value == 135: ax_all.set_xlim(-0.055, 0.09)
fig_all.savefig(fig_path.joinpath("prediction_"+str(angle_value)+"deg_linear_" + suffix + "." + fig_format), dpi=dpi, transparent=True)
