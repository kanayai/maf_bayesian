
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
# import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from pathlib import Path
import arviz as az
import arviz as az
from tqdm import tqdm
from maf_gp import model_n_hv, posterior_predict

jax.config.update("jax_enable_x64", True)
matplotlib.rcParams["axes.formatter.limits"] = (-4,4)
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = 'sans-serif'

# Automatic file detection
results_dir = Path("results_mcmc")
pattern = "bias_alpha_hv_*_MAF_linear.nc" 
files = list(results_dir.glob(pattern))
if not files:
    print("Warning: No bias_alpha files found. Falling back to no_bias files for testing.")
    pattern_fallback = "no_bias_hv_*_MAF_linear.nc"
    files = list(results_dir.glob(pattern_fallback))
    if not files:
        pattern_h5 = "no_bias_hv_*_MAF_linear.h5"
        files = list(results_dir.glob(pattern_h5))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} or fallback found in {results_dir}")

# Sort by modification time (newest first)
latest_file = max(files, key=lambda f: f.stat().st_mtime)
file_path = latest_file
print(f"Using latest posterior samples file: {file_path}")
idata = az.from_netcdf(file_path)

f = {}
for key in idata.posterior.data_vars:
    data = idata.posterior[key].values
    flat_data = data.reshape(-1, *data.shape[2:])
    f[key] = jnp.array(flat_data)

# path of data folder (need to change for horizontal prediction or vertical prediction)
data_path = Path("data_extension_v")
# data for which loading angles to use (need to change according to requirement)
angles = [45, 90, 135] 
# loading angle need to predict
angle_value = 45
# figure format, dpi
fig_format = 'jpeg'
dpi = 300
# folder to save figures etc.
if len(angles) == 3:
    fig_path = Path("figures_bias_alpha")
else:
    fig_path = Path("figures_bias_alpha_leave_out")

fig_path.mkdir(exist_ok=True)

# direction
direction = data_path.stem[-1]
print(f"Direction: {direction}")

# experimental data
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
    
num_exp = len(input_xy_exp_h)

# simulation data
input_xy_sim = jnp.array(np.loadtxt(data_path_h / "input_load_angle_sim.txt", delimiter=","))
input_theta_sim = jnp.array(np.loadtxt(data_path_h / "input_theta_sim.txt", delimiter=","))
data_sim_h = jnp.array(np.loadtxt(data_path_h / "data_extension_sim.txt", delimiter=",").mean(axis=1))
data_sim_v = jnp.array(np.loadtxt(data_path_v / "data_extension_sim.txt", delimiter=",").mean(axis=1))

# experiment date for the prediction loading angle (plotting)
input_xy_exp_plt = []
data_exp_plt = []
data_path_plt = data_path_v if direction == 'v' else data_path_h

for file_load_angle, file_ext in zip( sorted(data_path_plt.glob("input_load_angle_exp_*")),
                             sorted(data_path_plt.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if np.abs(np.rad2deg(load_angle[0,1]) - angle_value) < 1e-6:
        input_xy_exp_plt.append(load_angle)
        data_exp_plt.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1))


# Plot prior and posterior distributions
prior_dist = [] 
prior_dist.append(dist.Normal(154900, 5050))
prior_dist.append(dist.Normal(10285, 300))
prior_dist.append(dist.Normal(0.33, 0.015))
prior_dist.append(dist.Normal(0.435, 0.0125))
prior_dist.append(dist.Normal(5115, 100))

keys = ['E_1', 'E_2', 'v_12', 'v_23', 'G_12']
x_labels = ['$E_1$ [MPa]', '$E_2$ [MPa]', '$\\nu_{12}$ [-]', '$\\nu_{23}$ [-]', '$G_{12}$ [MPa]'] 
dic_0 = {}
dic = {}
fig, ax = plt.subplots(nrows=1, ncols=len(keys), figsize=(20,4), layout='constrained')
for i in range(len(keys)):
    dic_0[x_labels[i]] = f[keys[i]] 
    dic[x_labels[i]] = f[keys[i]][::3]
    if i==0:
        x_prior = jnp.linspace(140000,175000)
    else:
        if i==1:
            x_prior = jnp.linspace(6000,13000)
        else:     
            if i==2:
               x_prior = jnp.linspace(0.25,0.4)
            else:
                if i==3:
                   x_prior = jnp.linspace(0.35,0.5)
                else:                  
                    x_prior = jnp.linspace(4600,5600)    
    ax[i].plot(x_prior, jnp.exp(prior_dist[i].log_prob(x_prior)), color='tab:green', lw=1.5, label='Prior')   
    ax[i].hist(f[keys[i]], bins=30, rwidth=0.9, color="tab:blue", density=True, label='Posterior')
    ax[i].set_xlabel(x_labels[i])
    if i == 0:
        ax[i].set_xticks(np.arange(1.4, 1.75, 0.1)* 1e5)
        ax[i].set_ylabel("PDF")
        ax[i].legend()

if len(angles) == 3:
    suffix = data_path.stem[-1]
else:
    suffix = data_path.stem[-1]
    for i in angles:
        suffix = suffix + '_' + str(i)

fig.savefig(fig_path.joinpath("inference_theta_linear_bias_alpha" + suffix[1:] + "." + fig_format),
            dpi=dpi, transparent=True)

df_0 = pd.DataFrame(dic_0)
post_stats = {"mean": df_0.mean(),
             "variance": df_0.var(),
              "std": df_0.std()}
pd.concat(post_stats, axis=1).to_csv(fig_path.joinpath("inference_theta_linear_bias_alpha_stats" + suffix[1:] + ".csv"))

#pair plots
df = pd.DataFrame(dic)
grid = sns.pairplot(df, kind='scatter', diag_kind='hist')
grid.savefig("inference_theta_grid_linear_bias_alpha_" + suffix + "." + fig_format, dpi=dpi, transparent=True)

# bias 
keys_b = []
x_labels_b = []
dic_b = {}
bias_exists = "b_1_alpha" in f
if bias_exists:
    for i in range(len(input_xy_exp_h)):
        keys_b.append("b_"+str(i+1)+"_alpha")
        x_labels_b.append("$b_{\\alpha,"+str(i+1)+"}$ [rad]")
        dic_b["$b_{\\alpha,"+str(i+1)+"}$ [rad]"] = f[keys_b[i]]

    if len(angles) < 3:
        fig_b, ax_b = plt.subplots(nrows=1, ncols=len(keys_b), figsize=(len(keys_b)*4, 4), layout='constrained')
        for i in range(len(keys_b)):
            x_prior = np.linspace(-0.1, 0.1, 100)
            ax_b[i].plot(x_prior, np.exp(dist.Normal(0,0.01).log_prob(x_prior)), color='tab:green', lw=1.5, label='Prior')   
            ax_b[i].hist(f[keys_b[i]], bins=30, rwidth=0.9, color='tab:blue', density=True, label='posterior')
            ax_b[i].set_xlabel(x_labels_b[i])
            if i == 0:
                ax_b[i].legend()
                ax_b[i].set_ylabel("PDF")
    else:
        fig_b, ax_b = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), layout="constrained")
        for i in range(3):
            for j in range(3):
                if i==2 and j==2:
                    ax_b[i,j].set_axis_off()
                    break
                x_prior = np.linspace(-0.1, 0.1, 100)
                ax_b[i,j].plot(x_prior, np.exp(dist.Normal(0,0.01).log_prob(x_prior)), color='tab:green', lw=1.5, label='Prior')
                ax_b[i,j].hist(f[keys_b[i*3+j]], bins=30, rwidth=0.9, color='tab:blue', density=True, label='Posterior')
                ax_b[i,j].set_xlabel(x_labels_b[i*3+j])
                if i==0 and j==0:
                    ax_b[i,j].legend()
                    ax_b[i,j].set_ylabel("PDF")
                if (i==1 or i==2) and j==0:
                    ax_b[i,j].set_ylabel("PDF")

    df_b = pd.DataFrame(dic_b)
    post_stats_b = {"mean": df_b.mean(),
                    "variance": df_b.var(),
                    "std": df_b.std()}
    pd.concat(post_stats_b, axis=1).to_csv(fig_path.joinpath("inference_b_linear_bias_alpha_stats" + suffix[1:] + ".csv"))

    fig_b.savefig(fig_path.joinpath("inference_b_linear_bias_alpha" + suffix[1:] + "." + fig_format),
                  dpi=dpi, transparent=True)
else:
    print("Skipping bias plots as bias keys not found in samples.")


prior_dist_hyper = [] 
prior_dist_hyper.append(dist.Normal(0, 0.01))
prior_dist_hyper.append(dist.Exponential(20.))
prior_dist_hyper.append(dist.LogNormal(1.5, 0.5))
prior_dist_hyper.append(dist.LogNormal(0.34, 0.5))
prior_dist_hyper.append(dist.LogNormal(11., 0.5))
prior_dist_hyper.append(dist.LogNormal(8.3, 0.5))
prior_dist_hyper.append(dist.LogNormal(-0.80, 0.5))
prior_dist_hyper.append(dist.LogNormal(-0.80, 0.5))
prior_dist_hyper.append(dist.LogNormal(7.7, 0.5))
prior_dist_hyper.append(dist.Exponential(100.))

# hyper-parameters
keys_hyper = ['mu_emulator', 'sigma_emulator', 'lambda_P', 'lambda_alpha', 'lambda_E1', 
              'lambda_E2', 'lambda_v12', 'lambda_v23', 'lambda_G12', 'sigma_measure']
x_labels_hyper = [r'$\beta$ [mm/kN]', r'$\sigma_{\eta}$ [mm]', r'$\lambda_{\eta,P}$ [kN]', r'$\lambda_{\eta,\alpha}$ [rad]', 
                  r'$\lambda_{\eta,E_1}$ [MPa]', r'$\lambda_{\eta,E_2}$ [MPa]', r'$\lambda_{\eta,v_{12}}$ [-]', r'$\lambda_{\eta,v_{23}}$ [-]', 
                  r'$\lambda_{\eta,G_{12}}$ [GPa]', r'$\sigma_{\epsilon}$ [mm/$\sqrt{\mathrm{kN}}$]']
dic_hyper_0 = {}
dic_hyper = {}
fig_hyper, ax_hyper = plt.subplots(nrows=2, ncols=5, figsize=(20,8), layout='constrained')
for i in range(2):
    for j  in range(5):
        dic_hyper_0[x_labels_hyper[i*5+j]] = f[keys_hyper[i*5+j]]
        dic_hyper[x_labels_hyper[i*5+j]] = f[keys_hyper[i*5+j]][::3]
        # fig, ax = plt.subplots()
        samples_prior = prior_dist_hyper[i*5+j].sample(random.PRNGKey(0), (5000,))
        x_prior = jnp.linspace(samples_prior.min(), samples_prior.max(), 100)
        ax_hyper[i,j].plot(x_prior, jnp.exp(prior_dist_hyper[i*5+j].log_prob(x_prior)), color='tab:green', lw=1.5, label='Prior')  
        ax_hyper[i,j].hist(f[keys_hyper[i*5+j]], bins=30, rwidth=0.9, color='tab:blue', density=True, label='Posterior')
        ax_hyper[i,j].set_xlabel(x_labels_hyper[i*5+j])
        if i==0 and j==0:
            ax_hyper[i,j].legend(loc='upper left')
            ax_hyper[i,j].set_ylabel("PDF")
        if i==1 and j==0:
            ax_hyper[i,j].set_ylabel("PDF")
        if i*5+j == 0:
            ax_hyper[i,j].set_ylim([0,60])
        if i*5+j == 1:
            ax_hyper[i,j].set_ylim([0,50])
            ax_hyper[i,j].set_xlim([-0.02,0.2])
        if i*5+j == 9:
            ax_hyper[i,j].set_ylim([0,200])
            ax_hyper[i,j].set_xlim([-0.002,0.02])
        if i*5+j == 3:
            ax_hyper[i,j].set_xlim([-0.1,4]) 
        if i*5+j == 4:
            ax_hyper[i,j].set_xlim([-0.1e5,2e5]) 

fig_hyper.savefig(fig_path.joinpath("inference_hyper_linear_bias_alpha" + suffix[1:] + "." + fig_format),
                  dpi=dpi, transparent=True)


df_hyper_0 = pd.DataFrame(dic_hyper_0)
post_stats_hyper = {"mean": df_hyper_0.mean(),
                     "variance": df_hyper_0.var(),
                    "std": df_hyper_0.std()}
pd.concat(post_stats_hyper, axis=1).to_csv(fig_path.joinpath("inference_hyper_linear_bias_alpha_stats" + suffix[1:] + ".csv"))


# predict prior
rng_key = random.PRNGKey(0)
rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)

# prediction points
samples_load = jnp.linspace(0, 100, 100)
test_xy = []
for i in range(len(samples_load)):
    test_xy.append(jnp.array([samples_load[i], jnp.deg2rad(angle_value)]))
test_xy = jnp.stack(test_xy)

# Prior prediction
print("Running prior prediction...")
prior_predictive = Predictive(model_n_hv, num_samples=500)
prior_samples = prior_predictive(rng_key_prior, input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, add_bias_alpha=True)

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
def predict_batch_biased(rng_key, theta, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, samples_dict):
    def single_predict(key, t, me, se, lx, lt, sm, bias_vals):
        # Construct biased input_xy_exp
        # We need to reconstruct the list of inputs with bias added
        input_xy_exp_biased_list = []
        for k in range(num_exp):
            xy_b = input_xy_exp_h[k] # Base inputs
            # Add bias to angle (index 1)
            # bias_vals[k] corresponds to b_{k+1}_alpha
            xy_b = xy_b.at[:,1].add(bias_vals[k])
            input_xy_exp_biased_list.append(xy_b)
        input_xy_exp_biased = jnp.concatenate(input_xy_exp_biased_list, axis=0)
        
        input_theta_exp = jnp.tile(t, (input_xy_exp_biased.shape[0], 1))
        
        # Add sampled bias for prediction
        key, subkey = random.split(key)
        bias_new = dist.Normal(0, 0.01).sample(subkey)
        test_xy_b = test_xy.at[:,1].add(bias_new)
        
        # Select target data based on direction
        data_exp_target = jnp.concatenate(data_exp_v, axis=0) if direction == 'v' else jnp.concatenate(data_exp_h, axis=0)
        data_sim_target = data_sim_v if direction == 'v' else data_sim_h

        return posterior_predict(key, input_xy_exp_biased, input_xy_sim, input_theta_exp, input_theta_sim, 
                                 data_exp_target, data_sim_target, test_xy_b, t, me, se, lx, lt, sm, direction=direction)[2]

    # Extract bias values for all samples
    # bias_vals will be shape (num_samples, num_exp)
    bias_vals_list = []
    for k in range(num_exp):
        bias_key = "b_"+str(k+1)+"_alpha"
        if bias_key in samples_dict:
            bias_vals_list.append(samples_dict[bias_key])
        else:
            bias_vals_list.append(jnp.zeros(theta.shape[0]))
    bias_vals = jnp.stack(bias_vals_list, axis=1)
    
    return vmap(single_predict)(random.split(rng_key, theta.shape[0]), theta, mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, bias_vals)

prior_predictions = predict_batch_biased(rng_key_prior, prior_theta, prior_mean_emulator, prior_stdev_emulator, prior_length_xy, prior_length_theta, prior_stdev_measure, prior_samples)

prior_predictions = jnp.stack(prior_predictions)
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
fig_prior.savefig(fig_path.joinpath("prior_prediction_" + str(angle_value) + "deg_linear_bias_alpha_" + suffix + "." + fig_format), dpi=dpi, transparent=True)


# Posterior prediction
print("Running posterior prediction...")
# Extract posterior parameters from 'f' dict
post_mean_emulator = f["mu_emulator"]
post_stdev_emulator = f["sigma_emulator"]
post_stdev_measure = f["sigma_measure"]
post_length_xy = jnp.stack([f["lambda_P"], f["lambda_alpha"]], axis=1)
post_length_theta = jnp.stack([f["lambda_E1"], f["lambda_E2"], 
                               f["lambda_v12"], f["lambda_v23"], 
                               f["lambda_G12"]], axis=1)
post_theta = jnp.stack([f["E_1"], f["E_2"], 
                        f["v_12"], f["v_23"], 
                        f["G_12"]], axis=1)

# Use a subset of samples if too many (e.g. 500)
num_post_samples = min(500, post_theta.shape[0])
if post_theta.shape[0] > 0:
    indices = np.random.choice(post_theta.shape[0], num_post_samples, replace=False)
else:
    indices = []

predictions_post = []
if len(indices) > 0:
    # Slice parameters
    post_mean_emulator_sel = post_mean_emulator[indices]
    post_stdev_emulator_sel = post_stdev_emulator[indices]
    post_stdev_measure_sel = post_stdev_measure[indices]
    post_length_xy_sel = post_length_xy[indices]
    post_length_theta_sel = post_length_theta[indices]
    post_theta_sel = post_theta[indices]
    
    # Slice bias values from 'f' dictionary
    # We need to pass a sliced dictionary or handle slicing inside the helper
    # Let's create a sliced dictionary for bias keys
    samples_dict_sel = {}
    for k in range(num_exp):
        bias_key = "b_"+str(k+1)+"_alpha"
        if bias_key in f:
            samples_dict_sel[bias_key] = f[bias_key][indices]
            
    predictions_post = predict_batch_biased(rng_key_post, post_theta_sel, post_mean_emulator_sel, post_stdev_emulator_sel, 
                                            post_length_xy_sel, post_length_theta_sel, post_stdev_measure_sel, samples_dict_sel)

if len(predictions_post) > 0:
    mean_prediction_post = jnp.mean(predictions_post, axis=0)
    percentiles_post = jnp.percentile(predictions_post, jnp.array([5.0, 95.0]), axis=0)
else:
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
fig_post.savefig(fig_path.joinpath("post_prediction_"+str(angle_value)+"deg_linear_bias_alpha_" + suffix + "." + fig_format), dpi=dpi, transparent=True)

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
fig_all.savefig(fig_path.joinpath("prediction_"+str(angle_value)+"deg_linear_bias_alpha_" + suffix + "." + fig_format), dpi=dpi, transparent=True)
