# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import datetime
from pathlib import Path

import jax
# from jax import vmap, jit
import jax.numpy as jnp
import jax.random as random

jax.config.update("jax_enable_x64", True)
# numpyro.set_host_device_count(2)

from maf_gp import model_n_hv, run_inference_hv

# %%

# get data
data_path_h = Path("./data_extension_h")
data_path_v = Path("./data_extension_v")
# data to use
angles = [45, 90, 135] 

# experimental data
input_xy_exp = []
data_exp_h = []
for file_load_angle, file_ext in zip( sorted(data_path_h.glob("input_load_angle_exp_*")),
                sorted(data_path_h.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if (np.abs(np.rad2deg(load_angle[0,1]) - np.array(angles)) < 1e-6).any():
        input_xy_exp.append(load_angle)
        data_exp_h.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1))

data_exp_v = []
for file_load_angle, file_ext in zip( sorted(data_path_v.glob("input_load_angle_exp_*")),
                sorted(data_path_v.glob("data_extension_exp_*")) ):
    load_angle = np.loadtxt(file_load_angle, delimiter=",")
    if (np.abs(np.rad2deg(load_angle[0,1]) - np.array(angles)) < 1e-6).any():
        # input_xy_exp.append(load_angle)
        data_exp_v.append(np.loadtxt(file_ext, delimiter=",").mean(axis=1))
    
# simulation data
input_xy_sim = jnp.array(np.loadtxt(data_path_h / "input_load_angle_sim.txt", delimiter=","))
input_theta_sim = jnp.array(np.loadtxt(data_path_h / "input_theta_sim.txt", delimiter=","))
data_sim_h = jnp.array(np.loadtxt(data_path_h / "data_extension_sim.txt", delimiter=",")).mean(axis=1)

data_sim_v = jnp.array(np.loadtxt(data_path_v / "data_extension_sim.txt", delimiter=",")).mean(axis=1)


# %%
# numpyro.render_model(model, model_args=(input_xy_exp, input_xy_sim, input_theta_sim, data_exp, data_sim))
# data_exp_h

# %%
# do inference
rng_key, rng_key_predict = random.split(random.PRNGKey(12345))
# whether to add bias_E1
add_bias_E1 = False
# whether to add bias_alpha
add_bias_alpha = False
# direction = data_path.stem[-1]
mcmc = run_inference_hv(model_n_hv, rng_key, input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, 
                     add_bias_E1=add_bias_E1, add_bias_alpha=add_bias_alpha)
samples = mcmc.get_samples()

# %%
# save posterior samples
date_str = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_")
file_path = Path("results_mcmc")
file_path.mkdir(exist_ok=True)
if len(angles) == 3:
    suffix = "hv"
else:
    suffix = "hv"
    for i in angles:
        suffix = suffix + '_' + str(i)

if not add_bias_E1 and not add_bias_alpha:
    prefix = "no_bias_"
else:
    prefix = "bias_"

if add_bias_E1:
    prefix = prefix + "E1_"
if add_bias_alpha:
    prefix = prefix + "alpha_"

file_path =  file_path.joinpath(prefix + suffix + date_str + "MAF_linear.h5")
    
f = h5py.File(file_path, 'w')
for key in samples.keys():
    f.create_dataset(key, data=samples[key])
f.close()

# %%
# plot prior-posterior distributions
import numpyro.distributions as dist
matplotlib.rcParams["axes.formatter.limits"] = [-4,4]
matplotlib.rcParams["font.size"] = 14
prior_dist = [] 
prior_dist.append(dist.Normal(154900, 5050))
prior_dist.append(dist.Normal(10285, 650))
prior_dist.append(dist.Normal(0.33, 0.015))
prior_dist.append(dist.Normal(0.435, 0.0125))
prior_dist.append(dist.Normal(5115, 100))

keys = ['E_1', 'E_2', 'v_12', 'v_23', 'G_12']
fig, ax = plt.subplots(nrows=1, ncols=len(keys), figsize=(20,4))
dic = {}
for i in range(len(keys)):
    x_prior = jnp.linspace(prior_dist[i].mean - 3 * prior_dist[i].variance**(0.5), prior_dist[i].mean + 3 * prior_dist[i].variance**(0.5), 100)
    ax[i].plot(x_prior, jnp.exp(prior_dist[i].log_prob(x_prior)), 'r')  
    ax[i].hist(samples[keys[i]], bins=30, rwidth=0.9, density=True)
    ax[i].set_xlabel(keys[i])
    dic[keys[i]] = samples[keys[i]]
fig.tight_layout()



