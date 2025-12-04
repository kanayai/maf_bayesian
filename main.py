import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
import datetime
from pathlib import Path
import time

from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv

def run_inference(model, rng_key, data_dict, config):
    """
    Runs MCMC inference.
    """
    mcmc_cfg = config["mcmc"]
    
    init_strategy = init_to_median(num_samples=30)
    kernel = NUTS(model, init_strategy=init_strategy)
    
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_cfg["num_warmup"],
        num_samples=mcmc_cfg["num_samples"],
        num_chains=mcmc_cfg["num_chains"],
        thinning=mcmc_cfg["thinning"],
        progress_bar=True,
    )
    
    # Unpack data for model
    # model_n_hv(input_xy_exp, input_xy_sim, input_theta_sim, data_exp_h, data_exp_v, data_sim_h, data_sim_v, config)
    mcmc.run(rng_key, 
             data_dict["input_xy_exp"], 
             data_dict["input_xy_sim"], 
             data_dict["input_theta_sim"], 
             data_dict["data_exp_h"], 
             data_dict["data_exp_v"], 
             data_dict["data_sim_h"], 
             data_dict["data_sim_v"], 
             config)
    
    mcmc.print_summary()
    return mcmc

def save_results(mcmc, config):
    """
    Saves MCMC results to NetCDF.
    """
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_")
    
    # Construct filename
    angles = config["data"]["angles"]
    suffix = "hv" + "".join([f"_{i}" for i in angles]) if len(angles) != 3 else "hv"
    
    bias_flags = config["bias"]
    prefix = "bias_" if (bias_flags["add_bias_E1"] or bias_flags["add_bias_alpha"]) else "no_bias_"
    if bias_flags["add_bias_E1"]: prefix += "E1_"
    if bias_flags["add_bias_alpha"]: prefix += "alpha_"
    
    filename = f"{prefix}{suffix}{date_str}MAF_linear.nc"
    file_path = output_dir / filename
    
    print(f"Saving results to {file_path}...")
    idata = az.from_numpyro(mcmc)
    az.to_netcdf(idata, file_path)
    print("Done.")
    return file_path

def main():
    print("Starting inference pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    data_dict = load_all_data(config)
    
    # 2. Run Inference
    print("Running MCMC...")
    rng_key = random.PRNGKey(0)
    
    # Select model based on config (currently only model_n_hv is fully refactored and wired)
    if config["model_type"] == "model_n_hv":
        model = model_n_hv
    else:
        raise NotImplementedError(f"Model {config['model_type']} not yet implemented in main.py")

    start_time = time.time()
    print("Compiling model and warming up... (this may take a moment)")
    mcmc = run_inference(model, rng_key, data_dict, config)
    print(f"Inference completed in {time.time() - start_time:.2f}s")
    
    # 3. Save Results
    save_results(mcmc, config)

if __name__ == "__main__":
    main()
