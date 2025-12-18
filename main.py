import jax.random as random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
import datetime
from pathlib import Path
import time
import argparse

from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, model_n, model_empirical

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
    if config["model_type"] == "model_n_hv":
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
    elif config["model_type"] == "model_n":
        # model_n(input_xy_exp, input_xy_sim, input_theta_sim, data_exp, data_sim, config)
        direction = config["data"].get("direction", "h") # 'h' or 'v'
        
        if direction == "h":
            data_exp = data_dict["data_exp_h"]
            data_sim = data_dict["data_sim_h"]
        else: # 'v'
            data_exp = data_dict["data_exp_v"]
            data_sim = data_dict["data_sim_v"]
            
        mcmc.run(rng_key,
                 data_dict["input_xy_exp"],
                 data_dict["input_xy_sim"],
                 data_dict["input_theta_sim"],
                 data_exp,
                 data_sim,
                 config)
    elif config["model_type"] == "model_empirical":
        # model_empirical(input_xy_exp, data_exp_h, data_exp_v, config)
        # Pre-calculate angle indices map
        # Map each experiment to the index of its angle in standard_angles list
        standard_angles = [45, 90, 135]
        exp_angle_indices = []
        for i in range(len(data_dict["input_xy_exp"])):
            ang_rad = data_dict["input_xy_exp"][i][0, 1]
            ang_deg = int(round(jnp.degrees(ang_rad)))
            try:
                idx = standard_angles.index(ang_deg)
                exp_angle_indices.append(idx)
            except ValueError:
                exp_angle_indices.append(-1) # Should not happen if data is consistent
        
        mcmc.run(rng_key,
                 data_dict["input_xy_exp"],
                 data_dict["data_exp_h_raw"],
                 data_dict["data_exp_v_raw"],
                 jnp.array(exp_angle_indices), # Pass angle indices
                 config)
    
    mcmc.print_summary()
    return mcmc

def save_results(mcmc, config, output_mode="default"):
    """
    Saves MCMC results to NetCDF.
    
    Args:
        mcmc: MCMC object with results
        config: Configuration dictionary
        output_mode: One of "experimental", "final", or "default"
                    - "experimental": saves to results/tmp/
                    - "final": saves to results/final/
                    - "default": saves to results/
    """
    # Determine output directory based on mode
    if output_mode == "experimental":
        output_dir = Path("results") / "tmp"
    elif output_mode == "final":
        output_dir = Path("results") / "final"
    else:
        output_dir = Path("results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_")
    
    # Construct filename
    angles = config["data"]["angles"]
    model_type = config["model_type"]
    
    if model_type == "model_n_hv":
        suffix = "hv" + "".join([f"_{i}" for i in angles]) if len(angles) != 3 else "hv"
    else:
        # For model_n, include direction
        direction = config["data"].get("direction", "h")
        dir_tag = "shear" if direction == "h" else "normal"
        suffix = f"{dir_tag}" + "".join([f"_{i}" for i in angles])
        
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Bayesian inference on composite laminate data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Default: saves to results/
  python main.py --experimental   # Saves to results/tmp/
  python main.py --final          # Saves to results/final/
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--experimental',
        action='store_true',
        help='Save results to results/tmp/ (for experimental/testing runs)'
    )
    mode_group.add_argument(
        '--final',
        action='store_true',
        help='Save results to results/final/ (for important/final runs)'
    )
    
    args = parser.parse_args()
    
    # Determine output mode
    if args.experimental:
        output_mode = "experimental"
        print("🧪 Running in EXPERIMENTAL mode - results will be saved to results/tmp/")
    elif args.final:
        output_mode = "final"
        print("📌 Running in FINAL mode - results will be saved to results/final/")
    else:
        output_mode = "default"
        print("Running in default mode - results will be saved to results/")
    
    print("Starting inference pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    data_dict = load_all_data(config)
    
    # 2. Run Inference
    print("Running MCMC...")
    rng_key = random.PRNGKey(config.get("seed", 0))
    
    # Select model based on config (currently only model_n_hv is fully refactored and wired)
    if config["model_type"] == "model_n_hv":
        model = model_n_hv
    elif config["model_type"] == "model_n":
        model = model_n
    elif config["model_type"] == "model_empirical":
        model = model_empirical
    else:
        raise NotImplementedError(f"Model {config['model_type']} not yet implemented in main.py")

    start_time = time.time()
    print("Compiling model and warming up... (this may take a moment)")
    mcmc = run_inference(model, rng_key, data_dict, config)
    print(f"Inference completed in {time.time() - start_time:.2f}s")
    
    # 3. Save Results
    save_results(mcmc, config, output_mode)

if __name__ == "__main__":
    main()
