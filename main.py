import jax.random as random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
import time
import argparse
import sys

from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, model_n, model_empirical, model_simple
from src.io.run_bundle import mark_run_completed, mark_run_failed, prepare_run_bundle

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
    elif config["model_type"] == "model_simple":
         # Check standard angles
         standard_angles = [45, 90, 135]
         exp_angle_indices = []
         for i in range(len(data_dict["input_xy_exp"])):
             ang_rad = data_dict["input_xy_exp"][i][0, 1]
             ang_deg = int(round(jnp.degrees(ang_rad)))
             try:
                 idx = standard_angles.index(ang_deg)
                 exp_angle_indices.append(idx)
             except ValueError:
                 exp_angle_indices.append(-1)
         
         mcmc.run(rng_key,
                  data_dict["input_xy_exp"],
                  data_dict["data_exp_h_raw"],
                  data_dict["data_exp_v_raw"],
                  jnp.array(exp_angle_indices),
                  config)
    
    mcmc.print_summary()
    return mcmc

def save_results(mcmc, bundle):
    """
    Saves MCMC results to the prepared immutable run bundle.
    
    Args:
        mcmc: MCMC object with results
        bundle: Run bundle metadata and target paths
    """
    print(f"Saving results to {bundle.result_path}...")
    idata = az.from_numpyro(mcmc)
    az.to_netcdf(idata, bundle.result_path)
    manifest = mark_run_completed(bundle)
    print(f"Done. Run bundle written to {bundle.bundle_dir}")
    print(f"Manifest status: {manifest['status']}")
    return bundle

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Bayesian inference on composite laminate data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Default: saves to results/<run_id>/
  python main.py --experimental   # Saves to results/tmp/<run_id>/
  python main.py --final          # Saves to results/final/<run_id>/
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--experimental',
        action='store_true',
        help='Save results bundle to results/tmp/<run_id>/ (for experimental/testing runs)'
    )
    mode_group.add_argument(
        '--final',
        action='store_true',
        help='Save results bundle to results/final/<run_id>/ (for important/final runs)'
    )
    
    args = parser.parse_args()
    
    # Determine output mode
    if args.experimental:
        output_mode = "experimental"
        print("🧪 Running in EXPERIMENTAL mode - results will be saved to results/tmp/<run_id>/")
    elif args.final:
        output_mode = "final"
        print("📌 Running in FINAL mode - results will be saved to results/final/<run_id>/")
    else:
        output_mode = "default"
        print("Running in default mode - results will be saved to results/<run_id>/")
    
    print("Starting inference pipeline...")

    bundle = prepare_run_bundle(config, output_mode, [sys.executable, *sys.argv])
    print(f"Prepared run bundle at {bundle.bundle_dir}")

    try:
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
        elif config["model_type"] == "model_simple":
            model = model_simple
        else:
            raise NotImplementedError(f"Model {config['model_type']} not yet implemented in main.py")

        start_time = time.time()
        print("Compiling model and warming up... (this may take a moment)")
        mcmc = run_inference(model, rng_key, data_dict, config)
        print(f"Inference completed in {time.time() - start_time:.2f}s")

        # 3. Save Results
        save_results(mcmc, bundle)
    except Exception as exc:
        mark_run_failed(bundle, str(exc))
        raise

if __name__ == "__main__":
    main()
