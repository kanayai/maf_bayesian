
import arviz as az
import numpy as np
from pathlib import Path

def main():
    results_dir = Path(".")
    
    # regex glob for result files
    files = list(results_dir.glob("results/no_bias_*.nc"))
    if not files:
        files = list(results_dir.glob("results/*.nc"))
        
    if not files:
        print("No result files found.")
        return

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading {latest_file}")
    idata = az.from_netcdf(latest_file)
    
    print("\n--- Posterior Statistics ---")
    for key in ["gamma_scale_v", "gamma_scale_h", "sigma_measure"]:
        if key in idata.posterior.data_vars:
            vals = idata.posterior[key].values.flatten()
            print(f"{key}: Mean={np.mean(vals):.6f}, Std={np.std(vals):.6f}, Min={np.min(vals):.6f}, Max={np.max(vals):.6f}")
        else:
            print(f"{key}: Not found")

if __name__ == "__main__":
    main()
