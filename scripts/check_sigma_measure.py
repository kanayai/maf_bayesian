
import arviz as az
import numpy as np
from pathlib import Path
import jax.numpy as jnp

def main():
    results_dir = Path(".")
    
    files = list(results_dir.glob("results/no_bias_*.nc"))
    if not files:
        files = list(results_dir.glob("results/*.nc"))
    if not files:
        print("No result files found.")
        return

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading {latest_file}")
    idata = az.from_netcdf(latest_file)

    if "sigma_measure" in idata.posterior.data_vars:
        sm = idata.posterior["sigma_measure"].values.flatten()
        print(f"sigma_measure: Mean={np.mean(sm):.6f}, Std={np.std(sm):.6f}, Min={np.min(sm):.6f}, Max={np.max(sm):.6f}")
    else:
        print("sigma_measure not found in posterior.")

    if "sigma_measure_n" in idata.posterior.data_vars:
        smn = idata.posterior["sigma_measure_n"].values.flatten()
        print(f"sigma_measure_n: Mean={np.mean(smn):.6f}, Std={np.std(smn):.6f}")

if __name__ == "__main__":
    main()
