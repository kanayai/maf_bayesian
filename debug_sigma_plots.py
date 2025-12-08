
import arviz as az
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy.stats import expon

def main():
    print("Starting debug analysis for sigma parameters...")
    
    # 1. Find Data
    results_dir = Path("results")
    files = list(results_dir.glob("*.nc"))
    if not files:
        print("No result files found in results/")
        return
        
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from {latest_file}")
    idata = az.from_netcdf(latest_file)
    
    # 2. Extract Samples
    samples = {}
    for key in ["sigma_emulator", "sigma_measure"]:
        if key in idata.posterior.data_vars:
            data = idata.posterior[key].values
            flat_data = data.reshape(-1) # Flatten chains and draws
            samples[key] = flat_data
            print(f"Loaded {key}: {len(flat_data)} samples")
        else:
            print(f"Warning: {key} not found in posterior!")
            
    # 3. Check for Negatives
    print("\n--- Value Analysis ---")
    for key, val in samples.items():
        min_val = np.min(val)
        max_val = np.max(val)
        n_neg = np.sum(val < 0)
        print(f"{key}: Min = {min_val:.6e}, Max = {max_val:.6e}, Negative Count = {n_neg}")
        
    # 4. Plotting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = Path("figures") / f"debug_sigma_{timestamp}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving debug figures to {figures_dir}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, key in enumerate(samples.keys()):
        val = samples[key]
        ax = axes[i]
        
        # Hist
        sns.histplot(val, ax=ax, stat="density", kde=False, bins=50, label="Posterior", alpha=0.5)
        
        # Prior Line (Exponential)
        # sigma_measure ~ Exp(rate=100) -> scale = 0.01
        # sigma_emulator ~ Exp(rate=20) -> scale = 0.05
        # Note: These are defined in analyze.py, repeating logic here for viz
        if key == "sigma_measure":
            rate = 100.0
        elif key == "sigma_emulator":
            rate = 20.0
        else:
            rate = 1.0 # Should not happen
            
        x_grid = np.linspace(0, np.max(val), 200)
        ax.plot(x_grid, expon.pdf(x_grid, scale=1/rate), color='green', lw=2, label=f"Prior Exp({rate})")
        
        ax.set_title(key)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(figures_dir / "sigma_plots_standard.png")
    plt.close()
    
    # Zoomed Plots (Vertical Zoom)
    # The user asked to "change the vertical scale ... to something much smaller"
    # This usually means zooming in heavily on the Y-axis to see the bottom of the bars/lines.
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, key in enumerate(samples.keys()):
        val = samples[key]
        ax = axes[i]
        
        # Use more bins for detail
        sns.histplot(val, ax=ax, stat="density", kde=False, bins=100, label="Posterior", alpha=0.5)
        
        # Prior
        if key == "sigma_measure": rate = 100.0
        else: rate = 20.0
        x_grid = np.linspace(0, np.max(val), 200)
        ax.plot(x_grid, expon.pdf(x_grid, scale=1/rate), color='green', lw=2, label="Prior")
        
        # ZOOM Y-Axis
        # We want to see if kde/hist goes negative (impossible for hist) or where it touches 0
        # Let's set the ylim to a small fraction of the max density
        # Actually, if the concern is "plot going negative", maybe they saw a KDE line dipping below 0?
        # KDEs can oversmooth into negative regions. Since we use kde=False, we just see bins.
        # But analyze.py uses kde=False.
        
        # Let's zoom in on the bottom 10% of the plot
        # Get current y-limits
        # y_max = ax.get_ylim()[1]
        # ax.set_ylim(-0.05 * y_max, 0.2 * y_max) # Show a bit of negative space and the bottom
        ax.set_ylim(bottom=-0.1) # Specifically show negative space
        
        ax.set_title(f"{key} (Zoomed Y)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(figures_dir / "sigma_plots_zoomed_y.png")
    plt.close()

if __name__ == "__main__":
    main()
