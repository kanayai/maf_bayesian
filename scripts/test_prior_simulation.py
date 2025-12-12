
import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.data_loader import load_all_data
from configs.default_config import config
from src.core.models import sample_prior_predictive_curves

def test_prior_simulation_combined():
    # We want to test 3 angles for both H and V directions
    angles_deg = [45, 90, 135]
    directions = ["h", "v"]
    
    test_loads = jnp.linspace(0, 11, 50) # up to 11kN as per max_load
    test_angles_rad = jnp.deg2rad(jnp.array(angles_deg))
    
    rng_key = jax.random.PRNGKey(42)
    num_samples = 500
    
    # 2 Rows (Directions), 3 Columns (Angles)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for row_idx, direction in enumerate(directions):
        print(f"\nProcessing Direction: {direction.upper()}")
        
        # Override config for this direction
        config["data"]["angles"] = angles_deg
        config["data"]["direction"] = direction
        
        # Load Data
        all_data = load_all_data(config)
        
        sim_data = {
            'input_xy_sim': all_data['input_xy_sim'],
            'input_theta_sim': all_data['input_theta_sim'],
            'data_sim_h': all_data['data_sim_h'],
            'data_sim_v': all_data['data_sim_v']
        }
        print(f"DEBUG: input_xy_sim shape: {sim_data['input_xy_sim'].shape}")
        print(f"DEBUG: input_xy_sim head:\n{sim_data['input_xy_sim'][:5]}")
        
        # Sample Prior Curves
        means, sigmas, samples_path, test_xy = sample_prior_predictive_curves(
            num_samples=num_samples,
            rng_key=rng_key,
            config=config,
            sim_data=sim_data,
            test_loads=test_loads,
            test_angles=test_angles_rad
        )
        
        # Plotting
        points_per_angle = len(test_loads)
        
        exp_inputs = all_data['input_xy_exp']
        exp_data = all_data[f'data_exp_{direction}'] # 'data_exp_h' or 'data_exp_v'
        
        for col_idx, angle in enumerate(angles_deg):
            ax = axes[row_idx, col_idx]
            
            # --- 1. Plot Simulation Curves ---
            start_idx = col_idx * points_per_angle
            end_idx = (col_idx + 1) * points_per_angle
            
            # test_xy slice for this angle
            # Axes Swap: X=Extension (means), Y=Load (test_loads)
            
            # Collect all samples for this angle to compute percentiles
            angle_samples = []
            for s in range(num_samples):
                # Use samples_path instead of means (latent f samples)
                exts = samples_path[s, start_idx:end_idx]
                angle_samples.append(exts)
                ax.plot(exts, test_loads, color='blue', alpha=0.02)
            
            # Compute and Plot Percentiles (2.5% and 97.5%)
            # Stack samples: (N_samples, N_load_points)
            angle_samples = np.array(angle_samples)
            # Percentiles along axis 0 (across samples) for each load point
            # pct_vals shape: (2, N_load_points)
            pct_vals = np.percentile(angle_samples, [2.5, 97.5], axis=0)
            
            # Plot bounds
            ax.plot(pct_vals[0], test_loads, color='green', linestyle='--', linewidth=2, label='2.5% / 97.5%')
            ax.plot(pct_vals[1], test_loads, color='green', linestyle='--', linewidth=2)
                
            # --- 2. Overlay Experimental Data ---
            for exp_idx, inp in enumerate(exp_inputs):
                # inp is (N, 2) [Load, Angle]
                exp_angle_rad = np.mean(inp[:, 1])
                exp_angle_deg = np.rad2deg(exp_angle_rad)
                
                # Check if this experiment matches the current column angle
                if np.abs(exp_angle_deg - angle) < 2.0:
                    loads = inp[:, 0]
                    # Get corresponding extension data
                    # Note: all_data['data_exp_h'] is a list of arrays
                    exts_exp = exp_data[exp_idx]
                    
                    # Axes Swap: X=Extension, Y=Load
                    ax.scatter(exts_exp, loads, color='red', s=10, alpha=0.6)
            
            # Labels and Titles
            if row_idx == 0:
                ax.set_title(f"Angle {angle}°")
            
            if row_idx == 1:
                ax.set_xlabel("Extension (mm)")
            
            if col_idx == 0:
                ax.set_ylabel(f"{direction.upper()} - Load (kN)")
                ax.legend(loc='upper left', fontsize=8)
            
            # Special formatting for Normal (v) at 90 degrees
            if direction == "v":
                if int(angle) == 90:
                    ax.set_xlim(-0.05, 0.05) # Keep tight for vertically stiff 90 deg
                elif int(angle) == 45:
                    ax.set_xlim(-0.05, 0.1)
                elif int(angle) == 135:
                    ax.set_xlim(-0.1, 0.1)
                else:
                    ax.set_xlim(-0.05, 0.05)
            else:
                # Shear (Horizontal)
                if int(angle) == 90:
                    ax.set_xlim(-0.05, 0.15)
                elif int(angle) == 45:
                    ax.set_xlim(-0.05, 0.1)
                elif int(angle) == 135:
                    ax.set_xlim(-0.05, 0.1)
                else:
                    ax.set_xlim(0, 0.15)
            ax.grid(True)
            # Invert Y axis? Usually Load-Extension plots have Load on Y ascending.
            # No inversion needed if 0 is at bottom.
            
    plt.tight_layout()
    output_path = "figures/prior_simulation_test_combined.png"
    plt.savefig(output_path)
    print(f"Combined plot saved to {output_path}")

if __name__ == "__main__":
    test_prior_simulation_combined()
