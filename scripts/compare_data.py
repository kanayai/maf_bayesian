
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.data_loader import load_all_data
from configs.default_config import config
import matplotlib.pyplot as plt
import numpy as np

def compare_data():
    data = load_all_data(config)
    
    # We want truncated raw data to match what the model sees
    # data "data_exp_h_raw" is a list of arrays (one per angle)
    # Each array is (N, 3) 
    
    angles = config['data']['angles']
    
    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, angle in enumerate(angles):
        ax = axes[i]
        
        # Get raw data for this angle
        # Shape: (N_experiments, N_points, 3 sensors) usually, but loader returns list of (N_points, 3)
        # Wait, load_experiment_data returns 'inputs' and 'extensions' as lists of arrays.
        # Each element in list corresponds to one experiment file. 
        # But 'load_all_data' merges them? No.
        
        # Let's check data_loader.py again. 
        # load_experiment_data returns inputs (list), extensions (list).
        # load_all_data returns truncated lists in 'data_exp_h_raw'.
        # Since currently we only have ONE file per angle typically, the list has length 1 (or small number).
        # We need to match H and V data for the SAME experiment.
        
        h_data_list = data['data_exp_h_raw'] # List of (N, 3) arrays
        v_data_list = data['data_exp_v_raw'] # List of (N, 3) arrays
        
        # We assume they align by index if they come from sorted glob.
        # Let's iterate and concat all points for this angle.
        
        # However, load_all_data flattens everything?
        # No, load_experiment_data iterates angles. 
        # Ah, 'load_experiment_data' returns a list of ALL valid files found matching ANY angle.
        # This is tricky because we need to separate by angle for plotting.
        
        # Let's inspect 'inputs' to filter by angle.
        inputs_list = data['input_xy_exp']
        
        shear_vals = []
        normal_vals = []
        
        for j, inp in enumerate(inputs_list):
            # inp is (N, 2), columns: [Load, Angle]
            # Check if this experiment matches current angle
            # We use mean angle of the experiment
            exp_angle_rad = np.mean(inp[:, 1]) 
            exp_angle_deg = np.rad2deg(exp_angle_rad)
            
            if np.isclose(exp_angle_deg, angle, atol=1.0):
                # This experiment is for the current angle
                h_ext = h_data_list[j] # (N, 3)
                v_ext = v_data_list[j] # (N, 3)
                
                # Average across sensors (L, C, R)
                h_mean = h_ext.mean(axis=1) # (N,)
                v_mean = v_ext.mean(axis=1) # (N,)
                
                shear_vals.extend(h_mean)
                normal_vals.extend(v_mean)
        
        if not shear_vals:
            print(f"No data found for {angle}")
            continue
            
        shear_vals = np.array(shear_vals)
        normal_vals = np.array(normal_vals)
        
        # Correlation
        corr = np.corrcoef(shear_vals, normal_vals)[0, 1]
        
        # Plot
        ax.scatter(shear_vals, normal_vals, alpha=0.5)
        ax.set_title(f"Angle {angle}°\nCorrelation: {corr:.4f}")
        ax.set_xlabel("Shear Extension (mm)")
        ax.set_ylabel("Normal Extension (mm)")
        ax.grid(True)
        
    plt.tight_layout()
    output_path = "figures/correlation_check.png"
    plt.savefig(output_path)
    print(f"Correlation plot saved to {output_path}")

if __name__ == "__main__":
    compare_data()
