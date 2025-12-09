import numpy as np
import jax.numpy as jnp
from pathlib import Path

def load_experiment_data(data_path, target_angles, direction=None):
    """
    Loads experimental data using glob, filtering by angle and optional direction.
    """
    inputs = []
    extensions = []
    
    # Get all potential files
    angle_files = sorted(data_path.glob("input_load_angle_exp_*"))
    
    # Filter by direction if provided (e.g. 'h' or 'v' in filename)
    if direction:
        angle_files = [f for f in angle_files if f"_{direction}_" in f.name]
    
    for angle_file in angle_files:
        # Construct expected extension filename
        # Pattern: input_load_angle_exp_X... -> data_extension_exp_X...
        ext_filename = angle_file.name.replace("input_load_angle", "data_extension")
        ext_file = data_path / ext_filename
        
        if not ext_file.exists():
            continue

        # Load angle data
        load_angle = np.loadtxt(angle_file, delimiter=",")
        
        # Check angle
        current_angle = np.rad2deg(load_angle[0, 1])
        if np.any(np.isclose(current_angle, target_angles, atol=1e-6)):
            # Load extension data
            ext_data = np.loadtxt(ext_file, delimiter=",")
            
            # Identify valid rows (no NaNs, no negative loads)
            # load_angle shape: (N, 2), ext_data shape: (N, 3)
            
            # 1. NaN Check
            no_nans = ~np.isnan(load_angle).any(axis=1) & ~np.isnan(ext_data).any(axis=1)
            
            # 2. Negative Load Check (allow effectively zero, remove anything significantly negative)
            # Using -1e-6 as tolerance to keep floating point zeros
            non_negative_load = load_angle[:, 0] >= -1e-6
            
            valid_mask = no_nans & non_negative_load
            
            if np.any(valid_mask):
                inputs.append(load_angle[valid_mask])
                extensions.append(ext_data[valid_mask])
            
    return inputs, extensions

def load_sim_file(path, filename):
    """Helper to load simulation data directly into a JAX array."""
    return jnp.array(np.loadtxt(path / filename, delimiter=","))

def truncate_data(inputs, extensions, max_load):
    """Truncates data arrays to rows where load <= max_load."""
    inputs_trunc = []
    extensions_trunc = []
    for inp, ext in zip(inputs, extensions):
        mask = inp[:, 0] <= max_load
        inputs_trunc.append(inp[mask])
        extensions_trunc.append(ext[mask])
    return inputs_trunc, extensions_trunc

def load_all_data(config):
    """
    Main function to load all required data based on config.
    Returns both full and truncated datasets.
    """
    # Use full experimental data path
    base_path = Path(config["data"]["base_path"])
    full_exp_path = base_path / "experimental" / "full_experimental_data"
    
    sim_path_h = base_path / "simulation" / "h"
    sim_path_v = base_path / "simulation" / "v"
    
    angles = config["data"]["angles"]
    max_load = config["data"].get("max_load", 10.0)
    
    # 1. Load Full Experimental Data
    input_xy_exp_full, data_exp_h_full_raw = load_experiment_data(full_exp_path, angles, direction='h')
    _, data_exp_v_full_raw = load_experiment_data(full_exp_path, angles, direction='v')
    
    # 2. Truncate Data for Inference
    input_xy_exp_trunc, data_exp_h_trunc_raw = truncate_data(input_xy_exp_full, data_exp_h_full_raw, max_load)
    _, data_exp_v_trunc_raw = truncate_data(input_xy_exp_full, data_exp_v_full_raw, max_load) # Input xy same for h/v per angle? Assuming yes/compatible
    
    # 3. Load Simulation Data
    input_xy_sim = load_sim_file(sim_path_h, "input_load_angle_sim.txt")
    input_theta_sim = load_sim_file(sim_path_h, "input_theta_sim.txt")
    data_sim_h = load_sim_file(sim_path_h, "data_extension_sim.txt").mean(axis=1)
    data_sim_v = load_sim_file(sim_path_v, "data_extension_sim.txt").mean(axis=1)
    
    # 4. Process For Inference (Mean across sensors)
    data_exp_h_mean = [d.mean(axis=1) for d in data_exp_h_trunc_raw]
    data_exp_v_mean = [d.mean(axis=1) for d in data_exp_v_trunc_raw]
    
    # Process Full Data Mean (for plotting averaged)
    data_exp_h_full_mean = [d.mean(axis=1) for d in data_exp_h_full_raw]
    data_exp_v_full_mean = [d.mean(axis=1) for d in data_exp_v_full_raw]

    return {
        # Truncated (for Model)
        "input_xy_exp": input_xy_exp_trunc,
        "data_exp_h": data_exp_h_mean,
        "data_exp_v": data_exp_v_mean,
        "data_exp_h_raw": data_exp_h_trunc_raw, # For prediction plots (raw, truncated)
        "data_exp_v_raw": data_exp_v_trunc_raw,
        
        # Full (for Initial Plots)
        "input_xy_exp_full": input_xy_exp_full,
        "data_exp_h_full_raw": data_exp_h_full_raw,
        "data_exp_v_full_raw": data_exp_v_full_raw,
        "data_exp_h_full_mean": data_exp_h_full_mean,
        "data_exp_v_full_mean": data_exp_v_full_mean,
        
        # Simulation
        "input_xy_sim": input_xy_sim,
        "input_theta_sim": input_theta_sim,
        "data_sim_h": data_sim_h,
        "data_sim_v": data_sim_v
    }
