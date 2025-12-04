import numpy as np
import jax.numpy as jnp
from pathlib import Path

def load_experiment_data(data_path, target_angles):
    """
    Loads experimental data from the specified path, filtering by target angles.
    
    Args:
        data_path (Path): Path to the directory containing data files (e.g. data/experimental/h).
        target_angles (list): List of angles (in degrees) to include.
        
    Returns:
        inputs: List of load/angle arrays (input_xy).
        extensions: List of extension data arrays (data_extension).
    """
    inputs = []
    extensions = []
    
    # Get sorted file lists to ensure matching pairs of input and data files
    angle_files = sorted(data_path.glob("input_load_angle_exp_*"))
    ext_files   = sorted(data_path.glob("data_extension_exp_*"))
    
    for angle_file, ext_file in zip(angle_files, ext_files):
        # Load angle data (contains load and angle information)
        load_angle = np.loadtxt(angle_file, delimiter=",")
        
        # Check if the angle matches one of our targets
        # Using isclose for robust floating point comparison against the list of target angles
        current_angle = np.rad2deg(load_angle[0, 1])
        if np.any(np.isclose(current_angle, target_angles, atol=1e-6)):
            inputs.append(load_angle)
            # Store all columns (Left, Center, Right) instead of mean
            extensions.append(np.loadtxt(ext_file, delimiter=","))
            
    return inputs, extensions

def load_sim_file(path, filename):
    """Helper to load simulation data directly into a JAX array."""
    return jnp.array(np.loadtxt(path / filename, delimiter=","))

def load_all_data(config):
    """
    Main function to load all required data based on config.
    """
    base_path = Path(config["data"]["base_path"])
    exp_path_h = base_path / "experimental" / "h"
    exp_path_v = base_path / "experimental" / "v"
    sim_path_h = base_path / "simulation" / "h"
    sim_path_v = base_path / "simulation" / "v"
    
    angles = config["data"]["angles"]
    
    # 1. Load Horizontal Experimental Data
    input_xy_exp, data_exp_h_raw = load_experiment_data(exp_path_h, angles)
    
    # 2. Load Vertical Experimental Data
    # We ignore inputs here, assuming they match horizontal
    _, data_exp_v_raw = load_experiment_data(exp_path_v, angles)
    
    # 3. Load Simulation Data
    input_xy_sim = load_sim_file(sim_path_h, "input_load_angle_sim.txt")
    input_theta_sim = load_sim_file(sim_path_h, "input_theta_sim.txt")
    
    # Simulation data is usually mean already or single column? 
    # In original code: .mean(axis=1) was used on sim data too.
    # Let's check if they are multi-column.
    # Original: load_sim_file(...).mean(axis=1)
    data_sim_h = load_sim_file(sim_path_h, "data_extension_sim.txt").mean(axis=1)
    data_sim_v = load_sim_file(sim_path_v, "data_extension_sim.txt").mean(axis=1)
    
    # 4. Process Experimental Data for Inference (Mean across sensors)
    # The model expects lists of arrays for experimental data if we want to keep the structure
    # But for inference we often take the mean.
    # In original MAF_gp_hv.py: data_exp_h_mean = [d.mean(axis=1) for d in data_exp_h]
    data_exp_h_mean = [d.mean(axis=1) for d in data_exp_h_raw]
    data_exp_v_mean = [d.mean(axis=1) for d in data_exp_v_raw]
    
    return {
        "input_xy_exp": input_xy_exp, # List of arrays
        "data_exp_h": data_exp_h_mean, # List of arrays (meaned)
        "data_exp_v": data_exp_v_mean, # List of arrays (meaned)
        "data_exp_h_raw": data_exp_h_raw, # Raw data for plotting
        "data_exp_v_raw": data_exp_v_raw, # Raw data for plotting
        "input_xy_sim": input_xy_sim,
        "input_theta_sim": input_theta_sim,
        "data_sim_h": data_sim_h,
        "data_sim_v": data_sim_v
    }
