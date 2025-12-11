import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Mock numpyro to avoid dependency issues when just reading config
from unittest.mock import MagicMock
sys.modules["numpyro"] = MagicMock()
sys.modules["numpyro.distributions"] = MagicMock()

from configs.default_config import config

def load_file_content(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception as e:
        print(f"Error loading {path.name}: {e}")
        return None

def find_matching_full_data(curr_data, full_files):
    """
    Finds a matching full data file for a given truncated data file.
    Matches if the truncated data is a subset (prefix/internal) of the full data.
    """
    if curr_data is None or len(curr_data) == 0:
        return None

    # Brute force search sufficient for these small files
    match_val = curr_data[0, 0] # Match first load value
    
    for full_file in full_files:
        full_data = load_file_content(full_file)
        if full_data is None: continue
        
        # Find potential start indices
        matches = np.where(np.isclose(full_data[:, 0], match_val, atol=1e-5))[0]
        
        for start_idx in matches:
            end_idx = start_idx + len(curr_data)
            if end_idx <= len(full_data):
                sub_arr = full_data[start_idx:end_idx]
                if np.allclose(curr_data, sub_arr, equal_nan=True):
                    return full_data, full_file
    
    return None, None

def main():
    max_load = config["data"].get("max_load", 10.0)
    print(f"🔄 Updating truncated data files to Max Load: {max_load} kN")
    
    base_path = project_root / "data" / "experimental"
    full_path = base_path / "full_experimental_data"
    
    if not full_path.exists():
        print(f"❌ Full data directory not found: {full_path}")
        return

    full_files = sorted(list(full_path.glob("input_load_angle_exp_*.txt")))
    print(f"📚 Found {len(full_files)} source files in full dataset.")

    dirs_to_update = ["h", "v"]
    
    for direction in dirs_to_update:
        target_dir = base_path / direction
        if not target_dir.exists():
            continue
            
        print(f"\n📂 Processing directory: {direction}/")
        target_files = sorted(list(target_dir.glob("input_load_angle_exp_*.txt")))
        
        updated_count = 0
        
        for target_file in target_files:
            # Load existing just to identify which file it corresponds to
            curr_data = load_file_content(target_file)
            
            # Find matching full data
            full_data, full_file_path = find_matching_full_data(curr_data, full_files)
            
            if full_data is not None:
                # Truncate full data to current max_load
                mask = full_data[:, 0] <= max_load
                new_data = full_data[mask]
                
                # Check if update is needed (length change or content change)
                # We overwrite anyway to be sure, but let's log size diff
                old_len = len(curr_data)
                new_len = len(new_data)
                
                # Save input load/angle
                np.savetxt(target_file, new_data, delimiter=",", fmt="%.16e")
                
                # Update corresponding extension file
                target_ext_file = target_file.parent / target_file.name.replace("input_load_angle", "data_extension")
                full_ext_file = full_file_path.parent / full_file_path.name.replace("input_load_angle", "data_extension")
                
                if full_ext_file.exists():
                    full_ext_data = load_file_content(full_ext_file)
                    new_ext_data = full_ext_data[mask]
                    np.savetxt(target_ext_file, new_ext_data, delimiter=",", fmt="%.16e")
                    
                    print(f"  ✅ Updated {target_file.name}: {old_len} -> {new_len} rows")
                    updated_count += 1
                else:
                    print(f"  ⚠️  Extension file missing for {full_file_path.name}")
            else:
                print(f"  ❓ No match found for {target_file.name}")

    print(f"\n✨ Update complete. {updated_count} files processed.")

if __name__ == "__main__":
    main()
