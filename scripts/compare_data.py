import numpy as np
from pathlib import Path
import os

def load_file_content(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def compare_arrays(arr1, arr2):
    # Check if arr1 is a subset of arr2 (allowing for different start point and truncation)
    # Check if arr1 is found within arr2
    
    # We look for the first element of arr1 in arr2
    if len(arr1) == 0: return "EMPTY"
    
    # Simple brute force search (sufficient for small arrays)
    matches = np.where(np.isclose(arr2[:,0], arr1[0,0], atol=1e-5))[0]
    
    for start_idx in matches:
        # Check if subsequent elements match
        end_idx = start_idx + len(arr1)
        if end_idx <= len(arr2):
            sub_arr2 = arr2[start_idx:end_idx]
            if np.allclose(arr1, sub_arr2, equal_nan=True):
                if start_idx == 0 and end_idx == len(arr2):
                    return "IDENTICAL"
                elif start_idx == 0:
                    return f"PREFIX SUBSET (Matches first {len(arr1)} rows)"
                elif end_idx == len(arr2):
                    return f"SUFFIX SUBSET (Matches last {len(arr1)} rows)"
                else:
                    return f"INTERNAL SUBSET (Matches rows {start_idx} to {end_idx-1})"
        else:
             # arr1 is longer than the remaining part of arr2
             # Check if arr2 is a prefix of arr1 (unlikely for "Full" data, but possible)
             pass

    return "DIFFERENT"

def find_subset_indices(arr1, arr2):
    """Returns (status, start_index, end_index) if arr1 is found in arr2."""
    if len(arr1) == 0: return ("EMPTY", 0, 0)
    
    matches = np.where(np.isclose(arr2[:,0], arr1[0,0], atol=1e-5))[0]
    
    for start_idx in matches:
        end_idx = start_idx + len(arr1)
        if end_idx <= len(arr2):
            sub_arr2 = arr2[start_idx:end_idx]
            if np.allclose(arr1, sub_arr2, equal_nan=True):
                return ("MATCH", start_idx, end_idx)
    return ("DIFFERENT", -1, -1)

def main():
    # context: script is in scripts/, data is in data/experimental
    # Use parent of parent to go from scripts/compare_data.py -> project_root -> data
    base_path = Path(__file__).resolve().parent.parent / "data" / "experimental"
    current_path_h = base_path / "h"
    current_path_v = base_path / "v"
    full_path = base_path / "full_experimental_data"
    
    print("Listing Current Data Files:")
    current_files_h = sorted(list(current_path_h.glob("input_load_angle_exp_*.txt")))
    full_files_load = sorted(list(full_path.glob("input_load_angle_exp_*.txt")))
    
    print(f"Found {len(current_files_h)} current files (h) and {len(full_files_load)} full files.")
    
    # For each current file, find a matching full file
    for curr_file in current_files_h:
        curr_data = load_file_content(curr_file)
        if curr_data is None: continue
        
        # Get the corresponding extension file
        curr_ext_file = curr_file.parent / curr_file.name.replace("input_load_angle", "data_extension")
        curr_ext_data = load_file_content(curr_ext_file)
        
        match_found = False
        print(f"\nChecking {curr_file.name} (Max Load: {curr_data[:,0].max():.3f})...")
        
        for full_file in full_files_load:
            full_data = load_file_content(full_file)
            if full_data is None: continue
            
            # Compare Load/Angle
            status_load = compare_arrays(curr_data, full_data)
            
            if status_load != "DIFFERENT":
                # Check Extension
                full_ext_file = full_file.parent / full_file.name.replace("input_load_angle", "data_extension")
                full_ext_data = load_file_content(full_ext_file)
                
                status_ext = compare_arrays(curr_ext_data, full_ext_data)
                
                print(f"  MATCH FOUND: {full_file.name}")
                print(f"    Subset Type: {status_load}")
                # Parse indices from status string or pass them back
                # Let's verify exact indices again for reporting
                subset_load, start, end = find_subset_indices(curr_data, full_data)
                print(f"    Reconstruction: full_data[{start}:{end}]")
                print(f"    Full Max Load: {full_data[:,0].max():.3f}")
                match_found = True
                break
        
        if not match_found:
            print("  NO MATCH FOUND in full dataset.")

if __name__ == "__main__":
    main()
