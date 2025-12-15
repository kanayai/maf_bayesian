import json
import numpyro.distributions as dist


def _serialize_distribution(d):
    """
    Convert a numpyro distribution object to a human-readable string.
    Extracts the key parameters (rate, scale, loc) from the distribution.
    """
    dist_name = d.__class__.__name__
    params = []
    
    # Common parameters for various distributions
    if hasattr(d, 'rate'):
        params.append(f"rate={float(d.rate):.6g}")
    if hasattr(d, 'scale') and not hasattr(d, 'rate'):  # Avoid duplication for Exponential
        params.append(f"scale={float(d.scale):.6g}")
    if hasattr(d, 'loc'):
        loc_val = float(d.loc)
        if loc_val != 0.0:  # Only show if non-zero
            params.append(f"loc={loc_val:.6g}")
    if hasattr(d, 'concentration'):
        params.append(f"concentration={float(d.concentration):.6g}")
    
    if params:
        return f"{dist_name}({', '.join(params)})"
    else:
        return f"{dist_name}()"


def _config_encoder(obj):
    """
    Custom JSON encoder for config objects.
    Handles numpyro distributions and Path objects.
    """
    # Check if it's a numpyro distribution
    if isinstance(obj, dist.Distribution):
        return _serialize_distribution(obj)
    # Fallback to string representation
    return str(obj)


def save_config_log(config, output_dir, results_filename):
    """
    Saves the configuration dictionary to a Markdown file.
    
    Args:
        config (dict): The configuration dictionary.
        output_dir (Path): The directory to save the log file.
        results_filename (str or Path): The name of the .nc results file used.
    """
    import subprocess

    def get_git_revision_hash():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "Unknown (Git error or not a repo)"

    log_path = output_dir / "config_log.md"
    
    git_hash = get_git_revision_hash()
    
    with open(log_path, "w") as f:
        f.write("# Analysis Configuration Log\n\n")
        
        f.write("## Reproducibility Info\n")
        f.write(f"- **Git Commit**: `{git_hash}`\n")
        f.write(f"- **Results File**: `{results_filename}`\n\n")

        f.write("## Configuration Settings\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=4, default=_config_encoder))
        f.write("\n```\n\n")
    
    print(f"Saved configuration log to {log_path}")
