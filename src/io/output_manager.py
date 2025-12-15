import json
import numpyro.distributions as dist


def _serialize_distribution(d):
    """
    Convert a numpyro distribution object to a human-readable string.
    Uses full precision for reproducibility.
    Format: dist.DistributionName(param=value, ...)
    """
    dist_name = d.__class__.__name__
    params = []
    
    # Handle TruncatedDistribution (e.g., TruncatedNormal)
    if "Truncated" in dist_name:
        # TruncatedDistribution wraps a base distribution - extract params from there
        base = getattr(d, 'base_dist', None)
        if hasattr(base, 'loc'):
            params.append(f"loc={float(base.loc)}")
        if hasattr(base, 'scale'):
            params.append(f"scale={float(base.scale)}")
        # low/high are on the truncated distribution itself
        if hasattr(d, 'low'):
            params.append(f"low={float(d.low)}")
        if hasattr(d, 'high'):
            high_val = float(d.high)
            if high_val < 1e10:  # Don't show if it's effectively infinity
                params.append(f"high={high_val}")
        return f"dist.TruncatedNormal({', '.join(params)})"
    
    # Exponential
    if hasattr(d, 'rate'):
        return f"dist.Exponential({float(d.rate)})"
    
    # Normal
    if dist_name == "Normal":
        loc = float(d.loc) if hasattr(d, 'loc') else 0.0
        scale = float(d.scale) if hasattr(d, 'scale') else 1.0
        return f"dist.Normal(loc={loc}, scale={scale})"
    
    # LogNormal
    if dist_name == "LogNormal":
        loc = float(d.loc) if hasattr(d, 'loc') else 0.0
        scale = float(d.scale) if hasattr(d, 'scale') else 1.0
        return f"dist.LogNormal(loc={loc}, scale={scale})"
    
    # Fallback: try common parameters
    if hasattr(d, 'scale'):
        params.append(f"scale={float(d.scale)}")
    if hasattr(d, 'loc'):
        params.append(f"loc={float(d.loc)}")
    if hasattr(d, 'concentration'):
        params.append(f"concentration={float(d.concentration)}")
    
    if params:
        return f"dist.{dist_name}({', '.join(params)})"
    else:
        return f"dist.{dist_name}()"


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
