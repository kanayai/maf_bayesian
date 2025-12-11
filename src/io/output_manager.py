import json

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
        f.write(json.dumps(config, indent=4, default=str)) # default=str to handle Path objects
        f.write("\n```\n\n")
    
    print(f"Saved configuration log to {log_path}")
