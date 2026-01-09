import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# Set global style settings
matplotlib.rcParams["axes.formatter.limits"] = [-4,4]
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = 'sans-serif'

def plot_experimental_data(data_dict, save_path=None):
    """
    Plots experimental data (Shear and Normal) with color-coded angles and position markers.
    """
    input_xy_exp = data_dict["input_xy_exp"]
    data_exp_h = data_dict["data_exp_h_raw"]
    data_exp_v = data_dict["data_exp_v_raw"]
    
    plt.figure(figsize=(15, 6))
    
    # Settings
    angle_colors = {45: 'r', 90: 'g', 135: 'b'} 
    pos_markers = ['o', 's', '^'] 
    
    # Helper for plotting one direction
    def plot_direction(ax, data_exp, title, xlabel):
        for i in range(len(data_exp)):
            load = input_xy_exp[i][:, 0]
            angle = np.rad2deg(input_xy_exp[i][0, 1])
            angle_key = int(round(angle))
            c = angle_colors.get(angle_key, 'k')
            
            for col in range(3): # 3 positions
                ax.plot(data_exp[i][:, col], load, color=c, marker=pos_markers[col], 
                        linestyle='-', alpha=0.3, markersize=6)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Load [kN]')
        ax.set_title(title)
        ax.grid(True)

    # Plot Shear
    ax1 = plt.subplot(1, 2, 1)
    plot_direction(ax1, data_exp_h, 'Shear Extension', 'Shear Extension [mm]')
    
    # Custom Legend
    custom_lines_ang = [Line2D([0], [0], color='k', lw=0, label='Angles:'),
                        Line2D([0], [0], color=angle_colors[45], lw=2, label='45°'),
                        Line2D([0], [0], color=angle_colors[90], lw=2, label='90°'),
                        Line2D([0], [0], color=angle_colors[135], lw=2, label='135°')]
    custom_lines_pos = [Line2D([0], [0], color='k', lw=0, label='Positions:'),
                        Line2D([0], [0], color='k', marker=pos_markers[0], linestyle='None', label='Left'),
                        Line2D([0], [0], color='k', marker=pos_markers[1], linestyle='None', label='Center'),
                        Line2D([0], [0], color='k', marker=pos_markers[2], linestyle='None', label='Right')]
    
    ax1.legend(handles=custom_lines_ang + custom_lines_pos, loc='best')

    # Plot Normal
    ax2 = plt.subplot(1, 2, 2)
    plot_direction(ax2, data_exp_v, 'Normal Extension', 'Normal Extension [mm]')
    ax2.legend(handles=custom_lines_ang + custom_lines_pos, loc='best')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved experimental data plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_averaged_experimental_data(data_dict, save_path=None):
    """
    Plots averaged experimental data (Shear and Normal) across positions with color-coded angles.
    """
    input_xy_exp = data_dict["input_xy_exp"]
    data_exp_h = data_dict["data_exp_h_raw"]
    data_exp_v = data_dict["data_exp_v_raw"]
    
    plt.figure(figsize=(15, 6))
    
    # Settings
    angle_colors = {45: 'r', 90: 'g', 135: 'b'} 
    
    # Helper for plotting one direction
    def plot_direction(ax, data_exp, title, xlabel):
        for i in range(len(data_exp)):
            load = input_xy_exp[i][:, 0]
            angle = np.rad2deg(input_xy_exp[i][0, 1])
            angle_key = int(round(angle))
            c = angle_colors.get(angle_key, 'k')
            
            # Calculate mean across columns (positions)
            mean_ext = np.mean(data_exp[i], axis=1)
            
            ax.plot(mean_ext, load, color=c, marker='o', 
                    linestyle='-', alpha=0.6, markersize=4, label=f"{angle_key}°" if i < 3 else "")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Load [kN]')
        ax.set_title(title)
        ax.grid(True)

    # Plot Shear
    ax1 = plt.subplot(1, 2, 1)
    plot_direction(ax1, data_exp_h, 'Averaged Shear Extension', 'Shear Extension [mm]')
    
    # Custom Legend
    custom_lines_ang = [Line2D([0], [0], color=angle_colors[45], lw=2, label='45°'),
                        Line2D([0], [0], color=angle_colors[90], lw=2, label='90°'),
                        Line2D([0], [0], color=angle_colors[135], lw=2, label='135°')]
    
    ax1.legend(handles=custom_lines_ang, loc='best')

    # Plot Normal
    ax2 = plt.subplot(1, 2, 2)
    plot_direction(ax2, data_exp_v, 'Averaged Normal Extension', 'Normal Extension [mm]')
    ax2.legend(handles=custom_lines_ang, loc='best')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved averaged experimental data plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_posterior_distributions(samples, prior_pdf_fn=None, prior_samples=None, save_path=None, layout_rows=None, shared_xlim_groups=None):
    """
    Plots histograms of posterior samples for all parameters.
    If prior_pdf_fn is provided, plots analytical prior density as a green line.
    prior_pdf_fn(key, x_vals) -> pdf_vals
    If prior_samples is provided (dict), plots prior histogram for matching keys.
    If shared_xlim_groups is provided (dict), parameters in the same group share x-axis range.
    """
    keys = list(samples.keys())
    num_vars = len(keys)
    
    if layout_rows:
        rows = layout_rows
        cols = (num_vars + rows - 1) // rows
    else:
        cols = 4
        rows = (num_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Ensure axes is always a flat array/list of Axes objects
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Pre-compute shared xlim ranges if groups are specified
    shared_xlim_cache = {}
    if shared_xlim_groups:
        for group_name, group_keys in shared_xlim_groups.items():
            # Compute union of ranges for all keys in the group
            group_min, group_max = float('inf'), float('-inf')
            for gk in group_keys:
                if gk in samples:
                    vals = np.asarray(samples[gk])
                    p_min, p_max = np.percentile(vals, [1, 99])
                    group_min = min(group_min, float(p_min))
                    group_max = max(group_max, float(p_max))
            
            if group_min < float('inf'):
                p_range = group_max - group_min
                padding = 0.05 * p_range if p_range > 1e-12 else 0.1
                group_xlim = (group_min - padding, group_max + padding)
                # Cache for each key in the group
                for gk in group_keys:
                    shared_xlim_cache[gk] = group_xlim
    
    for i, key in enumerate(keys):
        # Posterior: Histogram only (stat="density" to match KDE scale)
        # Convert to numpy to avoid matplotlib hanging with JAX arrays
        samples_np = np.asarray(samples[key])
        sns.histplot(samples_np, ax=axes[i], kde=False, stat="density", label="Posterior", alpha=0.4)
        
        # Prior: Samples (Histogram)
        if prior_samples is not None and key in prior_samples:
            # Flatten if necessary (e.g. if chains dim exists)
            p_s = np.asarray(prior_samples[key]).flatten()
            sns.histplot(p_s, ax=axes[i], kde=False, stat="density", color='green', alpha=0.2, label="Prior (Sampled)")

        # Custom axis limits based on Posterior to avoid "single line" plots
        # Check if this key has a shared xlim from a group
        if key in shared_xlim_cache:
            custom_xlim = shared_xlim_cache[key]
        else:
            # Calculate range from posterior samples ONLY
            try:
                vals = np.asarray(samples[key])
                p_min, p_max = np.percentile(vals, [1, 99])
                p_min, p_max = float(p_min), float(p_max)
                
                p_range = p_max - p_min
                
                if p_range <= 1e-12: # Effectively zero
                    scale = abs(p_min) * 0.1 if abs(p_min) > 1e-12 else 1.0
                    custom_xlim = (p_min - 5*scale, p_max + 5*scale)
                else:
                    # Add padding (e.g., 5% on each side)
                    padding = 0.05 * p_range
                    custom_xlim = (p_min - padding, p_max + padding)
            except Exception as e:
                print(f"Warning: could not calculate percentiles for {key}: {e}")
                custom_xlim = None

        # Prior: Analytical PDF
        if prior_pdf_fn is not None and custom_xlim is not None:
            # Use the zoomed-in grid for evaluating PDF to ensure resolution
            x_grid = np.linspace(custom_xlim[0], custom_xlim[1], 200)
            
            # Get PDF values
            pdf_vals = prior_pdf_fn(key, x_grid)
            
            if pdf_vals is not None:
                axes[i].plot(x_grid, pdf_vals, color='green', linewidth=2, label="Prior (Analytic)")
        
        # Apply custom xlim
        if custom_xlim is not None:
            axes[i].set_xlim(custom_xlim)
            
        axes[i].set_title(key)
        # axes[i].legend() 
        
    # Hide unused subplots
    if num_vars > 1:
        for i in range(num_vars, len(axes)):
            axes[i].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved posterior distribution plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_prediction(samples_load, mean_pred, percentiles, input_xy_exp_plt, data_exp_plt, angle, title, save_path=None, interval_label="90% interval", training_info_label=None, color="blue", mean_color=None):
    """
    Plots prediction (mean + interval) against experimental data.
    """
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    
    # Use color for mean if mean_color not provided
    if mean_color is None:
        mean_color = color

    # Interval
    ax.fill_betweenx(samples_load, percentiles[0, :], percentiles[1, :], color=color, alpha=0.5, label=interval_label)
    
    # Mean
    ax.plot(mean_pred, samples_load, color=mean_color, ls="solid", lw=0.5, label="Mean prediction")
    
    # Data
    sz=1.5
    # Averaged Data Plotting
    for i in range(len(input_xy_exp_plt)):
        # Calculate mean across columns (positions)
        mean_ext = np.mean(data_exp_plt[i], axis=1)
        lbl = "Exp Data (Avg)" if i == 0 else "_nolegend_"
        
        ax.plot(mean_ext, input_xy_exp_plt[i][:,0], 
                "o", color="black", markerfacecolor="white", markeredgewidth=0.5, 
                markersize=sz, linewidth=0, alpha=0.7, label=lbl)

    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=rf"{title} (${angle}^\circ$)")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Limits
    if "Normal" in title:
        ax.set_xlim(-0.05, 0.05)
    else:
        # Shear / Horizontal
        ax.set_xlim(0, 0.15)
    
    if training_info_label:
        # Add text at the bottom center of the figure
        fig.text(0.5, 0.01, training_info_label, ha='center', fontsize=10, style='italic', color='gray')
    
    if save_path:
        # Increase figure width to accommodate legend
        plt.gcf().set_size_inches(7, 5)
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_prediction(samples_load, mean_prior, pct_prior, mean_post, pct_post, input_xy_exp_plt, data_exp_plt, angle, title, save_path=None, interval_label="95% interval", training_info_label=None):
    """
    Plots combined prediction (Prior + Posterior + Data).
    Prior: Green dashed
    Posterior: Blue solid
    """
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    
    # Prior
    ax.fill_betweenx(samples_load, pct_prior[0, :], pct_prior[1, :], alpha=0.3, color="lightgreen", label=f'Prior {interval_label}')
    ax.plot(mean_prior, samples_load, c="green", ls="dashed", lw=1., label='Prior mean')
    
    # Posterior
    ax.fill_betweenx(samples_load, pct_post[0, :], pct_post[1, :], alpha=0.5, color="orange", label=f'Posterior {interval_label}')
    ax.plot(mean_post, samples_load, c="red", ls="solid", lw=0.5, label="Posterior mean")
    
    # Data
    # Data
    sz=1.5
    for i in range(len(input_xy_exp_plt)):
        # Calculate mean across columns (positions)
        mean_ext = np.mean(data_exp_plt[i], axis=1)
        lbl = "Exp Data (Avg)" if i == 0 else "_nolegend_"
        
        ax.plot(mean_ext, input_xy_exp_plt[i][:,0], 
                "o", color="black", markerfacecolor="white", markeredgewidth=0.5, 
                markersize=sz, linewidth=0, alpha=0.7, label=lbl)
        
    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=rf"{title} (${angle}^\circ$)")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Limits
    if "Normal" in title:
        ax.set_xlim(-0.05, 0.05)
    else:
        # Shear / Horizontal
        ax.set_xlim(0, 0.15)

    if training_info_label:
        # Add text at the bottom center of the figure
        fig.text(0.5, 0.01, training_info_label, ha='center', fontsize=10, style='italic', color='gray')
    
    if save_path:
        # Increase figure width to accommodate legend
        plt.gcf().set_size_inches(7, 5)
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved combined prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_grid_prediction(predictions_collection, angles, save_path=None, interval_label="95% interval"):
    """
    Plots a 2x3 grid of predictions (Rows: Shear/Normal, Cols: Angles).
    
    Args:
        predictions_collection: Dict [angle][direction] -> {
            'samples_load': ..., 'mean_post': ..., 'pct_post': ..., 
            'mean_prior': ..., 'pct_prior': ..., 
            'input_xy_exp': ..., 'data_exp': ..., 'training_info': ...
        }
        angles: List of angles [45, 90, 135]
    """
    directions = ["h", "v"] # Top row: Shear (h), Bottom row: Normal (v)
    rows = 2
    cols = len(angles)
    
    # 5 inch width per col, 4 inch height per row
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
    
    # Ensure axes is 2D array
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    elif cols == 1: axes = axes[:, None]
        
    for r, direction in enumerate(directions):
        dir_label = "Shear" if direction == "h" else "Normal"
        
        for c, angle in enumerate(angles):
            ax = axes[r, c]
            
            # Check if we have data for this cell
            if angle in predictions_collection and direction in predictions_collection[angle]:
                data = predictions_collection[angle][direction]
                
                samples_load = data['samples_load']
                
                # --- Prior ---
                if 'mean_prior' in data and data['mean_prior'] is not None:
                    ax.fill_betweenx(samples_load, data['pct_prior'][0], data['pct_prior'][1], 
                                     alpha=0.3, color="lightgreen", label=f'Prior {interval_label}')
                    ax.plot(data['mean_prior'], samples_load, c="green", ls="dashed", lw=1., label='Prior mean')
                
                # --- Posterior ---
                if 'mean_post' in data and data['mean_post'] is not None:
                    ax.fill_betweenx(samples_load, data['pct_post'][0], data['pct_post'][1], 
                                     alpha=0.5, color="orange", label=f'Posterior {interval_label}')
                    ax.plot(data['mean_post'], samples_load, c="red", ls="solid", lw=0.5, label="Posterior mean")
                
                # --- Experimental Data ---
                if 'data_exp' in data:
                    input_xy = data['input_xy_exp']
                    data_exp = data['data_exp']
                    
                    # Averaged Data Plotting
                    sz=1.5
                    for i in range(len(input_xy)):
                        mean_ext = np.mean(data_exp[i], axis=1)
                        lbl = "Exp Data (Avg)" if i == 0 else "_nolegend_"
                        ax.plot(mean_ext, input_xy[i][:,0], 
                                "o", color="black", markerfacecolor="white", markeredgewidth=0.5, 
                                markersize=sz, linewidth=0, alpha=0.7, label=lbl)

                # --- Labels & Info ---
                if r == 0:
                    ax.set_title(f"Angle {angle}°")
                if r == 1:
                    ax.set_xlabel("Extension [mm]")
                if c == 0:
                    ax.set_ylabel(f"{dir_label} - Load [kN]")
                
                # Limits
                if direction == "v":
                    if int(angle) == 90:
                        ax.set_xlim(-0.05, 0.05)
                    else:
                        ax.set_xlim(-0.05, 0.05)
                else:
                    ax.set_xlim(0, 0.15)
                    
                ax.grid(True)
                
                # Legend only on first subplot per row or consolidated?
                if c == 0:
                    ax.legend(fontsize=8, loc='upper left')
                    
                if 'training_info' in data and data['training_info']:
                     ax.text(0.5, 0.02, data['training_info'], transform=ax.transAxes, 
                             ha='center', fontsize=8, style='italic', color='gray')
            else:
                ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_spaghetti_verification(
    test_loads,
    samples,
    percentiles,  # (2, num_points)
    angle,
    direction,
    input_xy_exp=None,
    data_exp=None,
    plot_type="Posterior",
    save_path=None
):
    """
    Plots a spaghetti plot of samples with overlaid percentiles.
    To verify that bands match the distribution of samples.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    num_samples = samples.shape[0]
    
    # Plot Samples (Spaghetti)
    # Axes Swap: X=Extension (samples), Y=Load (test_loads)
    for s in range(min(num_samples, 200)): # Plot max 200 to avoid clutter
        ax.plot(samples[s], test_loads, color='blue', alpha=0.1, lw=1)
        
    # Plot Percentiles
    if percentiles is not None:
        ax.plot(percentiles[0], test_loads, color='green', linestyle='--', linewidth=1, label='2.5% / 97.5%')
        ax.plot(percentiles[1], test_loads, color='green', linestyle='--', linewidth=1)
        
    # Overlay Data
    if input_xy_exp is not None and data_exp is not None:
        val_label = "val"
        for i in range(len(input_xy_exp)):
            # Average sensors (Right, Center, Left -> axis 1)
            # Update: Plot all raw points
            if data_exp[i].ndim > 1 and data_exp[i].shape[1] > 1:
                 for col in range(data_exp[i].shape[1]):
                      ax.plot(data_exp[i][:, col], input_xy_exp[i][:,0], 
                            "o", color="black", markerfacecolor="white", markeredgewidth=0.5, 
                            markersize=1.5, linewidth=0, alpha=0.5, label=f'Exp {i+1}' if col==0 else "_nolegend_")
            else:
                 mean_ext = data_exp[i].flatten()
                 lbl = "Exp Data (Avg)" if i == 0 else "_nolegend_"
                 ax.plot(mean_ext, input_xy_exp[i][:,0], 
                        "o", color="black", markerfacecolor="white", markeredgewidth=0.5, 
                        markersize=1.5, linewidth=0, alpha=0.7, label=lbl)

    dir_label = "Shear" if direction == "h" else "Normal"
    ax.set_title(f"{plot_type} Spaghetti Verification - {angle}° {dir_label}")
    ax.set_xlabel("Extension [mm]")
    ax.set_ylabel("Load [kN]")
    
    # Limits (consistent with grid)
    if direction == "v":
        if int(angle) == 90:
            ax.set_xlim(-0.05, 0.05)
        else:
            ax.set_xlim(-0.05, 0.05)
    else:
        ax.set_xlim(0, 0.15)
        
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved posterior spaghetti plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_grid_spaghetti(prediction_data, angles, save_path=None, title_prefix="Posterior"):
    """
    Plots a 2xN grid of spaghetti plots.
    Rows: Shear (H), Normal (V)
    Cols: Angles provided in 'angles' list
    
    prediction_data: dict[angle][direction] -> {samples_load, post_y_samples, pct_post, ...}
    """
    directions = ["v", "h"]  # Row 1: Normal (v), Row 2: Shear (h)
    num_cols = len(angles)
    
    # Dynamic figsize: ~5 inch width per col, 10 inch height total (2 rows)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), squeeze=False)
    
    for row_idx, direction in enumerate(directions):
        for col_idx, angle in enumerate(angles):
            ax = axes[row_idx, col_idx]
            
            p_data = prediction_data.get(angle, {}).get(direction)
            if not p_data:
                ax.axis('off')
                continue
            
            # Extract data depends on whether we are plotting Prior or Posterior
            # The caller handles logic, but prediction_data usually has keys:
            # 'samples_load', 'post_y_samples' / 'prior_y_samples', 'pct_post' / 'pct_prior'
            
            test_loads = p_data['samples_load']
            input_xy_exp = p_data['input_xy_exp']
            data_exp = p_data['data_exp']
            training_info = p_data.get('training_info')

            # Determine what to plot based on title_prefix or available keys
            # Now we have both function (_f) and observation (_y) percentiles
            if "Prior" in title_prefix:
                samples = p_data.get('prior_f_samples')  # Use function samples for spaghetti
                pct_f = p_data.get('pct_prior_f')  # Function uncertainty
                pct_y = p_data.get('pct_prior_y')  # Observation uncertainty
            else:
                samples = p_data.get('post_f_samples')  # Use function samples for spaghetti
                pct_f = p_data.get('pct_post_f')  # Function uncertainty
                pct_y = p_data.get('pct_post_y')  # Observation uncertainty

            # Plot Samples (Spaghetti) - function samples
            if samples is not None:
                num_samples = samples.shape[0]
                # Plot ~100 lines max for visibility
                for s in range(min(num_samples, 100)):
                    ax.plot(samples[s], test_loads, color='blue', alpha=0.1, lw=0.5)

            # Plot Observation Uncertainty Band (outer, wider) - includes noise
            if pct_y is not None:
                ax.fill_betweenx(test_loads, pct_y[0], pct_y[1], 
                                 color='lightblue', alpha=0.3, label='95% Observation')
                ax.plot(pct_y[0], test_loads, color='blue', linestyle=':', linewidth=0.5)
                ax.plot(pct_y[1], test_loads, color='blue', linestyle=':', linewidth=0.5)

            # Plot Function Uncertainty Band (inner, narrower) - epistemic only
            if pct_f is not None:
                ax.fill_betweenx(test_loads, pct_f[0], pct_f[1],
                                 color='lightgreen', alpha=0.5, label='95% Function')
                ax.plot(pct_f[0], test_loads, color='green', linestyle='--', linewidth=0.8)
                ax.plot(pct_f[1], test_loads, color='green', linestyle='--', linewidth=0.8)
            
            # Overlay Data (Averaged)
            if input_xy_exp is not None and data_exp is not None:
                markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
                for i in range(len(input_xy_exp)):
                    marker = markers[i % len(markers)]
                    # Average sensors (Right, Center, Left -> axis 1)
                    if data_exp[i].ndim > 1 and data_exp[i].shape[1] > 1:
                        # Plot each column (sensor) separately
                        for col in range(data_exp[i].shape[1]):
                             ax.plot(data_exp[i][:, col], input_xy_exp[i][:,0], 
                                    marker, color="black", markerfacecolor="white", markeredgewidth=0.5, 
                                    markersize=2, linewidth=0, alpha=0.6, label=f'Exp {i+1}' if col==0 else "_nolegend_")
                    else:
                        mean_ext = data_exp[i].flatten()
                        ax.plot(mean_ext, input_xy_exp[i][:,0], 
                                marker, color="black", markerfacecolor="white", markeredgewidth=0.5, 
                                markersize=2, linewidth=0, alpha=0.8, label=f'Exp {i+1}')

            # Labels and Limits
            if row_idx == 0:
                ax.set_title(f"Angle {angle}°")
            
            if row_idx == 1:
                ax.set_xlabel("Extension (mm)")
                
            if col_idx == 0:
                dir_label = "Shear" if direction == "h" else "Normal"
                ax.set_ylabel(f"{dir_label} - Load (kN)")
                
            # Limits (consistent with user request)
            if direction == "v":
                if int(angle) == 90:
                    ax.set_xlim(-0.05, 0.05)
                elif int(angle) == 45:
                     ax.set_xlim(-0.05, 0.1) # Updated
                elif int(angle) == 135:
                     ax.set_xlim(-0.1, 0.1) # Default for 135 V? User specified "135 degrees Shear".
                     # Assuming "both 45 degree plots" meant H/V for 45.
                     # For 135, user said "for the 135 degrees Shear". 
                     # I will leave 135 V as -0.1, 0.1 unless clarified, or set to -0.05, 0.1 if user implied "both" earlier logic applies?
                     # Ideally 135 Normal is stiff? 
                     # Let's keep 45 V as -0.05, 0.1.
                else:
                    ax.set_xlim(-0.05, 0.05)
            else:
                # Shear
                if int(angle) == 90:
                    ax.set_xlim(-0.05, 0.15) # Updated
                elif int(angle) == 45:
                    ax.set_xlim(-0.05, 0.1) # Updated
                elif int(angle) == 135:
                    ax.set_xlim(-0.05, 0.1) # Updated
                else:
                    ax.set_xlim(0, 0.15)
                
            ax.grid(True, alpha=0.5)
            
            if training_info:
                 ax.text(0.5, 0.02, training_info, transform=ax.transAxes, 
                         ha='center', fontsize=8, style='italic', color='gray')
            
            if row_idx==0 and col_idx==0:
                 # Ensure unique labels in legend
                 handles, labels = ax.get_legend_handles_labels()
                 by_label = dict(zip(labels, handles))
                 ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=6)

    # Global Title
    fig.suptitle(f'{title_prefix} Prediction Spaghetti Grid', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title_prefix} spaghetti grid plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_distributions_grid_2x3(grouped_data, angles, save_path=None, prior_pdf_fn=None, prior_samples=None, title_prefix="Distributions", use_gamma_logic=True):
    """
    Plots distributions in a 2x3 grid (Rows: Shear/Normal, Cols: Angles).
    
    Args:
        grouped_data: dict[direction][angle] -> list of (label, samples_array, key) tuples
        angles: List of angles [45, 90, 135]
        prior_pdf_fn: Function to get analytical prior PDF (optional)
    prior_samples: Dict of prior samples for keys (optional)
        title_prefix: Title prefix for subplots
        use_gamma_logic: If True, applies specific range unification and abs transformation for Gamma plots.
    """
    directions = ["v", "h"]  # Row 1: Normal (v), Row 2: Shear (h)
    rows = 2
    cols = len(angles)
    
    # Pre-compute shared x-limits for specific angles
    # Helper to get range from grouped_data
    def get_range(direction, angle):
        cell_data = grouped_data.get(direction, {}).get(angle, [])
        if not cell_data:
            return None
        all_samps = np.concatenate([np.asarray(item[1]) for item in cell_data])
        p_min, p_max = np.percentile(all_samps, [1, 99])
        return (float(p_min), float(p_max))
    
    shared_xlim = {}  # (direction, angle) -> xlim
    transform_135_v = False

    if use_gamma_logic:
        # 45°: Union of v and h ranges
        range_v_45 = get_range("v", 45)
        range_h_45 = get_range("h", 45)
        if range_v_45 and range_h_45:
            union_min = min(range_v_45[0], range_h_45[0])
            union_max = max(range_v_45[1], range_h_45[1])
            padding = (union_max - union_min) * 0.05
            xlim_45 = (union_min - padding, union_max + padding)
            shared_xlim[("v", 45)] = xlim_45
            shared_xlim[("h", 45)] = xlim_45
        
        # 135°: For Normal, plot absolute value; for Shear, actual value
        # Use union of absolute value ranges for both
        range_v_135 = get_range("v", 135)
        range_h_135 = get_range("h", 135)
        
        if range_v_135 and range_h_135:
            # Get absolute value ranges
            abs_v_min, abs_v_max = abs(range_v_135[0]), abs(range_v_135[1])
            abs_v_range = (min(abs_v_min, abs_v_max), max(abs_v_min, abs_v_max))
            # For h, we need the actual range since v values are negative
            # Absolute value of h range
            abs_h_min, abs_h_max = abs(range_h_135[0]), abs(range_h_135[1])
            abs_h_range = (min(abs_h_min, abs_h_max), max(abs_h_min, abs_h_max))
            
            # Union of absolute value ranges
            union_min = min(abs_v_range[0], abs_h_range[0], 0)  # Include 0
            union_max = max(abs_v_range[1], abs_h_range[1])
            padding = (union_max - union_min) * 0.05
            xlim_135 = (union_min - padding, union_max + padding)
            
            shared_xlim[("h", 135)] = xlim_135
            shared_xlim[("v", 135)] = xlim_135
            transform_135_v = True  # Mark that we need to transform v data
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows), constrained_layout=True)
    
    # Ensure axes is 2D array
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    elif cols == 1: axes = axes[:, None]
    
    for r, direction in enumerate(directions):
        dir_label = "Shear" if direction == "h" else "Normal"
        
        for c, angle in enumerate(angles):
            ax = axes[r, c]
            
            # Check if we have data for this cell
            cell_data = grouped_data.get(direction, {}).get(angle, [])
            
            if cell_data:
                # Use shared xlim if available, otherwise compute from data
                if (direction, angle) in shared_xlim:
                    xlim = shared_xlim[(direction, angle)]
                else:
                    # Determine x-limits from data
                    all_samps = np.concatenate([np.asarray(item[1]) for item in cell_data])
                    p_min, p_max = np.percentile(all_samps, [1, 99])
                    p_range = p_max - p_min
                    padding = p_range * 0.05
                    xlim = (p_min - padding, p_max + padding)
                
                x_grid = np.linspace(xlim[0], xlim[1], 200)

                # --- Prior Plotting ---
                prior_plotted = False
                
                # 1. Try Analytic PDF first
                if prior_pdf_fn:
                    # Try to find a valid key for prior lookup from the data items
                    prior_key = None
                    for item in cell_data:
                        if len(item) >= 3:
                            prior_key = item[2]
                            break
                    
                    # Fallback: if no key provided, try using the label of the first item
                    if prior_key is None and cell_data:
                         prior_key = cell_data[0][0]

                    # Fetch PDF
                    pdf_vals = prior_pdf_fn(prior_key, x_grid)
                    
                    if pdf_vals is not None:
                         ax.plot(x_grid, pdf_vals, color='green', linewidth=2, label="Prior (Analytic)", alpha=0.8)
                         prior_plotted = True
                
                # 2. If no analytic PDF, try Prior Samples (KDE)
                if not prior_plotted and prior_samples:
                    # Collect prior samples for all keys in this cell
                    cell_prior_samps = []
                    for item in cell_data:
                        # item is (label, samples, key) or (label, samples)
                        key = item[2] if len(item) >= 3 else item[0]
                        if key in prior_samples:
                            # Flatten
                            ps = np.array(prior_samples[key]).flatten()
                            cell_prior_samps.append(ps)
                    
                    if cell_prior_samps:
                        combined_prior = np.concatenate(cell_prior_samps)
                        # Plot KDE as green line
                        sns.kdeplot(combined_prior, ax=ax, color='green', linewidth=2, label="Prior (Sim)", alpha=0.8)
                        prior_plotted = True

                # --- Posterior Plotting ---
                # If multiple items, use different colors/labels
                for item in cell_data:
                    label = item[0]
                    samples = item[1]
                    # Convert to numpy to avoid matplotlib hanging with JAX arrays
                    samples_np = np.asarray(samples)
                    
                    # Special case: 135° Normal - plot absolute values
                    if angle == 135 and direction == "v" and transform_135_v:
                        samples_np = np.abs(samples_np)
                        label = f"|{label}|"  # Indicate absolute value
                    
                    sns.histplot(samples_np, ax=ax, label=label, kde=False, stat="density", alpha=0.4)
                    
                ax.set_title(f"{title_prefix} - {angle}° {dir_label}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                # Set X-Limits Explicitly based on data or shared logic
                ax.set_xlim(xlim)
                
                # Format axes to use scientific notation for small/large values
                ax.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
                
                # Add vertical reference line for theoretical gamma values ONLY if using gamma logic
                if use_gamma_logic:
                    sqrt2_2 = np.sqrt(2) / 2
                    ref_values = {
                        "h": {45: sqrt2_2, 90: 1, 135: sqrt2_2},
                        "v": {45: sqrt2_2, 90: 0, 135: sqrt2_2 if transform_135_v else -sqrt2_2}
                    }
                    if direction in ref_values and angle in ref_values[direction]:
                        ref_val = ref_values[direction][angle]
                        ax.axvline(x=ref_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f"Theory ({ref_val:.3f})")
            else:
                ax.axis('off')
                
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title_prefix} grid plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_bias_column_layout(bias_data_by_angle, save_path=None, prior_pdf_fn=None, prior_samples=None):
    """
    Plots bias parameters in 3 columns (45°, 90°, 135°).
    Rows depend on the number of parameters per angle.
    X-axis is symmetric around zero for easy visual comparison.
    """
    angles = [45, 90, 135]
    
    # Determine max rows
    max_rows = 0
    for ang in angles:
        max_rows = max(max_rows, len(bias_data_by_angle.get(ang, [])))
    
    if max_rows == 0:
        print("No bias data to plot.")
        return

    cols = 3
    rows = max_rows
    
    # Compute global symmetric xlim centered on zero
    # Find the maximum absolute value across ALL samples to use as symmetric bound
    all_abs_max = 0
    for ang in angles:
        for item in bias_data_by_angle.get(ang, []):
            samples_np = np.asarray(item[1]).flatten()
            p_min, p_max = np.percentile(samples_np, [1, 99])
            abs_max = max(abs(p_min), abs(p_max))
            all_abs_max = max(all_abs_max, abs_max)
    
    # Add 10% padding and make symmetric
    xlim_bound = all_abs_max * 1.1
    xlim = (-xlim_bound, xlim_bound)
    
    # Dynamic height
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
    
    # Ensure axes is 2D
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    elif cols == 1: axes = axes[:, None]
    
    for c, angle in enumerate(angles):
        items = bias_data_by_angle.get(angle, [])
        
        for r in range(rows):
            ax = axes[r, c]
            
            if r < len(items):
                label, samples, key = items[r]
                
                # Convert samples to numpy first
                samples_np = np.asarray(samples).flatten()
                
                x_grid = np.linspace(xlim[0], xlim[1], 200)
                
                # --- Prior ---
                prior_plotted = False

                if prior_pdf_fn:
                     pdf_vals = prior_pdf_fn(key, x_grid)
                     if pdf_vals is not None:
                         ax.plot(x_grid, pdf_vals, color='green', linewidth=2, label="Prior", alpha=0.8)
                         prior_plotted = True
                
                if not prior_plotted and prior_samples:
                    if key in prior_samples:
                        ps = np.asarray(prior_samples[key]).flatten()
                        # Use histplot instead of KDE for robustness
                        try:
                           sns.histplot(ps, ax=ax, color='green', stat="density", element="step", fill=False, label="Prior (Sim)", alpha=0.5, binrange=xlim)
                           prior_plotted = True
                        except Exception as e:
                           print(f"Warning: Prior plot failed for {key}: {e}")
                
                # --- Posterior ---
                sns.histplot(samples_np, ax=ax, label=label, kde=False, stat="density", alpha=0.4, binrange=xlim)
                
                # Add vertical line at zero for reference
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                ax.set_title(f"{label}") 
                ax.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
                ax.set_xlim(xlim)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
    
    plt.tight_layout() # Use strict layout instead of constrained

    
    # Add Column Headers
    # We can add sub-titles or just rely on individual titles? 
    # User asked for "Left 45...", maybe a super title?
    # Or just let individual titles speak? "b_45_1" is descriptive.
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Bias column plot to {save_path}")
    else:
        plt.show()
    plt.close()
