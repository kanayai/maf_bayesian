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

def plot_posterior_distributions(samples, prior_pdf_fn=None, prior_samples=None, save_path=None, layout_rows=None):
    """
    Plots histograms of posterior samples for all parameters.
    If prior_pdf_fn is provided, plots analytical prior density as a green line.
    prior_pdf_fn(key, x_vals) -> pdf_vals
    If prior_samples is provided (dict), plots prior histogram for matching keys.
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
    
    for i, key in enumerate(keys):
        # Posterior: Histogram only (stat="density" to match KDE scale)
        sns.histplot(samples[key], ax=axes[i], kde=False, stat="density", label="Posterior", alpha=0.4)
        
        # Prior: Samples (Histogram)
        if prior_samples is not None and key in prior_samples:
            # Flatten if necessary (e.g. if chains dim exists)
            p_s = np.array(prior_samples[key]).flatten()
            sns.histplot(p_s, ax=axes[i], kde=False, stat="density", color='green', alpha=0.2, label="Prior (Sampled)")

        # Custom axis limits based on Posterior to avoid "single line" plots
        # We calculate range from posterior samples ONLY.
        p_min = np.min(samples[key])
        p_max = np.max(samples[key])
        p_range = p_max - p_min
        if p_range == 0:
            scale = abs(p_min) * 0.1 if p_min != 0 else 1.0
            custom_xlim = (p_min - 5*scale, p_max + 5*scale)
        else:
            # Add padding (e.g., 30% on each side)
            padding = 0.3 * p_range
            custom_xlim = (p_min - padding, p_max + padding)

        # Prior: Analytical PDF
        if prior_pdf_fn is not None:
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

    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=f"{title} (${angle}^\circ$)")
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
        
    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=f"{title} (${angle}^\circ$)")
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
    directions = ["h", "v"]
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
