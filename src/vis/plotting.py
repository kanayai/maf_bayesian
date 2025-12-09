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

def plot_posterior_distributions(samples, prior_pdf_fn=None, save_path=None, layout_rows=None):
    """
    Plots histograms of posterior samples for all parameters.
    If prior_pdf_fn is provided, plots analytical prior density as a green line.
    prior_pdf_fn(key, x_vals) -> pdf_vals
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
    if num_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, key in enumerate(keys):
        # Posterior: Histogram only (stat="density" to match KDE scale)
        # Use stat="density" to normalize area to 1 (PDF scale)
        sns.histplot(samples[key], ax=axes[i], kde=False, stat="density", label="Posterior", alpha=0.4)
        
        # Prior: Analytical PDF
        if prior_pdf_fn is not None:
            # Create grid based on posterior range (or wider if needed)
            # We want to show the prior context, so maybe a bit wider than posterior
            data_min = np.min(samples[key])
            data_max = np.max(samples[key])
            data_range = data_max - data_min
            # If range is 0 (constant), add some buffer
            if data_range == 0: data_range = 1.0
            
            x_grid = np.linspace(data_min - 0.5*data_range, data_max + 0.5*data_range, 200)
            
            # Get PDF values
            pdf_vals = prior_pdf_fn(key, x_grid)
            
            if pdf_vals is not None:
                axes[i].plot(x_grid, pdf_vals, color='green', linewidth=2, label="Prior")
            
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

def plot_prediction(samples_load, mean_pred, percentiles, input_xy_exp_plt, data_exp_plt, angle, title, save_path=None, interval_label="90% interval"):
    """
    Plots prediction (mean + interval) against experimental data.
    """
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    
    # Interval
    ax.fill_betweenx(samples_load, percentiles[0, :], percentiles[1, :], color="lightblue", label=interval_label)
    
    # Mean
    ax.plot(mean_pred, samples_load, "blue", ls="solid", lw=0.5, label="Mean prediction")
    
    # Data
    sz=4
    positions = ["Left", "Center", "Right"]
    markers = ['o', 's', '^']
    colors = ['k', 'k', 'k'] # Or use different colors? Original experimental plot used angle colors and pos markers.
    # But here we are in a single angle plot.
    # Let's use black or specific colors. The prediction is blue.
    # Let's use simple colors or just markers.
    
    for i in range(len(input_xy_exp_plt)):
        # Assuming data_exp_plt[i] has 3 columns for L, C, R
        for col in range(data_exp_plt[i].shape[1]):
            # Only label the first experiment's columns to avoid duplicate legend entries
            lbl = positions[col] if i == 0 else "_nolegend_"
            # If there are multiple experiments, maybe distinguish them? 
            # But usually we have 1 exp per angle.
            
            ax.plot(data_exp_plt[i][:, col], input_xy_exp_plt[i][:,0], 
                    f"-{markers[col]}", markersize=sz, linewidth=1, alpha=0.7, label=lbl)

    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=f"{title} (${angle}^\circ$)")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Limits
    if "Normal" in title:
        ax.set_xlim(-0.05, 0.05)
    else:
        # Shear / Horizontal
        ax.set_xlim(0, 0.15)
    
    if save_path:
        # Increase figure width to accommodate legend
        plt.gcf().set_size_inches(7, 5)
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_prediction(samples_load, mean_prior, pct_prior, mean_post, pct_post, input_xy_exp_plt, data_exp_plt, angle, title, save_path=None, interval_label="95% interval"):
    """
    Plots combined prediction (Prior + Posterior + Data).
    Prior: Green dashed
    Posterior: Blue solid
    """
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    
    # Prior
    ax.fill_betweenx(samples_load, pct_prior[0, :], pct_prior[1, :], alpha=0.75, color="lightgreen", label=f'Prior {interval_label}')
    ax.plot(mean_prior, samples_load, c="green", ls="dashed", lw=1., label='Prior mean')
    
    # Posterior
    ax.fill_betweenx(samples_load, pct_post[0, :], pct_post[1, :], alpha=1, color="lightblue", label=f'Posterior {interval_label}')
    ax.plot(mean_post, samples_load, c="blue", ls="solid", lw=0.5, label="Posterior mean")
    
    # Data
    sz=4
    positions = ["Left", "Center", "Right"]
    markers = ['o', 's', '^']
    
    for i in range(len(input_xy_exp_plt)):
        for col in range(data_exp_plt[i].shape[1]):
            lbl = positions[col] if i == 0 else "_nolegend_"
            ax.plot(data_exp_plt[i][:, col], input_xy_exp_plt[i][:,0], 
                    f"-{markers[col]}", markersize=sz, linewidth=1, alpha=0.7, label=lbl)
        
    ax.set(xlabel="Extension [mm]", ylabel="Load [kN]", title=f"{title} (${angle}^\circ$)")
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Limits
    if "Normal" in title:
        ax.set_xlim(-0.05, 0.05)
    else:
        # Shear / Horizontal
        ax.set_xlim(0, 0.15)
    
    if save_path:
        # Increase figure width to accommodate legend
        plt.gcf().set_size_inches(7, 5)
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved combined prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()
