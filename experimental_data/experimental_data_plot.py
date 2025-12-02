import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
mpl.rcParams["font.size"] = 12

prop_cycle = mpl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors_dict = {'v':colors[0:3], 'h':colors[3:6]}

angles = [45, 90, 135]
directions = ['v', 'h']
# legend_labels = ['Vertical data', 'Horizontal data']
legend_labels = ['Normal data','Shear data']
legend_styles = ['o', '^']

data_path = Path(__file__).parent
fig, ax = plt.subplots(ncols=3, figsize=(3*4,4), layout='tight', sharey='row')
for ind_a, angle in enumerate(angles):
    for ind_d, direction in enumerate(directions):
        load_path = sorted(data_path.glob("input_load_angle_exp_" + str(angle) + '_' + direction + '*'))
        extension_path = sorted(data_path.glob("data_extension_exp_" + str(angle) + '_' + direction + '*'))

        for ind_l, (load_file, extension_file) in enumerate(zip(load_path, extension_path)):
            load = np.loadtxt(load_file, delimiter=",")[:,0]
            extension = np.loadtxt(extension_file, delimiter=",").mean(axis=1)
            ind_notnan = ~np.isnan(extension)
            ax[ind_a].plot(extension[ind_notnan], load[ind_notnan], linestyle='', marker=legend_styles[ind_d], 
                           color=colors_dict[direction][ind_l], markersize=6, alpha=0.5, label=legend_labels[ind_d]+str(ind_l+1))
    
    ax[ind_a].axhline(y=10, linestyle='--', color='gray')
    ax[ind_a].set_ylim(bottom=-0.5, top=18)
    ax[ind_a].set_yticks(np.arange(0,18,2))
    if angle == 135:
        ax[ind_a].set_xlim((-0.1, 0.16))
        
    ax[ind_a].set_xlabel('Extension [mm]')
    if ind_a == 0:
        ax[ind_a].set_ylabel('Load [kN]')
        ax[ind_a].legend(fontsize=11)

    ax[ind_a].set_title(r'$\alpha =' + str(angle) + '^\circ$')

fig.savefig(data_path.joinpath("experimental_data.webp"), dpi=200, transparent=True)
plt.show()

