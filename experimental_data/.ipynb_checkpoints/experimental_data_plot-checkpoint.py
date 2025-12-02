import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
mpl.rcParams["font.size"] = 14

angles = [45, 90, 135]
directions = ['h', 'v']
legend_labels = ['Horizontal data', 'Vertical data']
legend_styles = ['o', '^']

data_path = Path(__file__).parent
for angle in angles:
    fig, ax = plt.subplots(figsize=(5,5), layout='constrained')
    for ind_d, direction in enumerate(directions):
        load_path = sorted(data_path.glob("input_load_angle_exp_" + str(angle) + '_' + direction + '*'))
        extension_path = sorted(data_path.glob("data_extension_exp_" + str(angle) + '_' + direction + '*'))

        for ind_l, (load_file, extension_file) in enumerate(zip(load_path, extension_path)):
            load = np.loadtxt(load_file, delimiter=",")[:,0]
            extension = np.loadtxt(extension_file, delimiter=",").mean(axis=1)
            ind_notnan = ~np.isnan(extension)
            ax.plot(extension[ind_notnan], load[ind_notnan], linestyle='', marker=legend_styles[ind_d], markersize=8, alpha=0.6, label=legend_labels[ind_d]+str(ind_l+1))
    
    ax.axhline(y=10, linestyle='--', color='gray')
    if angle == 135:
        ax.set_xlim((-0.1, 0.16))
    if angle == 45:
        ax.set_yticks(np.arange(0,18,2))
    ax.set_xlabel('Extension [mm]')
    ax.set_ylabel('Load [kN]')
    ax.set_title(r'$\alpha =' + str(angle) + '^\circ$')
    ax.legend(fontsize=12)
    fig.savefig(data_path.joinpath("experimental_data_"+str(angle)+".svg"))
    plt.show()

