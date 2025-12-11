import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
mpl.rcParams["font.size"] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams["axes.formatter.limits"] = (-3,4)

data_path = Path(__file__).parent
sa_index_h = pd.read_csv(data_path.joinpath('sa_h.txt'), index_col=0, delimiter=',')
sa_index_v = pd.read_csv(data_path.joinpath('sa_v.txt'), index_col=0, delimiter=',')

labels = ['$E_1$','$E_2$',r'$\nu_{12}$', r'$\nu_{23}$', '$G_{12}$']
sa_index_h.index = labels
sa_index_v.index = labels
# sa_index_h.sort_values(by='mi', ascending=False, inplace=True)
# sa_index_v.sort_values(by='mi', ascending=False, inplace=True)

fig_h, ax_h = plt.subplots(figsize=(4,4), layout="constrained")
for i in sa_index_h.index:
    ax_h.plot(sa_index_h.at[i,'mi'], sa_index_h.at[i,'sigma'], 'o', markersize=15)

ax_h.grid(True, linewidth=0.5) 
ax_h.set(xlabel="mean of EE", ylabel="standard deviation of EE")
ax_h.legend(sa_index_h.index)
fig_h.savefig(data_path.joinpath("sa_index_h.svg"))
plt.show()

fig_v, ax_v = plt.subplots(figsize=(4,4), layout="constrained")
for i in sa_index_v.index:
    ax_v.plot(sa_index_v.at[i,'mi'], sa_index_v.at[i,'sigma'], 'o', markersize=15)

ax_v.grid(True, linewidth=0.5) 
ax_v.set(xlabel="mean of EE", ylabel="standard deviation of EE")
ax_v.legend(sa_index_v.index)
fig_v.savefig(data_path.joinpath("sa_index_v.svg"))
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(2*4,4), layout="tight")
for i in sa_index_v.index:
    ax[0].plot(sa_index_v.at[i,'mi'], sa_index_v.at[i,'sigma'], 'o', markersize=15)
ax[0].grid(True, linewidth=0.5) 
ax[0].set(xlabel="mean of EE", ylabel="standard deviation of EE")
ax[0].legend(sa_index_v.index)
ax[0].set_title("Sensitivity analysis on normal extension", fontsize=11)

for i in sa_index_h.index:
    ax[1].plot(sa_index_h.at[i,'mi'], sa_index_h.at[i,'sigma'], 'o', markersize=15)
ax[1].grid(True, linewidth=0.5) 
ax[1].set(xlabel="mean of EE")
ax[1].set_title("Sensitivity analysis on shear extension", fontsize=11)

fig.savefig(data_path.joinpath("sa_index.webp"), dpi=200, transparent=True)
plt.show()

