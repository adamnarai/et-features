"""


@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Params
study = 'letter_spacing'
group = 'control_old'
cond_list = ['MS', 'NS', 'DS']
meas_list = ['meas_list_min', 'meas_list_min_3']

# study = 'dys'
# group = 'dyslexic'
# cond_list = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
# meas_list = ['meas_list_min_zscore_y', 'meas_list_min_3_zscore_y']

maps = dict()
maps['beta'] = pd.DataFrame()
maps['pval'] = pd.DataFrame()
for meas_list in meas_list:
    for cond in cond_list:
        path = '../../results/regression/'+ study + '/' + group + '/' + cond + '/' + meas_list + \
            '/ET_regr_fold10_perm1000_enet_L1N20_alphN20_ridge_alphN20_cvinperm1.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        temp_b = pd.Series(data['enet']['beta'][0], name=cond+'_'+meas_list)
        temp_p = pd.Series(data['enet']['pval'], index=temp_b.index)
        temp_p.name = cond+'_'+meas_list
        if (cond == 'MS') and (meas_list == 'meas_list_min'):
            maps['beta'] = temp_b
            maps['pval'] = temp_p
        else:
            maps['beta'] = pd.concat([maps['beta'], temp_b], axis=1, join='outer')
            maps['pval'] = pd.concat([maps['pval'], temp_p], axis=1, join='outer')
        
        
# Coef heatmap
for stat_var in ['beta', 'pval']:
    if stat_var == 'beta':
        cmap = 'RdBu_r'
        vmin = -1
        vmax = 1
        fmt = '.2f'
        fontsize = 10
    elif stat_var == 'pval':
        cmap = 'hot'
        vmin = 0
        vmax = 0.2
        fmt = '.3f'
        fontsize = 9
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(1, 100, figure=fig)
    
    fig.add_subplot(gs[0, 20:50])
    cond_num = len(cond_list)
    ax = sns.heatmap(maps[stat_var].iloc[:,:cond_num], cmap = cmap, vmin = vmin, vmax = vmax, 
                     annot = True, fmt=fmt, cbar=False, annot_kws={"fontsize":fontsize})
    plt.xticks([i+.5 for i in range(cond_num)], cond_list, rotation=0)
    plt.title('Set 1')
    
    # Loop over data dimensions and create text annotations.
    for i in range(maps['beta'].iloc[:,:cond_num].shape[1]):
        for j in range(maps['beta'].iloc[:,:cond_num].shape[0]):
            if maps['pval'].iloc[:,:cond_num].iloc[j,i] < 0.05:
                ax.add_patch(Rectangle((i,j), 1, 1, linewidth=1, edgecolor='k', fill=False, zorder=2))
                
    # Coef heatmap
    ax = fig.add_subplot(gs[0, 52:82])
    cbar_ax = fig.add_subplot(gs[0, 84:87])
    ax = sns.heatmap(maps[stat_var].iloc[:,cond_num:], cmap = cmap, vmin = vmin, vmax = vmax, 
                     annot = True, fmt=fmt, cbar=True, yticklabels=False, ax=ax, 
                     cbar_ax=cbar_ax, annot_kws={"fontsize":fontsize})
    plt.gca().tick_params(labelsize=9) 
    plt.sca(ax)
    plt.xticks([i+.5 for i in range(cond_num)], cond_list, rotation=0)
    plt.title('Set 2')
    
    # Loop over data dimensions and create text annotations.
    for i in range(maps['beta'].iloc[:,cond_num:].shape[1]):
        for j in range(maps['beta'].iloc[:,cond_num:].shape[0]):
            if maps['pval'].iloc[:,cond_num:].iloc[j,i] < 0.05:
                ax.add_patch(Rectangle((i,j), 1, 1, linewidth=1, edgecolor='k', fill=False, zorder=2))


