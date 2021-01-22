"""


@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from na_py_tools.defaults import RESULTS_DIR
import ptitprince as pt
from sklearn.preprocessing import StandardScaler

# Params
study = 'letter_spacing'
group = 'control'
cond = 'NS'
meas_list = ['meas_list_min_zscore_y', 'meas_list_min_3_zscore_y', 
             'meas_list_min_4_zscore_y', 'meas_list_min_2_zscore_y']
meas_list_v = ['Set 1', 'Set 2', 'Set 3', 'Set 4']

# study = 'dys'
# group = 'dyslexic'
# cond_list = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
# meas_list = ['meas_list_min_zscore_y', 'meas_list_min_3_zscore_y']

meas_name_list = ['Med_rspeed_wnum', 'Med_fixdur', 'Med_fsaccamp', 'Perc_fsacc', 'Freq_fsacc', 'Med_fsnum_wnum']
meas_name_list_v = ['Median reading speed\n (words/sec)', 
                    'Median fixation duration (sec)',
                    'Median progressive saccade\n amplitude (°)',
                    'Percentage of progressive\n saccades (%)',
                    'Frequency of progressive\n saccades (1/s)',
                    'Median number of progrressive\n saccades per word']

# Raincloud generation
def raincloud_spacing_group(data, y_label = '', p_values = None, legend_loc=1):
    ax = pt.RainCloud(x='measure', y='value', data=data, palette=[sns.color_palette()[0]], bw='scott',
                  width_viol=.7, orient='h' , alpha=.6, width_box=.12, point_size=4, 
                  offset=.35, move=-.2, cut=1, dodge=True, rain_split=False, pointplot=False, linewidth=0)
    remove_boxplot_outliers(ax)
    locs, labels = plt.yticks()
    plt.yticks(locs-.55, labels, rotation=0, ha='right')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    for label in ax.yaxis.get_majorticklabels():
        label.set_x(0.2)
    ax.set(xlabel='Standard deviation (SD)', ylabel='')
    return ax

def adjust_box_widths(ax, fac):
    # Iterating through axes childrens
    for c in ax.get_children():
        # Searching for PathPatches
        if isinstance(c, PathPatch):
            # Getting current width of box
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # Setting new width for box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # Setting new width of median line
            for line in ax.lines:
                if np.all(line.get_xdata() == [xmin, xmax]):
                    line.set_xdata([xmin_new, xmax_new])

def remove_boxplot_outliers(ax):
    # Iterating through axes childrens
    for c in ax.get_children():
        # Searching for PathPatches
        if isinstance(c, Line2D) and (c.get_marker() != 'None'):
            c.set_marker('None')
         
# Read ET data
data = pd.read_pickle(os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
data = data.loc[(data['study'] == study) & (data['condition'] == cond) & (data['group'] == group), 
                ['subj_id'] + meas_name_list]
scaler = StandardScaler()
data[meas_name_list] = scaler.fit_transform(data[meas_name_list])
data_long = pd.melt(data, id_vars=['subj_id'], var_name = 'measure', value_name = 'value')

maps = dict()
maps['beta'] = pd.DataFrame()
maps['pval'] = pd.DataFrame()
for i, meas_list in enumerate(meas_list):
    path = '../../results/regression/'+ study + '/' + group + '/' + cond + '/' + meas_list + \
        '/ET_regr_fold10_perm1000_enet_L1N20_alphN20_ridge_alphN20_cvinperm1.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    temp_b = pd.Series(data['enet']['beta'][0], name=cond+'_'+meas_list)
    temp_p = pd.Series(data['enet']['pval'], index=temp_b.index)
    temp_p.name = meas_list
    if i == 0:
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
    gs = GridSpec(100, 100, figure=fig)
    
    # Raincloud plots
    ax = fig.add_subplot(gs[:, 5:45])

    ax = raincloud_spacing_group(data_long)    
    plt.text(-0.2,1.01,'(a)', fontsize=10, fontweight='bold', horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-3.5, 3.5)
    
    
    
    # ENET results
    fig.add_subplot(gs[:, 70:100])
    ax = sns.heatmap(maps[stat_var], cmap = cmap, vmin = vmin, vmax = vmax, 
                     annot = True, fmt=fmt, cbar=False, annot_kws={"fontsize":fontsize})
    plt.xticks([i+.5 for i in range(len(meas_list_v))], meas_list_v, rotation=0)
    plt.title('')
    plt.xlim(plt.xlim()[0]-np.abs(np.diff(plt.xlim()))*0.01, plt.xlim()[1]+np.abs(np.diff(plt.xlim()))*0.01)
    plt.ylim(plt.ylim()[0]+np.abs(np.diff(plt.ylim()))*0.01, plt.ylim()[1]-np.abs(np.diff(plt.ylim()))*0.01)
    
    # Loop over data dimensions and create text annotations.
    for i in range(maps['beta'].shape[1]):
        for j in range(maps['beta'].shape[0]):
            if maps['pval'].iloc[j,i] < 0.05:
                ax.add_patch(Rectangle((i,j), 1, 1, linewidth=1.5, edgecolor='k', fill=False, zorder=2))    
    
    plt.text(-0.55,1.01,'(b)', fontsize=10, fontweight='bold', horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.05)
        
    if stat_var == 'beta':
        plt.savefig('enet_ns.svg', format='svg')
        plt.savefig('enet_ns_1200dpi.png', dpi=1200)
        # plt.savefig('enet_ns_600dpi.png', dpi=600)
        # plt.savefig('enet_ns_220dpi.png', dpi=220)
                


