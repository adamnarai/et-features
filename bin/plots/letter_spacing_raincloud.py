import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from na_py_tools.utils import p2star
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR
import ptitprince as pt
import seaborn as sns

# Params
study = 'letter_spacing'
meas_name_list = ['Med_rspeed_wnum', 'Med_fixdur', 'Med_fsaccamp', 'Perc_fsacc', 'Freq_fsacc', 'Med_fsnum_wnum']
meas_name_list_v = ['Median reading speed\n (words/sec)', 
                    'Median fixation duration (sec)',
                    'Median progressive saccade\n amplitude (°)',
                    'Percentage of progressive\n saccades (%)',
                    'Frequency of progressive\n saccades (1/s)',
                    'Median number of progrressive\n saccades per word']
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

# Raincloud generation
def raincloud_spacing_group(data, y_label = '', p_values = None, legend_loc=1):
    bp = pt.RainCloud(x='condition', y='value', data=data, palette=[sns.color_palette()[0]], bw='scott',
                  width_viol=.5, orient='v' , alpha=.6, width_box=.12, point_size=4, 
                  offset=.35, move=-.2, cut=1, dodge=True, rain_split=False, pointplot=False, linewidth=0)
    remove_boxplot_outliers(bp)
    locs, labels = plt.xticks()
    plt.xticks(locs-.2, labels)
    bp.set(xlabel='')
    plt.ylabel(ylabel=y_label, labelpad=0)
    
    # Stats
    c = (.3,.3,.3)
    if p_values:
        y_min = min(data['value'])
        y_max = max(data['value'])
        y = y_min-(y_max-y_min)*0.035
        for i in range(0,5):
            x1, x2 = -0.18+i-.2, 0.18+i-.2
            stat_str = p2star(p_values[i])
            if stat_str:
                plt.plot([x1, x2], [y, y], lw=2, c=c)
                if stat_str == '.':
                    plt.text((x1+x2)*.5, y+(y_max-y_min)*0.05, stat_str, ha='center', va='top', color=c, fontsize=20)
                else:
                    plt.text((x1+x2)*.5, y-(y_max-y_min)*0.01, stat_str, ha='center', va='top', color=c)
    return bp

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

# Read data
data = pd.read_pickle(os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
data = data[(data['study'] == study)]
data_long = pd.melt(data, id_vars=['study', 'group', 'condition', 'subj_id'], var_name = 'measure', value_name = 'value')

fig = plt.figure(figsize=(10,5))
for idx, meas_name in enumerate(meas_name_list):
    meas_name = meas_name_list[idx]

    fig.add_subplot(2,3,idx+1)
    bp = raincloud_spacing_group(data_long[data_long['measure'] == meas_name], meas_name_list_v[idx])
    
    # Hide the right and top spines
    bp.spines['right'].set_visible(False)
    bp.spines['top'].set_visible(False)
    
    bp.set_xlim(-1, 2.5)
    
    # Decrease box width
    adjust_box_widths(bp, 0.8)
    
    if idx < 3:
        bp.get_xaxis().set_ticks([])
    
    # # Set common y label position
    # bp.yaxis.set_label_coords(-0.2, 0.5)
    
    # # Increase y limit for significance marks
    # ylim = bp.get_ylim()
    # bp.set_ylim(ylim[0]-(ylim[1]-ylim[0])*0.01, ylim[1])
    # bp.set_xlim(-.7, 4.3)

fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.05, right=0.95, wspace=0.35)