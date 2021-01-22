import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadr
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import PathPatch

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

# Load params YAML
with open('params.yaml') as file:
    p = yaml.full_load(file)
    
# Create reports dir
os.makedirs(p['reports_dir'], exist_ok=True)

def p2star(p):
    if p < 0.00001:
        return '*****'
    elif p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''
    
# Boxplot generation
def boxplot_spacing_group(data, y_label = '', p_values = None, legend_loc=1):
    bp = sns.boxplot(x='spacing', y='value', hue='group', data = data, 
                     showmeans=False, whis = np.inf, width=0.7, palette=[(1,1,1), (.45,.45,.45)])
    bp.set(xlabel='Spacing')
    plt.ylabel(ylabel=y_label, labelpad=10)
    
    # Stats
    c = (.3,.3,.3)
    if p_values:
        y_min = min(data['value'])
        y_max = max(data['value'])
        y = y_min-(y_max-y_min)*0.035
        for i in range(0,5):
            x1, x2 = -0.18+i, 0.18+i
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


# Load ET data
data = pd.read_excel(p['data_path'], sheet_name=p['cond_names'], index_col='Measures/Subjects')
data = dict((k, v.transpose()) for k, v in data.items())
data = pd.concat(data)
data.reset_index(level=0, inplace=True)

# Filter data
data = data.loc[p['subjects']['control']+p['subjects']['dyslexic'], :]
data = data.dropna()

# Rename groups
data.reset_index(level=0, inplace=True)
data = data.rename(columns={'level_0':'spacing', 'index':'subid', 'Group':'group'})
data['group'] = data['group'].astype('int32').astype('category').replace({1: 'dyslexic', 2: 'control'})
data_long = pd.melt(data, id_vars=['group','spacing','subid'], var_name = 'measure', value_name = 'value')

# %% Generate boxplots
meas_name_list = ['Med_rspeed_wnum', 'Med_saccamp', 'Med_fixdur']
meas_name_list_v = ['Median reading\n speed (words/sec)', 
                    'Median saccade\n amplitude (vis. deg.)', 
                    'Median fixation\n duration (sec)']

fig = plt.figure(figsize=(8, 11))
for idx in range(3):
    meas_name = meas_name_list[idx]
    # Load R stats
    anova_stats = pyreadr.read_r(p['stats_dir'] + '/ET_mixed_ANOVA_' + meas_name + '.RData')
    
    # Boxplot
    fig.add_subplot(3,1,idx+1)
    bp = boxplot_spacing_group(data_long[data_long['measure'] == meas_name], meas_name_list_v[idx], 
                          list(anova_stats['ph_summary']['p.value'][:5]))
    if idx < 2:
        bp.set(xlabel='')
        bp.get_xaxis().set_ticks([])
    if idx == 0:
        handles, labels = bp.get_legend_handles_labels()
        plt.legend(handles[0:2], ['Control', 'Dyslexic'], loc='center', 
                   bbox_to_anchor=(0.86, 0.94), borderaxespad=1.5, frameon=False)
    else:
        bp.get_legend().remove()
        
    # Hide the right and top spines
    bp.spines['right'].set_visible(False)
    bp.spines['top'].set_visible(False)
    
    # Decrease box width
    adjust_box_widths(bp, 0.8)
    
    # Set common y label position
    bp.yaxis.set_label_coords(-0.075, 0.5)
    
    # Increase y limit for significance marks
    ylim = bp.get_ylim()
    bp.set_ylim(ylim[0]-(ylim[1]-ylim[0])*0.01, ylim[1])

fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.05, right=0.95)

# Save fig
fig.savefig(p['reports_dir'] + '/ET_boxplots_manuscript.png')