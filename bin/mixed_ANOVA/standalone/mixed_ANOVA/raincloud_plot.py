"""
Raincloud plot for mixed ANOVA

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import yaml
import ptitprince as pt
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
import pingouin as pg

# Load params YAML
with open('params.yaml') as file:
    p = yaml.full_load(file)
    
# Create reports dir
os.makedirs(p['reports_dir'], exist_ok=True)

# Checking for outliers based on IQR
def iqr_outlier(x, coef=1.5):
  Q1 = x.quantile(0.25)
  Q3 = x.quantile(0.75)
  IQR = Q3 - Q1
  return (x < (Q1 - coef*IQR)) | (x > (Q3 + coef*IQR))

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
def boxplot_stats(data, p_values, ax): 
    # Stats
    c = (.3,.3,.3)
    if p_values:
        y_min = min(data['value'])
        y_max = max(data['value'])
        y = y_min-(y_max-y_min)*0.035
        for i in range(len(p_values)):
            x1, x2 = -0.18+i-.2, 0.18+i-.2
            stat_str = p2star(p_values[i])
            if stat_str:
                ax.plot([x1, x2], [y, y], lw=2, c=c)
                if stat_str == '.':
                    ax.text((x1+x2)*.5, y+(y_max-y_min)*0.03, stat_str, ha='center', va='top', color=c, fontsize=20)
                else:
                    ax.text((x1+x2)*.5, y-(y_max-y_min)*0.01, stat_str, ha='center', va='top', color=c)

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
            
# Load data
data = pd.read_csv(p['data_path'])
data_long = pd.melt(data, id_vars=['group', 'condition', 'subj_id'], var_name = 'measure', value_name = 'value')

# 1.5xIQR outliers
outl_subj = data_long[data_long.groupby(['group','condition','measure'])['value'].transform(iqr_outlier)]

# Remove outliers
data_long = data_long[~data_long.groupby(['group','condition','measure'])['value'].transform(iqr_outlier)]

# Stats
for gp in ('control', 'dyslexic'):
    for sp in ('SP1', 'SP2', 'SP3', 'SP4', 'SP5'):
        pval = pg.ttest(data_long[(data_long['group'] == gp) & (data_long['condition'] == sp)].value, 0)['p-val'].values
        print('{}-{}: P = {:.4f}'.format(gp, sp, min([1, float(pval*10)])))
# Plot
for meas_name in p['meas_list']:
    fig = plt.figure(figsize=(10, 6))
    verbose_meas_name = meas_name
    # Load R stats
    anova_stats = pyreadr.read_r(p['stats_dir'] + '/mixed_ANOVA_' + meas_name + '.RData')
    
    # Get data
    curr_data = data_long[data_long['measure'] == meas_name]
    curr_data['group'] = curr_data['group'].str.capitalize()
    
    # Rainclud plot
    ax = pt.RainCloud(x='condition', y='value', hue='group', data=curr_data, palette='Set1', bw='scott',
                  width_viol=.5, orient='v' , alpha=.6, width_box=.12, point_size=4, 
                  offset=.35, move=-.2, cut=0, dodge=True, rain_split=False, linewidth=0)
    
    # Plot settings
    ax.set_xlim(-.7, 4.3)
    plt.ylabel(ylabel=verbose_meas_name, axes=ax)
    boxplot_stats(curr_data, list(anova_stats['ph_summary']['p.value'][:5]), ax)
    ax.set(xlabel='Spacing')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:2], labels, loc='center', prop={'size': 10},
               bbox_to_anchor=(0.90, 0.12), borderaxespad=1.5, frameon=False)

        
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Decrease box width
    adjust_box_widths(ax, 0.6)
    
    remove_boxplot_outliers(ax)

    
    # Increase y limit for significance marks
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0]-(ylim[1]-ylim[0])*0.01, ylim[1])


    # Save fig
    fig.savefig(p['reports_dir'] + '/' + meas_name + '.png')
    