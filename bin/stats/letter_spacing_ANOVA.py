import os
import pandas as pd
import pingouin as pg
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadr
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR
from na_py_tools.utils import p2star

# Params
save_pdf = True
study = 'letter_spacing'
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
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
data = data[(data['study'] == study)]

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'stats', 'et_features')
os.makedirs(results_dir, exist_ok=True)

data_long = pd.melt(data, id_vars=['group','condition','subj_id'], var_name = 'measure', value_name = 'value')

# %% Generate pdf from figures
for meas_type in ['_min', '', '_all']:
    meas_list = p['params']['et']['meas_list' + meas_type]
    if save_pdf:
        pdf = PdfPages(results_dir + '/' + study + '_ET_stats' + meas_type + '.pdf')
    for meas_name in meas_list:
        # %%Boxplot
        fig = plt.figure(figsize=(16, 13))
        fig.add_axes([0.05, 0.5, 0.26, 0.4])
        ax = sns.boxplot(x='condition', y=meas_name, data=data, showmeans=False, color='b', width=.6)
        sns.swarmplot(x='condition', y=meas_name, data=data, color=(.3,.3,.3), ax=ax)
        ax.set(xlabel='Spacing', ylabel='');
        plt.title(meas_name)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # ANOVA table
        fig.add_axes([0.33, 0.82, 0.6, 0.08]).axis('off')
        stats = pg.rm_anova(data=data, dv=meas_name, within='condition', 
                            subject='subj_id', correction='auto', detailed=True)
        num_col = stats.columns.difference(['Source', 'sphericity'])
        stats.loc[0,num_col] = stats.loc[0,num_col].apply('{:.5g}'.format)
        the_table = plt.table(cellText=stats.values,
                              colWidths=[x*.3 for x in [.2, .2, .1, .2, .2, .3, .3, .2, .2, .2, .2, .3]],
                              colLabels=stats.columns,
                              cellLoc = 'center', rowLoc = 'center',
                              loc = 'center left')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8.5)
        the_table.scale(1, 1.5)
        plt.text(0, 1, 'Repeated measures ANOVA')
        
        # Post-hoc table parametric
        fig.add_axes([0.33, 0.7, 0.6, 0.1]).axis('off')
        stats = pg.pairwise_ttests(data=data, dv=meas_name, within='condition', 
                                   subject='subj_id', return_desc=True, padjust='bonf',
                                   effsize='cohen', parametric=True)
        num_col = ['p-unc', 'p-corr']
        stats[num_col] = stats[num_col].applymap('{:.5g}'.format)
        the_table = plt.table(cellText=stats.values,
                              colWidths=[x*.3 for x in [.2, .1, .1, .2, .2, .2, .2, .2, .25, .25, .2, .25, .3, .3, .2, .2, .2]],
                              colLabels=stats.columns,
                              cellLoc = 'center', rowLoc = 'center',
                              loc = 'center left')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8.5)
        the_table.scale(1, 1.5)
        plt.text(0, 1, 'Post-hoc (paired t-tests)')
        
        # Friedman table
        fig.add_axes([0.33, 0.6, 0.6, 0.06]).axis('off')
        stats = pg.friedman(data=data, dv=meas_name, within='condition', 
                            subject='subj_id')
        num_col = 'p-unc'
        stats[num_col] = stats[num_col].apply('{:.5g}'.format)
        the_table = plt.table(cellText=stats.values,
                              colWidths=[x*.3 for x in [.2, .2, .3, .3]],
                              colLabels=stats.columns,
                              cellLoc = 'center', rowLoc = 'center',
                              loc = 'center left')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8.5)
        the_table.scale(1, 1.5)
        plt.text(0, 1, 'Friedman test')
        
        # Post-hoc table non-parametric
        fig.add_axes([0.33, 0.48, 0.6, 0.1]).axis('off')
        stats = pg.pairwise_ttests(data=data, dv=meas_name, within='condition', 
                                   subject='subj_id', return_desc=True, padjust='bonf',
                                   effsize='cohen', parametric=False)
        num_col = ['p-unc', 'p-corr']
        stats[num_col] = stats[num_col].applymap('{:.5g}'.format)
        the_table = plt.table(cellText=stats.values,
                              colWidths=[x*.3 for x in [.2, .1, .1, .2, .2, .2, .2, .2, .25, .25, .25, .3, .3, .2, .2]],
                              colLabels=stats.columns,
                              cellLoc = 'center', rowLoc = 'center',
                              loc = 'center left')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8.5)
        the_table.scale(1, 1.5)
        plt.text(0, 1, 'Post-hoc (Wilcoxon signed-rank tests)')
        
        
        # Histograms
        for idx, sp_name in enumerate(p['studies'][study]['experiments']['et']['conditions']):
            ax = fig.add_axes([0.05+(idx)*0.2, 0.2, 0.16, 0.16])
            for gp in ['control', 'dyslexic']:
                x = data[(data.condition == sp_name) & (data.group == gp)][meas_name]
                sns.distplot(x, bins=8, label=gp, ax=ax)
                plt.title(sp_name)
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        
        # %%Save fig as pdf page
        if save_pdf:
            pdf.savefig()
            plt.close()
    if save_pdf:
        pdf.close()

