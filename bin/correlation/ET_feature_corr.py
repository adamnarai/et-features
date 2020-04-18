"""
ET feature correlations

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import yaml
import pandas as pd
import pingouin as pg
import numpy as np
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

# Params
save_pdf = False
plot_unsign = False

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))

meas_list = p['params']['et']['meas_list']
    
# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'correlations', 'et_features')
os.makedirs(results_dir, exist_ok=True)

# Functions
def onclick(event, corr_data, labels, study, gp, cond):  
    x = int(event.xdata)
    y = int(event.ydata)
    
    # Spearman correlation
    regr_stats = pg.corr(corr_data.iloc[:,x], corr_data.iloc[:,y], method='spearman')
    r, pval, outliers = pg.correlation.skipped(corr_data.iloc[:,x], corr_data.iloc[:,y], method='spearman')

    # Regrplot
    plt.figure(figsize=(7, 5.5))
    ax = sns.regplot(x=corr_data.columns[x], y=corr_data.columns[y], data=corr_data)
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (corr_data.iloc[i,x], corr_data.iloc[i,y]))
    plt.title('study: {} | group: {} | condition: {}\n'.format(study, gp, cond)
              + regr_stats.to_string(col_space=10) + '\n' 
              + 'skipped: r = {:.2f}  p = {:.4f}  outliers: {}'.format(r, pval, ', '.join(labels[outliers])))
    plt.subplots_adjust(top=0.85)

# %% Generate pdf figures
for study in list(p['studies'].keys()):
    print('Running study: {}'.format(study))
    if save_pdf:
        pdf = PdfPages(results_dir + '/ET_feature_corr_' + study + '.pdf')     
    
    # Correlation heatmaps
    for gp in list(p['studies'][study]['groups'].values()):
        for cond in ['SP2']:#p['studies'][study]['experiments']['et']['conditions']:
            if study == 'letter_spacing' and cond == 'SP2':
                cond= 'NS'
            corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].loc[:, meas_list]
            labels = data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].subj_id
            if corr_data.empty:
                continue
            print('Running corr heatmap group: {}, cond: {}'.format(gp, cond))
            
            # Spearman correlation (p values in upper triangle)
            r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
            r = r.replace('-', 1).apply(pd.to_numeric)
            
            # Triangle mask
            mask = np.zeros_like(r, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            
            # All r value
            if plot_unsign:
                fig = plt.figure(figsize=(18, 14))
                sns.heatmap(r,
                            vmin=-1,
                            vmax=1,
                            cmap='RdBu_r',
                            annot=True,
                            fmt=".2f",
                            annot_kws={"fontsize":8},
                            mask=mask)
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) ',
                          fontsize=18, fontweight='bold')
                if save_pdf:
                    pdf.savefig()
                    plt.close()
            
            # Significant r values
            mask[(r.T >= .05)] = True
            fig = plt.figure(figsize=(18, 14))            
            curr_func = partial(onclick, corr_data=corr_data, labels=labels, study=study, gp=gp, cond=cond)
            fig.canvas.mpl_connect('button_press_event', curr_func)
            sns.heatmap(r,
                        vmin=-1,
                        vmax=1,
                        cmap='RdBu_r',
                        annot=True,
                        fmt=".2f",
                        annot_kws={"fontsize":6},
                        mask=mask)
            plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                      fontsize=18, fontweight='bold')
            plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
            if save_pdf:
                pdf.savefig()
                plt.close()
    if save_pdf:
        pdf.close()


