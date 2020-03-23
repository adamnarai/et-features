"""
3DMH correlations

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh', '3dmh.pkl'))

meas_list = p['params']['3dmh']['meas_list']['St']
    
# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'correlations', '3dmh')
os.makedirs(results_dir, exist_ok=True)


# %% Generate pdf figures
for study in ['dys', 'dys_contr_2']:
    print('Running study: {}'.format(study))
    with PdfPages(results_dir + '/3dmh_corr_' + study + '.pdf') as pdf:   
        # Correlation heatmaps
        for gp in list(p['studies'][study]['groups'].values()):
            print('Running corr heatmap group: {}'.format(gp))
            corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, meas_list]
            corr_data = corr_data.dropna(axis='columns', how='all')
            
            # Spearman correlation
            r, pval = stats.spearmanr(corr_data)
            r = pd.DataFrame(r, index=corr_data.columns, columns=corr_data.columns)
            
            # Triangle mask
            mask = np.zeros_like(r, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            
            # All r value
            fig = plt.figure(figsize=(18, 14))
            sns.heatmap(r,
                        vmin=-1,
                        vmax=1,
                        cmap='RdBu_r',
                        annot=True,
                        fmt=".2f",
                        annot_kws={"fontsize":8},
                        mask=mask)
            plt.title(gp + ' - ' + study + ' - correlation (Spearman) ',
                      fontsize=18, fontweight='bold')
            pdf.savefig()
            plt.close()
            
            # Significant r values
            mask[(pval >= .05)] = True
            fig = plt.figure(figsize=(18, 14))
            sns.heatmap(r,
                        vmin=-1,
                        vmax=1,
                        cmap='RdBu_r',
                        annot=True,
                        fmt=".2f",
                        annot_kws={"fontsize":8},
                        mask=mask)
            plt.title(gp + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                      fontsize=18, fontweight='bold')
            pdf.savefig()
            plt.close()
        
        # Regrplots
        for gp in list(p['studies'][study]['groups'].values()):
            print('Running scatterplots group: {}'.format(gp))
            corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, meas_list]
            corr_data = corr_data.dropna(axis='columns', how='all')
            
            plt_idx = 0
            var_num = corr_data.shape[1]
            for i in range(var_num):
                for j in range(var_num):
                    if not plt_idx%var_num:
                        plt.figure(figsize=(20, 14))
                        plt.suptitle(gp)
                    if j==i:
                        plt_idx+=1
                        continue
                    plt.subplot(5,7,(plt_idx%var_num)+1)
                    sns.regplot(x=list(corr_data.columns)[i], y=list(corr_data.columns)[j], data=corr_data)
                    plt_idx+=1
                    if not plt_idx%var_num:
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

