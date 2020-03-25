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
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

# Params
run_regrplots = False

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


# %% Generate pdf figures
for study in list(p['studies'].keys()):
    print('Running study: {}'.format(study))
    with PdfPages(results_dir + '/ET_feature_corr_' + study + '.pdf') as pdf:      
        # Correlation heatmaps
        for gp in list(p['studies'][study]['groups'].values()):
            for cond in p['studies'][study]['experiments']['et']['conditions']:
                print('Running corr heatmap group: {}, cond: {}'.format(gp, cond))
                corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                                 & (data['group'] == gp)].loc[:, meas_list]
                
                # Spearman correlation (p values in upper triangle)
                r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
                r = r.replace('-', 1).apply(pd.to_numeric)
                
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
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) ',
                          fontsize=18, fontweight='bold')
                pdf.savefig()
                plt.close()
                
                # Significant r values
                mask[(r.T >= .05)] = True
                fig = plt.figure(figsize=(18, 14))
                sns.heatmap(r,
                            vmin=-1,
                            vmax=1,
                            cmap='RdBu_r',
                            annot=True,
                            fmt=".2f",
                            annot_kws={"fontsize":8},
                            mask=mask)
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                          fontsize=18, fontweight='bold')
                pdf.savefig()
                plt.close()
        
        # Regrplots
        if run_regrplots:
            for gp in list(p['studies'][study]['groups'].values()):
                for cond in p['studies'][study]['experiments']['et']['conditions']:
                    print('Running scatterplots group: {}, cond: {}'.format(gp, cond))
                    corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                                     & (data['group'] == gp)].loc[:, meas_list]
                    plt_idx = 0
                    var_num = corr_data.shape[1]
                    for i in range(var_num):
                        for j in range(var_num):
                            if not plt_idx%var_num:
                                plt.figure(figsize=(20, 14))
                                plt.suptitle(gp + ' - ' + cond)
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

