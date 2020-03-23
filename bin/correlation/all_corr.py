"""
ET feature correlations

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

# Params
run_regrplots = False

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data_et = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
meas_list_et = p['params']['et']['meas_list']

data_3dmh = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh', '3dmh.pkl'))
meas_list_3dmh = p['params']['3dmh']['meas_list']['St']

data_wais = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'wais', 'wais.pkl'))
meas_list_wais = p['params']['wais']['meas_list']
data_wais['study'] = 'dys'

# Merge all data
data = data_et.merge(data_3dmh, how='outer')
data = data.merge(data_wais, how='outer')

# Merge all measure varname
meas_list = meas_list_et + meas_list_3dmh + meas_list_wais

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'correlations', 'all')
os.makedirs(results_dir, exist_ok=True)


# %% Generate pdf figures
for study in ['dys']:
    print('Running study: {}'.format(study))
    with PdfPages(results_dir + '/all_corr_' + study + '.pdf') as pdf:      
        # Correlation heatmaps
        for gp in list(p['studies'][study]['groups'].values()):
            for cond in p['studies'][study]['experiments']['et']['conditions']:
                print('Running corr heatmap group: {}, cond: {}'.format(gp, cond))
                corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                                 & (data['group'] == gp)].loc[:, meas_list]
                corr_data = corr_data.dropna(axis='columns', how='all')
                
                # Spearman correlation
                r, pval = stats.spearmanr(corr_data)
                r = corr_data.corr(method='spearman')
                
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
                            annot_kws={"fontsize":6},
                            mask=mask)
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) ',
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
                            annot_kws={"fontsize":6},
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
                    corr_data = corr_data.dropna(axis='columns', how='all')
                    
                    plt_idx = 0
                    var_num = corr_data.shape[1]
                    for i in range(var_num):
                        for j in range(var_num):
                            if not plt_idx%round(var_num/2):
                                plt.figure(figsize=(20, 14))
                                plt.suptitle(gp + ' - ' + cond)
                            if j==i:
                                plt_idx+=1
                                continue
                            plt.subplot(5,7,(plt_idx%round(var_num/2))+1)
                            sns.regplot(x=list(corr_data.columns)[i], y=list(corr_data.columns)[j], data=corr_data)
                            plt_idx+=1
                            if not plt_idx%(round(var_num/2)):
                                plt.tight_layout()
                                pdf.savefig()
                                plt.close()
