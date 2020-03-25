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
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

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
                            annot_kws={"fontsize":6},
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
                            annot_kws={"fontsize":6},
                            mask=mask)
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                          fontsize=18, fontweight='bold')
                pdf.savefig()
                plt.close()
