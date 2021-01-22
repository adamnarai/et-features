"""


@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR
import yaml
import pingouin as pg

# Params
study = 'letter_spacing'
group = 'control'
cond = 'NS'

def try_pg_corr(x, y, method='spearman'):
    try:
        return pg.corr(x, y, method=corr_method)
    except:
        return dict({'r':np.nan, 'p-val':np.nan})

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)
    
# Read ET data
meas_list = p['params']['et']['meas_list_min']
data = pd.read_pickle(os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
data = data.loc[(data['study'] == study) & (data['condition'] == cond) & (data['group'] == group), 
                ['subj_id'] + meas_list]

# # Get perf data
# perf_df = pd.read_csv('D:/letter_spacing/letter_spacing_perf.csv')
# meas_list = meas_list + ['perf']
# data = pd.concat([data.set_index('subj_id'), perf_df.set_index('subj_id')], axis=1)

# Get correlation matrices
r = dict()
pval = dict()
for corr_method in ['pearson', 'spearman', 'skipped']:
    r[corr_method] = data.corr(method=lambda x, y: try_pg_corr(x, y, method=corr_method)['r'])
    pval[corr_method] = data.corr(method=lambda x, y: try_pg_corr(x, y, method=corr_method)['p-val'])

# Triangle mask
mask = np.zeros_like(r['pearson'], dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Create combined matrices
mask = np.zeros_like(r['pearson'], dtype=np.int)
mask[np.triu_indices_from(mask)] = 1

# Skipped Spearman + Pearson
r_skip_pear = pd.DataFrame(r['skipped'].values*mask.T + r['pearson'].values*mask,
                           index=r['pearson'].index, columns=r['pearson'].columns)
pval_skip_pear = pval['skipped'].values*mask.T + pval['pearson'].values*mask
mask_skip_pear = np.zeros_like(r['pearson'], dtype=np.bool)
mask_skip_pear[(pval_skip_pear >= .05)] = True

# Spearman + Pearson
r_spear_pear = pd.DataFrame(r['spearman'].values*mask.T + r['pearson'].values*mask,
                           index=r['pearson'].index, columns=r['pearson'].columns)
pval_spear_pear = pval['spearman'].values*mask.T + pval['pearson'].values*mask
mask_spear_pear = np.zeros_like(r['pearson'], dtype=np.bool)
mask_spear_pear[(pval_spear_pear >= .05)] = True

# %% Correlation plot
# fig = plt.figure(figsize=(6,5))
# sns.regplot(data=data, x="Med_rspeed_wnum", y="perf")
# plt.title(cond)
# stats = pg.corr(data["Med_rspeed_wnum"], data["perf"], method='pearson')
# print('Pearson:  r: {}, p: {}'.format(stats['r'][0], stats['p-val'][0]))
# stats = pg.corr(data["Med_rspeed_wnum"], data["perf"], method='spearman')
# print('Spearman:  r: {}, p: {}'.format(stats['r'][0], stats['p-val'][0]))

# corr_r = r_skip_pear
# corr_mask = mask_skip_pear

corr_r = r_spear_pear
corr_mask = mask_spear_pear

fig = plt.figure(figsize=(9.5, 9))
sns.heatmap(corr_r,
            vmin=-1,
            vmax=1,
            cmap='RdBu_r',
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize":7},
            mask= corr_mask,
            cbar_kws={'fraction':0.01, 'aspect':30, 'ticks':[-1, 0, 1]})
plt.plot((0, len(meas_list)), (0, len(meas_list)), color='gray')

plt.subplots_adjust(left=0.17, right=0.96, top=0.98, bottom=0.17)

plt.savefig('corr_mat.svg', format='svg')
plt.savefig('corr_mat_1200dpi.png', dpi=1200)
# plt.savefig('corr_mat_600dpi.png', dpi=600)
# plt.savefig('corr_mat_220dpi.png', dpi=220)

