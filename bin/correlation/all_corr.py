"""
ET feature and behav correlations

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
experiments = ['et']            # et, 3dmh, wais, perf, vsp, word_info
conditions = ['MS', 'NS', 'DS']      # SP1, SP2, SP3, SP4, SP5, MS, NS, DS
dmh_type = 'Ny'                 # Ny, St, Perc (only one at a time)
et_feature_list = 'meas_list'   # meas_list, meas_list_all
plot_unsign = False             # As separate figure
save_pdf = False
globalplot = True

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
    meas_list = dict()
# ET
data_et = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
meas_list['et'] = p['params']['et'][et_feature_list]

# 3DMH
data_3dmh = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh', '3dmh.pkl'))
meas_list['3dmh'] = p['params']['3dmh']['meas_list'][dmh_type]

# WAIS
data_wais = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'wais', 'wais.pkl'))
meas_list['wais'] = p['params']['wais']['meas_list']
data_wais['study'] = 'dys'

# VSP (est + sum)
data_vsp = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'vsp', 'vsp.pkl'))
data_vsp.columns = data_vsp.columns.astype(str)
meas_list['vsp'] = p['params']['vsp']['meas_list_all']

# Reading perf
data_perf = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_perf.pkl'))
meas_list['perf'] = ['perf']

# Word info
data_word_info = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'word_info.pkl'))
meas_list['word_info'] = p['params']['et']['word_info_list']

# Merge all data
data = data_et.merge(data_3dmh, how='outer', on=['study', 'group', 'subj_id'])
data = data.merge(data_wais, how='outer', on=['study', 'group', 'subj_id'])
data = data.merge(data_vsp, how='outer', on=['group', 'condition', 'spacing_size', 'subj_id'])
data = data.merge(data_perf, how='outer', on=['study', 'group', 'condition', 'spacing_size', 'subj_id'])
data = data.merge(data_word_info, how='outer', on=['study', 'group', 'condition', 'subj_id'])

# Merge all measure varname
plot_meas_list = sum([meas_list[exp] for exp in experiments], [])

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'correlations', 'all')
os.makedirs(results_dir, exist_ok=True)

if globalplot:
    fig_regrplot = plt.figure(figsize=(7, 5.5))
    fig_compare = plt.figure(figsize=(7, 5.5))
else:
    fig_regrplot = None
    fig_compare = None
    
### Functions ###
def onclick(event, corr_data, labels, study, gp, cond, cond_data=None, fig_regrplot=None, fig_compare=None):  
    x = int(event.xdata)
    y = int(event.ydata)
    
    # Remove outliers
    corr_data_sub = corr_data.dropna(subset=list(corr_data.columns[[x, y]]))
    labels_sub = labels[corr_data.iloc[:,x].notna() & corr_data.iloc[:,y].notna()]
                
    # Spearman correlation
    regr_stats = pg.corr(corr_data_sub.iloc[:,x], corr_data_sub.iloc[:,y], method='spearman')
    r, pval, outliers = pg.correlation.skipped(corr_data_sub.iloc[:,x], corr_data_sub.iloc[:,y], method='spearman')

    # Regrplot
    if not isinstance(fig_regrplot, plt.Figure):
        fig_regrplot = plt.figure(figsize=(7, 5.5))
    plt.figure(fig_regrplot.number)
    plt.cla()
    ax = sns.regplot(x=corr_data_sub.columns[x], y=corr_data_sub.columns[y], data=corr_data_sub)
    
    for i, txt in enumerate(labels_sub):
        ax.annotate(txt, (corr_data_sub.iloc[i,x], corr_data_sub.iloc[i,y]))
    plt.title('study: {} | group: {} | condition: {}\n'.format(study, gp, cond)
              + regr_stats.to_string(col_space=10) + '\n' 
              + 'skipped: r = {:.2f}  p = {:.4f}  outliers: {}'.format(r, pval, ', '.join(labels_sub[outliers])),
              fontsize=12)
    plt.subplots_adjust(top=0.85)
    plt.draw()
    
    # Compare correlations along conditions
    if isinstance(cond_data, pd.DataFrame):
        if not isinstance(fig_compare, plt.Figure):
            fig_compare = plt.figure(figsize=(7, 5.5))
        plt.figure(fig_compare.number)
        plt.cla()
        varname_x = corr_data.columns[x]
        varname_y = corr_data.columns[y]
        d = dict()
        cond_list = cond_data['condition'].unique()
        for cond in cond_list:
            curr_cond_data = cond_data[cond_data['condition'] == cond]
            stats = pg.corr(curr_cond_data[varname_x], curr_cond_data[varname_y], method='skipped')
            d[cond] = dict()
            d[cond]['r'] = stats.loc['skipped', 'r']
            d[cond]['CI'] = stats.loc['skipped', 'CI95%']
        r_list = [d[cond]['r'] for cond in cond_list]
        yerr = abs(np.stack([d[cond]['CI'] for cond in cond_list], axis=0).transpose() - r_list)
        plt.errorbar(cond_list, r_list, yerr)
        plt.title(f'{varname_x} - {varname_y}\nSkipped Spearman r with 95% CI')
        plt.ylim(-1, 1)
        plt.draw()
#################

# Generate figures
for study in ['letter_spacing']:#list(p['studies'].keys()):
    print('Running study: {}'.format(study))
    if save_pdf:
        pdf = PdfPages(results_dir + '/ET_feature_corr_' + study + '.pdf')     
    
    # Correlation heatmaps
    for gp in list(p['studies'][study]['groups'].values()):
        for cond in conditions:
            corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].loc[:, plot_meas_list]
            labels = data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].subj_id
            corr_data = corr_data.dropna(axis='columns', how='all')
            if corr_data.empty:
                continue
            print('Running corr heatmap group: {}, cond: {}'.format(gp, cond))
            
            # Spearman correlation (p values in upper triangle)
            r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
            r = r.replace('-', 1).apply(lambda x: pd.to_numeric(x, errors='coerce'))
            
            # Triangle mask
            mask = np.zeros_like(r, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            
            # All r value
            for sign in [False, True]:
                if sign:
                    mask[(r.T >= .05)] = True
                elif not plot_unsign:
                    continue

                # Correlation plot
                fig = plt.figure(figsize=(18, 14))
                cond_data = data[(data['study'] == study) & (data['group'] == gp)]
                curr_func = partial(onclick, corr_data=corr_data, labels=labels, 
                                    study=study, gp=gp, cond=cond, cond_data=cond_data,
                                    fig_regrplot=fig_regrplot, fig_compare=fig_compare)
                fig.canvas.mpl_connect('button_press_event', curr_func)
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
                plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
                if save_pdf:
                    pdf.savefig()
                    plt.close()
    if save_pdf:
        pdf.close()
