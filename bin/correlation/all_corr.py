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
experiments = ['et'] # et, 3dmh, wais, perf, vsp, word_info, eeg, proofreading, sentence_verification
conditions = ['SP2']            # SP1, SP2, SP3, SP4, SP5, MS, NS, DS
dmh_type = 'Ny'                 # Ny, St, Perc (only one at a time)
et_feature_list = 'meas_list'   # meas_list, meas_list_all
plot_unsign = False             # As separate figure
save_pdf = False
globalplot = False
corr_method = 'skipped'     # skipped, spearman, pearson

only_NS_et = False
only_NS_vsp = False
prefix = ''
if only_NS_et:
    prefix += '!!! only NS ET '
if only_NS_vsp:
    prefix += '!!! only NS VSP '

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
meas_list = dict()
# ET
data_et = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
meas_list['et'] = p['params']['et'][et_feature_list]
if only_NS_et:
    for sp in [1, 3, 4, 5]:
        data_et.loc[(data_et['condition'] == f'SP{sp}') & (data_et['study'] == 'dys'), 
                    (data_et.columns != 'condition') & (data_et.columns != 'spacing_size')] = \
            data_et.loc[(data_et['condition'] == 'SP2') & (data_et['study'] == 'dys'),
                        (data_et.columns != 'condition') & (data_et.columns != 'spacing_size')].values
        
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
if only_NS_vsp:
    for sp in [1, 3]:
        data_vsp.loc[data_vsp['condition'] == f'SP{sp}', 
                     (data_vsp.columns != 'condition') & (data_vsp.columns != 'spacing_size')] = \
            data_vsp.loc[data_vsp['condition'] == 'SP2', 
                         (data_vsp.columns != 'condition') & (data_vsp.columns != 'spacing_size')].values

# Reading perf
data_perf = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_perf.pkl'))
meas_list['perf'] = ['perf']

# Word info
data_word_info = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'word_info.pkl'))
meas_list['word_info'] = p['params']['et']['word_info_list']

# EEG
data_eeg = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'eeg', 'eeg_peak_data.pkl'))
meas_list['eeg'] = p['params']['eeg']['meas_list']

# Proofreading (ET features)
data_proofreading = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'proofreading', 'proofreading.pkl'))
meas_list['proofreading'] = p['params']['proofreading']['meas_list']

# Sentence verification (ET features)
data_sentence_verification = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'sentence_verification', 'sentence_verification.pkl'))
meas_list['sentence_verification'] = p['params']['sentence_verification']['meas_list']

# Merge all data
data = data_et.merge(data_3dmh, how='outer', on=['study', 'group', 'subj_id'])
data = data.merge(data_wais, how='outer', on=['study', 'group', 'subj_id'])
data = data.merge(data_vsp, how='outer', on=['group', 'condition', 'spacing_size', 'subj_id'])
data = data.merge(data_perf, how='outer', on=['study', 'group', 'condition', 'spacing_size', 'subj_id'])
data = data.merge(data_word_info, how='outer', on=['study', 'group', 'condition', 'subj_id'])
data = data.merge(data_eeg, how='outer', on=['study', 'group', 'condition', 'subj_id'])
data = data.merge(data_proofreading, how='outer', on=['study', 'group', 'subj_id'], suffixes=('','_proof'))
data = data.merge(data_sentence_verification, how='outer', on=['study', 'group', 'subj_id'], suffixes=('','_verif'))

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
        cond_list = cond_data.loc[cond_data[[varname_x, varname_y]].notna().all(axis=1), 'condition'].unique()
        print(cond_list)
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
for study in list(p['studies'].keys()):
    print('Running study: {}'.format(study))
    if save_pdf:
        pdf = PdfPages(results_dir + '/ET_feature_corr_' + study + '.pdf')     
    
    # Correlation heatmaps
    gp_list = list(p['studies'][study]['groups'].values())
    if len(gp_list) > 1:
        gp_list += ['all_gp']
    for gp in gp_list:
        for cond in conditions:
            if gp == 'all_gp':
                corr_data = data[(data['study'] == study) & (data['condition'] == cond)].loc[:, plot_meas_list]
                labels = data[(data['study'] == study) & (data['condition'] == cond)].subj_id
            else:
                corr_data = data[(data['study'] == study) & (data['condition'] == cond) 
                                 & (data['group'] == gp)].loc[:, plot_meas_list]
                labels = data[(data['study'] == study) & (data['condition'] == cond) 
                                 & (data['group'] == gp)].subj_id
            corr_data = corr_data.dropna(axis='columns', how='all')
            if corr_data.empty:
                continue
            print('Running corr heatmap group: {}, cond: {}'.format(gp, cond))
            
            # Spearman correlation (p values in upper triangle)
            # r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
            # r = r.replace('-', 1).apply(lambda x: pd.to_numeric(x, errors='coerce'))
            # pval = r.T
            r = corr_data.corr(method=lambda x, y: pg.corr(x, y, method=corr_method)['r'])
            pval = corr_data.corr(method=lambda x, y: pg.corr(x, y, method=corr_method)['p-val'])
            
            # Triangle mask
            mask = np.zeros_like(r, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            
            # All r value
            for sign in [False, True]:
                if sign:
                    mask[(pval >= .05)] = True
                elif not plot_unsign:
                    continue

                # Correlation plot
                fig = plt.figure(figsize=(18, 14))
                if gp == 'all_gp':
                    cond_data = data[data['study'] == study]
                else:
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
                plt.title(prefix + gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                          fontsize=18, fontweight='bold')
                plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
                if save_pdf:
                    pdf.savefig()
                    plt.close()
    if save_pdf:
        pdf.close()
        
# %% Spec figures
# Between spacings
study = 'dys'
# Correlation heatmaps
gp_list = list(p['studies'][study]['groups'].values())

for gp in gp_list:
    corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, plot_meas_list + ['condition', 'subj_id']]
    corr_data = corr_data.dropna(axis='columns', how='all')
    if corr_data.empty:
        continue

    corr_data = corr_data.pivot(index='subj_id', columns='condition', values='Med_rspeed_wnum')    
    labels = corr_data.index
    
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
        plt.title(prefix + gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                  fontsize=18, fontweight='bold')
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        
        
# Between SP2-5 perf, rspeed and EEG
study = 'dys'
# Correlation heatmaps
gp_list = list(p['studies'][study]['groups'].values())

for gp in gp_list:
    corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, meas_list['eeg'] + 
                                                                           ['perf', 'Med_rspeed_wnum', 'condition', 'subj_id']]
    corr_data = corr_data.dropna(axis='columns', how='all')

    rspeed_data = corr_data.pivot(index='subj_id', columns='condition', values='Med_rspeed_wnum')
    perf_data = corr_data.pivot(index='subj_id', columns='condition', values='perf')
    corr_data = corr_data[corr_data['condition'] == 'SP2']
    corr_data.set_index('subj_id', inplace=True)
    corr_data['rspeed_SP5-2'] = rspeed_data['SP5'] - rspeed_data['SP2']
    corr_data['perf_SP5-2'] = perf_data['SP5'] - perf_data['SP2']
    labels = corr_data.index
    corr_data = corr_data.drop(labels=['condition'], axis=1)
    
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
        plt.title(prefix + gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
                  fontsize=18, fontweight='bold')
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

