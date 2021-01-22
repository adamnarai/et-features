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
from utils import load_params, import_et_behav_data

# Params
experiments = ['et'] # et, 3dmh, wais, perf, vsp, word_info, eeg, proofreading, sentence_verification
conditions = ['NS']            # SP1, SP2, SP3, SP4, SP5, MS, NS, DS
dmh_type = 'Ny'                 # Ny, St, Perc (only one at a time)
et_feature_list = 'meas_list_min'   # meas_list, meas_list_all
plot_unsign = False             # As separate figure
save_pdf = False
globalplot = False
corr_method = 'pearson'     # skipped, spearman, pearson

# Load params YAML
p = load_params()

# Import data
data, plot_meas_list = import_et_behav_data(p, experiments=experiments, et_feature_list=et_feature_list)

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'correlations', 'all', corr_method)
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
    regr_stats_p = pg.corr(corr_data_sub.iloc[:,x], corr_data_sub.iloc[:,y], method='pearson')
    regr_stats_s = pg.corr(corr_data_sub.iloc[:,x], corr_data_sub.iloc[:,y], method='spearman')
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
              + regr_stats_p.to_string(col_space=10) + '\n' 
              + regr_stats_s.to_string(col_space=10) + '\n' 
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
        
def try_pg_corr(x, y, method='spearman'):
    try:
        return pg.corr(x, y, method=corr_method)
    except:
        return dict({'r':np.nan, 'p-val':np.nan})
    
    
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
            r = corr_data.corr(method=lambda x, y: try_pg_corr(x, y, method=corr_method)['r'])
            pval = corr_data.corr(method=lambda x, y: try_pg_corr(x, y, method=corr_method)['p-val'])
            
            # Triangle mask
            mask = np.zeros_like(r, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            
            # All r value
            for sign in [False, True]:
                if sign:
                    sign_str = 'significant (p<0.05)'
                    mask[(pval >= .05)] = True
                elif plot_unsign:
                    sign_str = ''
                else:
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
                plt.title(gp + ' - ' + cond + ' - ' + study + ' - correlation (' + corr_method + 
                          ') ' + sign_str, fontsize=18, fontweight='bold')
                plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
                if save_pdf:
                    pdf.savefig()
                    plt.close()
    if save_pdf:
        pdf.close()
        
# # %% Spec figures
# # Between spacings
# study = 'dys'
# # Correlation heatmaps
# gp_list = list(p['studies'][study]['groups'].values())

# for gp in gp_list:
#     corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, plot_meas_list + ['condition', 'subj_id']]
#     corr_data = corr_data.dropna(axis='columns', how='all')
#     if corr_data.empty:
#         continue

#     corr_data = corr_data.pivot(index='subj_id', columns='condition', values='Med_rspeed_wnum')    
#     labels = corr_data.index
    
#     # Spearman correlation (p values in upper triangle)
#     r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
#     r = r.replace('-', 1).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
#     # Triangle mask
#     mask = np.zeros_like(r, dtype=np.bool)
#     mask[np.triu_indices_from(mask)] = True
    
#     # All r value
#     for sign in [False, True]:
#         if sign:
#             mask[(r.T >= .05)] = True
#         elif not plot_unsign:
#             continue

#         # Correlation plot
#         fig = plt.figure(figsize=(18, 14))
#         cond_data = data[(data['study'] == study) & (data['group'] == gp)]
#         curr_func = partial(onclick, corr_data=corr_data, labels=labels, 
#                             study=study, gp=gp, cond=cond, cond_data=cond_data,
#                             fig_regrplot=fig_regrplot, fig_compare=fig_compare)
#         fig.canvas.mpl_connect('button_press_event', curr_func)
#         sns.heatmap(r,
#                     vmin=-1,
#                     vmax=1,
#                     cmap='RdBu_r',
#                     annot=True,
#                     fmt=".2f",
#                     annot_kws={"fontsize":8},
#                     mask=mask)
#         plt.title(prefix + gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
#                   fontsize=18, fontweight='bold')
#         plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        
        
# # Between SP2-5 perf, rspeed and EEG
# study = 'dys'
# # Correlation heatmaps
# gp_list = list(p['studies'][study]['groups'].values())

# for gp in gp_list:
#     corr_data = data[(data['study'] == study) & (data['group'] == gp)].loc[:, meas_list['eeg'] + 
#                                                                            ['perf', 'Med_rspeed_wnum', 'condition', 'subj_id']]
#     corr_data = corr_data.dropna(axis='columns', how='all')

#     rspeed_data = corr_data.pivot(index='subj_id', columns='condition', values='Med_rspeed_wnum')
#     perf_data = corr_data.pivot(index='subj_id', columns='condition', values='perf')
#     corr_data = corr_data[corr_data['condition'] == 'SP2']
#     corr_data.set_index('subj_id', inplace=True)
#     corr_data['rspeed_SP5-2'] = rspeed_data['SP5'] - rspeed_data['SP2']
#     corr_data['perf_SP5-2'] = perf_data['SP5'] - perf_data['SP2']
#     labels = corr_data.index
#     corr_data = corr_data.drop(labels=['condition'], axis=1)
    
#     # Spearman correlation (p values in upper triangle)
#     r = corr_data.rcorr(method='spearman', stars=False, decimals=4)
#     r = r.replace('-', 1).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
#     # Triangle mask
#     mask = np.zeros_like(r, dtype=np.bool)
#     mask[np.triu_indices_from(mask)] = True
    
#     # All r value
#     for sign in [False, True]:
#         if sign:
#             mask[(r.T >= .05)] = True
#         elif not plot_unsign:
#             continue

#         # Correlation plot
#         fig = plt.figure(figsize=(18, 14))
#         cond_data = data[(data['study'] == study) & (data['group'] == gp)]
#         curr_func = partial(onclick, corr_data=corr_data, labels=labels, 
#                             study=study, gp=gp, cond=cond, cond_data=cond_data,
#                             fig_regrplot=fig_regrplot, fig_compare=fig_compare)
#         fig.canvas.mpl_connect('button_press_event', curr_func)
#         sns.heatmap(r,
#                     vmin=-1,
#                     vmax=1,
#                     cmap='RdBu_r',
#                     annot=True,
#                     fmt=".2f",
#                     annot_kws={"fontsize":8},
#                     mask=mask)
#         plt.title(prefix + gp + ' - ' + cond + ' - ' + study + ' - correlation (Spearman) significant (p<0.05)',
#                   fontsize=18, fontweight='bold')
#         plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

