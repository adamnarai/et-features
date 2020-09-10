"""
ET feature compare between studies

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import yaml
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

# Params
save_pdf = True

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

study_list = list(p['studies'].keys()) + ['verif', 'proof']

# Boxplot statistics
def boxplot_stats(plot_data, study_list, ax):
    y_min = min(plot_data['value'])
    y_max = max(plot_data['value'])
    for i in range(0,5):
        if i == 2:
            a = 0
            b = 2
            y = y_min-(y_max-y_min)*0.095
        elif i > 2:
            a = 1
            b = i
            y = y_min-(y_max-y_min)*(0.095+0.050*(i-2))
        else:
            a = i
            b = i + 1
            y = y_min-(y_max-y_min)*0.035
        
        # Stats (independent)
        try:
            data_a = plot_data[plot_data.study == study_list[a]].value.dropna()
            data_b = plot_data[plot_data.study == study_list[b]].value.dropna()
            res_t = pg.ttest(x=data_a, y=data_b)
            res_t = res_t.loc['T-test', 'p-val']

            res_u = pg.mwu(x=data_a, y=data_b)
            res_u = res_u.loc['MWU', 'p-val']
            
            x1, x2 = a+0.1, b-0.1
            p_str = '{:.5f} / {:.5f}'.format(res_u, res_t)
            plt.plot([x1, x2], [y, y], lw=2, c=(.3,.3,.3))
            
            # Text color
            if (res_u >= .05) and (res_t >= 0.05):
                color = (.3,.3,.3)
            else:
                color = 'r'
            plt.text((x1+x2)*.5, y-(y_max-y_min)*0.015, p_str, ha='center',
                     va='top', color=color, fontsize=6)
            # Debug
            # print('A: {}, B: {}, Sum: {}'.format(len(data_a), len(data_b), len(data_a)+len(data_b)))
        except:
            pass

        plt.title('p-values: Welch T-test / Mann-Whitney U Test', fontsize = 8)
                
# %% ET features (3 studies)
# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'compare_studies', 'et_features')
os.makedirs(results_dir, exist_ok=True)

# Read data (ET from dys and control_2 studies)
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))

# Proofreading (ET features)
data_proofreading = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'proofreading', 'proofreading.pkl'))
data_proofreading['study'] = 'proof'
data_proofreading['condition'] = 'SP2'

# Sentence verification (ET features)
data_sentence_verification = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'sentence_verification', 'sentence_verification.pkl'))
data_sentence_verification['study'] = 'verif'
data_sentence_verification['condition'] = 'SP2'

data = pd.concat([data, data_proofreading, data_sentence_verification], ignore_index=True, sort=False)

# Generate plots
for meas_type in ['', '_all']:
    meas_list = p['params']['et']['meas_list' + meas_type]
    if save_pdf:
        pdf = PdfPages(results_dir + '/ET_feature_compare' + meas_type + '.pdf')
    for feature in meas_list:
        print('Running feature: {}'.format(feature))
        
        # Get data
        temp_data = dict()
        for study in study_list:
            gp = 'control'
            if study == 'letter_spacing':
                cond = 'NS'
            else:
                cond = 'SP2'
            temp_data[study] = (data[(data['study'] == study) & (data['condition'] == cond)
                              & (data['group'] == gp)].loc[:, feature]).reset_index(drop=True)
        plot_data = pd.DataFrame(temp_data).melt(var_name='study').dropna()
        
        # Boxplot
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        sns.swarmplot(x='study', y='value', data=plot_data, color=(.3,.3,.3), ax=ax)
        sns.boxplot(x='study', y='value', data=plot_data, showmeans=False, width=0.8, ax=ax)
                    # meanprops={'marker':'.', 'markerfacecolor':'red', 'markeredgecolor':'red'}
        plt.ylabel(feature)
        
        # Statistics
        boxplot_stats(plot_data, study_list, ax)
        
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.15)
        if save_pdf:
            pdf.savefig()
            plt.close()
    if save_pdf:
        pdf.close()
        
# %% Word info and 3DMH pseudoword reading (2 studies - 3 groups)
# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'compare_studies', 'behav')
os.makedirs(results_dir, exist_ok=True)

# 3DMH
data_3dmh = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh', '3dmh.pkl'))
meas_list_3dmh = sum(list(p['params']['3dmh']['meas_list'].values()), [])

# WAIS
data_wais = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'wais', 'wais.pkl'))
meas_list_wais = p['params']['wais']['meas_list']
data_wais['study'] = 'dys'

# VSP (est + sum)
data_vsp = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'vsp', 'vsp.pkl'))
meas_list_vsp = p['params']['vsp']['meas_list']

# Reading perf
data_perf = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_perf.pkl'))
meas_list_perf = ['perf']

# Word info
data_word_info = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'word_info.pkl'))
meas_list_word_info = p['params']['et']['word_info_list']

# Merge all data
data = data_3dmh.merge(data_wais, how='outer', on=['study', 'group', 'subj_id'])
data = data.merge(data_vsp, how='outer', on=['group', 'subj_id'])
data = data.merge(data_perf, how='outer', on=['study', 'group', 'condition', 'spacing_size', 'subj_id'])
data = data.merge(data_word_info, how='outer', on=['study', 'group', 'condition', 'subj_id'])

# Limit conditions
data = data[(data.condition == 'SP2') | data.condition.isna()]

# Merge all measure varname
meas_list = (
    meas_list_3dmh 
    + meas_list_wais
    + meas_list_perf
    + meas_list_vsp
    + meas_list_word_info
    )

if save_pdf:
    pdf = PdfPages(results_dir + '/behav_compare.pdf')

data['study'] = data[['study', 'group']].apply(lambda x: '_'.join(x), axis=1)
for feature in meas_list:
    print('Running feature: {}'.format(feature))
    
    # Boxplot
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    sns.swarmplot(x='study', y=feature, data=data, color=(.3,.3,.3), ax=ax)
    sns.boxplot(x='study', y=feature, data=data, showmeans=False, width=0.8, ax=ax)
    plt.ylabel(feature)
    
    # Statistics
    boxplot_stats(data.melt(id_vars='study', value_vars=feature), list(data.study.unique()), ax)
    
    plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.15)
    if save_pdf:
        pdf.savefig()
        plt.close()
if save_pdf:
    pdf.close()
