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
save_pdf = False

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))

study_list = list(p['studies'].keys())

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'compare_studies', 'et_features')
os.makedirs(results_dir, exist_ok=True)

# %% Generate pdf figures
for meas_type in ['', '_all']:
    meas_list = p['params']['et']['meas_list' + meas_type]
    if save_pdf:
        pdf = PdfPages(results_dir + '/ET_feature_compare' + meas_type + '.pdf')     
    for feature in meas_list:
        print('Running feature: {}'.format(feature))
        temp_data = dict()
        for study in study_list:
            gp = 'control'
            if study == 'letter_spacing':
                cond = 'NS'
            else:
                cond = 'SP2'
            temp_data[study] = (data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].loc[:, feature]).reset_index(drop=True)
        plot_data = pd.DataFrame(temp_data).melt(var_name='Study').dropna()
        
        # Boxplot
        c = (.3,.3,.3)
        fig = plt.figure(figsize=(5, 5))
        sns.swarmplot(x='Study', y='value', data=plot_data, color=c)
        sns.boxplot(x='Study', y='value', data=plot_data, showmeans=False, width=0.8)
                    # meanprops={'marker':'.', 'markerfacecolor':'red', 'markeredgecolor':'red'}
        plt.ylabel(feature)
        
        y_min = min(plot_data['value'])
        y_max = max(plot_data['value'])
        for i in range(0,3):
            if i == 2:
                a = 0
                b = 2
                y = y_min-(y_max-y_min)*0.095
            else:
                a = i
                b = i + 1
                y = y_min-(y_max-y_min)*0.035
            
            # Stats (independent)
            res_t = pg.ttest(x=plot_data[plot_data.Study == study_list[a]].value,
                             y=plot_data[plot_data.Study == study_list[b]].value)
            res_u = pg.mwu(x=plot_data[plot_data.Study == study_list[a]].value,
                            y=plot_data[plot_data.Study == study_list[b]].value)
            
            x1, x2 = a+0.1, b-0.1
            p_str = '{:.5f} / {:.5f}'.format(res_u.loc['MWU','p-val'], res_t.loc['T-test', 'p-val'])
            plt.plot([x1, x2], [y, y], lw=2, c=c)
            
            # Text color
            if (res_u.loc['MWU','p-val'] >= .05) and (res_t.loc['T-test', 'p-val'] >= 0.05):
                color = c
            else:
                color = 'r'
            plt.text((x1+x2)*.5, y-(y_max-y_min)*0.015, p_str, ha='center',
                     va='top', color=color, fontsize=8)
            plt.title('p-values: Welch T-test / Mann-Whitney U Test', fontsize = 8)
        
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.15)
        if save_pdf:
            pdf.savefig()
            plt.close()
    if save_pdf:
        pdf.close()
        
        
