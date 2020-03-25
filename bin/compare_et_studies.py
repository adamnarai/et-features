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

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))

meas_list = p['params']['et']['meas_list']
study_list = list(p['studies'].keys())

# Create reports dir
results_dir = os.path.join(RESULTS_DIR, 'compare_studies', 'et_features')
os.makedirs(results_dir, exist_ok=True)

# %% Generate pdf figures
with PdfPages(results_dir + '/ET_feature_compare.pdf') as pdf:      
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
        
        res = dict()
        res[0] = pg.mwu(plot_data[plot_data.Study == study_list[0]].value,
                             plot_data[plot_data.Study == study_list[1]].value)
        res[1] = pg.mwu(plot_data[plot_data.Study == study_list[1]].value,
                             plot_data[plot_data.Study == study_list[2]].value)
        
        # Stats
        if res:
            y_min = min(plot_data['value'])
            y_max = max(plot_data['value'])
            y = y_min-(y_max-y_min)*0.035
            for i in range(0,2):
                x1, x2 = i+0.1, i+0.9
                p_str = '{:.5f}'.format(res[i].loc['MWU','p-val'])
                if p_str:
                    plt.plot([x1, x2], [y, y], lw=2, c=c)
                    if res[i].loc['MWU','p-val'] >= .05:
                        plt.text((x1+x2)*.5, y-(y_max-y_min)*0.015, p_str, ha='center', va='top', color=c, fontsize=8)
                    else:
                        plt.text((x1+x2)*.5, y-(y_max-y_min)*0.015, p_str, ha='center', va='top', color='r', fontsize=8)
        
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.15)
        pdf.savefig()
        plt.close()
        
        
