"""
Regrplot GUI

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import yaml
import pandas as pd
import pingouin as pg
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
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

# Filter studies
data_all = dict()
valid_meas_list = dict()
for study in list(p['studies'].keys()):
    data_all[study] = data[(data['study'] == study)]
    data_all[study] = data_all[study].dropna(axis='columns', how='all')
    
    # Get valid measure list
    valid_meas_list[study] = sorted(list(set(data_all[study].columns) & set(meas_list)))
    
def plot_regrplot(study):
    gp = gp_entry[study].get()
    cond = cond_entry[study].get()
    x_var = x_var_entry[study].get()
    y_var = y_var_entry[study].get()

    # Filter data
    corr_data = data_all[study][(data_all[study]['condition'] == cond)
                                & (data_all[study]['group'] == gp)].loc[:, valid_meas_list[study]]
    
    # Spearman correlation
    regr_stats = pg.corr(corr_data[x_var], corr_data[y_var], method='spearman')

    # Regrplot
    plt.figure(figsize=(7, 5))
    sns.regplot(x=x_var, y=y_var, data=corr_data)
    plt.title('study: {} | group: {} | condition: {}\n'.format(study, gp, cond)
              + regr_stats.to_string(col_space=10))

# GUI for variable selection
window = tk.Tk()
window.title("Regrplot")

gp_entry = dict()
cond_entry = dict()
x_var_entry = dict()
y_var_entry = dict()
for idx, study in enumerate(list(p['studies'].keys())):
    tk.Label(window, text = study).grid(row=0, column=idx*2+1)
    if idx == 0:
        tk.Label(window, text = "group").grid(row=1, column=idx*2)
        tk.Label(window, text = "condition").grid(row=2, column=idx*2)
        tk.Label(window, text = "X variable").grid(row=3, column=idx*2)
        tk.Label(window, text = "Y variable").grid(row=4, column=idx*2)
        
    gp_list = list(p['studies'][study]['groups'].values())
    gp_entry[study] = tk.StringVar(window)
    gp_entry[study].set(gp_list[0])
    tk.OptionMenu(window, gp_entry[study], *gp_list).grid(row=1, column=idx*2+1)
    
    cond_list = p['studies'][study]['experiments']['et']['conditions']
    cond_entry[study] = tk.StringVar(window)
    cond_entry[study].set(cond_list[0])
    tk.OptionMenu(window, cond_entry[study], *cond_list).grid(row=2, column=idx*2+1)

    x_var_entry[study] = tk.StringVar(window)
    x_var_entry[study].set(valid_meas_list[study][0])
    tk.OptionMenu(window, x_var_entry[study], *valid_meas_list[study]).grid(row=3, column=idx*2+1)

    y_var_entry[study] = tk.StringVar(window)
    y_var_entry[study].set(valid_meas_list[study][1])
    tk.OptionMenu(window, y_var_entry[study], *valid_meas_list[study]).grid(row=4, column=idx*2+1)
    
    ttk.Button(window, text="Plot", command=partial(plot_regrplot, study)).grid(row=5, column=idx*2+1)
    
window.mainloop()
