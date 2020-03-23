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
study = 'dys'
gp = 'control'
cond = 'SP2'

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

# Filter study
data = data[(data['study'] == study)]
data = data.dropna(axis='columns', how='all')

# Get valid measure list
valid_meas_list = list(data.loc[:, meas_list].columns)
    
def plot_regrplot():
    x_var = x_var_entry.get()
    y_var = y_var_entry.get()
    gp = gp_entry.get()
    cond = cond_entry.get()
    
    # Filter data
    corr_data = data[(data['condition'] == cond) & (data['group'] == gp)].loc[:, meas_list]
    
    # Spearman correlation
    r2, pval = stats.spearmanr(corr_data)
    r2 = pd.DataFrame(r2, index=corr_data.columns, columns=corr_data.columns)
    pval = pd.DataFrame(pval, index=corr_data.columns, columns=corr_data.columns)
    r = corr_data.corr(method='spearman')

    fig = plt.figure(figsize=(12, 8))     
    sns.regplot(x=x_var, y=y_var, data=corr_data)
    plt.title('r = {:.2f} (scipy r = {:.2f})  p = {:.4f}'
              .format(r.loc[x_var, y_var], r2.loc[x_var, y_var], pval.loc[x_var, y_var]))

# GUI for variable selection
from tkinter import *
from tkinter import ttk
window = Tk()
window.title("Variable selector for regrplot")
window.geometry('600x400')
window.configure(background = "white")

Label(window, text = "group").grid(row=0, column=0)
gp_list = list(p['studies'][study]['groups'].values())
gp_entry = StringVar(window)
gp_entry.set(gp_list[0])
OptionMenu(window, gp_entry, *set(gp_list)).grid(row=0, column=1)

Label(window, text = "condition").grid(row=1, column=0)
cond_list = p['studies'][study]['experiments']['et']['conditions']
cond_entry = StringVar(window)
cond_entry.set(cond_list[0])
OptionMenu(window, cond_entry, *set(cond_list)).grid(row=1, column=1)

Label(window, text = "X variable").grid(row=2, column=0)
x_var_entry = StringVar(window)
x_var_entry.set(valid_meas_list[0])
OptionMenu(window, x_var_entry, *set(valid_meas_list)).grid(row=2, column=1)

Label(window, text = "Y variable").grid(row=3, column=0)
y_var_entry = StringVar(window)
y_var_entry.set(valid_meas_list[1])
OptionMenu(window, y_var_entry, *set(valid_meas_list)).grid(row=3, column=1)

btn = ttk.Button(window, text="Plot", command=plot_regrplot).grid(row=5, column=1)

window.mainloop()
