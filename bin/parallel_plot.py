"""
Paralell plot

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
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

meas_list = dict()
# VSP (est + sum)
data_vsp = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'vsp', 'vsp.pkl'))
data_vsp.columns = data_vsp.columns.astype(str)
meas_list['vsp'] = p['params']['vsp']['meas_list_all']

# Split cond
idx = 0
plt.figure()
for col in ['0', '2']:
    for gp in ['control', 'dyslexic']:
        idx += 1
        plt.subplot(2,2,idx)
        data = data_vsp[data_vsp['group'] == gp].pivot_table(index=['group', 'subj_id'], columns='condition', values=col).reset_index()
        parallel_coordinates(data, 'group', ['SP1', 'SP2', 'SP3'])
        plt.title(f'Gp: {gp}  Var name: {col}')
        
# Test corr
data = data_vsp
eeg = pd.read_csv('D:/Temp/eeg_li.csv')
data = data.merge(eeg, how='outer', on=['subj_id'])