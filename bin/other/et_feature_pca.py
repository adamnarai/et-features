import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler

# Read data
results_path = 'D:/control_experiment/results/et_features'
df = pd.read_pickle(results_path+'/df_et.pkl')

# Parameters
gp_list = ['control']               # Group filter
sp_list = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']   # Spacing filter
sp_list = ['SP2']

meas_list = [
    "group",
    "spacing",
    "Med_rspeed_wnum",
    "Med_fixdur",
    "Med_saccamp",
    "Perc_fsacc",
    "Perc_sacc_gliss",
    "Perc_fsacc_fgliss",
    "Perc_bsacc_fgliss",
    "Med_fsaccamp",
    "Med_bsaccamp",
    "Perc_fgliss",
    "Perc_fsacc_gliss",
    "Perc_bsacc_gliss",
    "Freq_fsacc",
    "Freq_bsacc",
    "Mean_meanfixvel",
    "Mean_stdevfixvel",
    "Med_saccpdetth",
    "Med_saccdur",
    "Med_fsaccdur",
    "Med_bsaccdur",
    "Med_glissdur",
    "Med_fglissdur",
    "Med_bglissdur",
    "Med_glissamp",
    "Med_fglissamp",
    "Med_bglissamp",
    "Med_tsnum_wnum",
    "Med_fsnum_wnum",
    "Med_bsnum_wnum",
    "Med_saccpvel",
    "Perc_lvel_gliss",
    "Med_glisspvel",
    "Med_lvglisspvel",
    "Med_hvglisspvel"
    ]

meas_list = [
    # "group",
    # "spacing",
    "Med_rspeed_wnum",
    "Med_fixdur",
#    "Med_saccamp",
    "Perc_fsacc",
#    "Perc_sacc_gliss",
    "Perc_fsacc_fgliss",
    "Perc_bsacc_fgliss",
    "Med_fsaccamp",
    "Med_bsaccamp",
    "Perc_fgliss",
    "Perc_fsacc_gliss",
    "Perc_bsacc_gliss",
    "Freq_fsacc",
    "Freq_bsacc",
#    "Mean_meanfixvel",
#    "Mean_stdevfixvel",
#    "Med_saccpdetth",
#    "Med_saccdur",
    "Med_fsaccdur",
    "Med_bsaccdur",
#    "Med_glissdur",
    "Med_fglissdur",
    "Med_bglissdur",
#    "Med_glissamp",
    "Med_fglissamp",
    "Med_bglissamp",
#    "Med_tsnum_wnum",
    # "Med_fsnum_wnum",
    # "Med_bsnum_wnum",
    "Med_saccpvel",
    "Perc_lvel_gliss",
#    "Med_glisspvel",
    "Med_lvglisspvel",
    "Med_hvglisspvel"
    ]
    
# Create reports dir
os.makedirs(results_path, exist_ok=True)

# Filters (group and spacing)
df = df[df.Group.isin(gp_list)]
df = df[df.spacing.isin(sp_list)]

# Change spacing variable to continuous
df['spacing'] = df['spacing'].replace({'SP1':0.707, 'SP2':1, 'SP3':1.3, 'SP4':1.6, 'SP5':1.9})

# Change group names to dummy coding
df = df.rename(columns={'Group': 'group'})
df['group'] = df['group'].replace({'control': 0, 'dyslexic': 1})
X = df[meas_list]

# z-score data
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns = X.columns)

#%% PCA
pca = PCA()
pca.fit(X)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
var = (pca.explained_variance_ratio_*100).round(2)
cmax = np.amax(abs(loadings))

# Plot PCA loadings
fig = plt.figure(figsize=(12,10))
ax = sns.heatmap(loadings, cmap='RdBu_r', vmin=-cmax, vmax=cmax, 
                 annot=False, fmt=".2f", annot_kws={"fontsize":8})
ax.set_xlabel('Explained variance (%)')
ax.set_ylabel('Features')
ax.set_yticklabels(list(X.columns), rotation = 0)
ax.set_xticklabels(var, rotation = 90)
ax.set_title('Groups: {}  Conditions: {}'.format(', '.join(gp_list), ', '.join(sp_list)), pad=20)
fig.subplots_adjust(top=0.90, bottom=0.15, hspace=0.3, left=0.2, right=0.95)

# Bugfix for matplotlib 3.1.1
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# #%% Sparse PCA
# pca = SparsePCA(alpha=0.01)
# pca.fit(X)
# loadings = pca.components_.T 
# cmax = np.amax(abs(loadings))

# # Plot PCA loadings
# fig = plt.figure(figsize=(12,10))
# ax = sns.heatmap(loadings, cmap='RdBu_r', vmin=-cmax, vmax=cmax, 
#                   annot=False, fmt=".2f", annot_kws={"fontsize":8})
# ax.set_xlabel('Explained variance (%)')
# ax.set_ylabel('Features')
# ax.set_yticklabels(list(X.columns), rotation = 0)
# ax.set_title('Groups: {}  Conditions: {}'.format(', '.join(gp_list), ', '.join(sp_list)), pad=20)

# # Bugfix for matplotlib 3.1.1
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)