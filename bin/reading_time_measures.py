import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Params
gp_list = ['control', 'dyslexic', 'control_2']
sp_list = [1, 2, 3, 4, 5]
sp_list = [2]
res_path_1 = 'Y:/dyslexia_git_pub/results/word_info/word_info_all.csv'
res_path_2 = 'Y:/control_experiment_pub/results/word_info/word_info_all.csv'
var_list = ['freq_pM', 'word_length', 'fix_num', 'first_fix_dur', 'first_pass_reading_time',
            'total_reading_time', 'regr_path_reading_time', 're_reading_time',
            'regr_sacc_num']

# Load data
df_1 = pd.read_csv(res_path_1, dtype = {'subj_code': str}, encoding = 'unicode_escape')
df_2 = pd.read_csv(res_path_2, dtype = {'subj_code': str}, encoding = 'unicode_escape')
df_2['group'] = 'control_2'
df = pd.concat([df_1, df_2], sort = False)

# Remove nan values
df.drop(columns='second_pass_reading_time', inplace = True)
df.dropna(subset = ['first_fix_dur'], inplace = True)

# Word length - freq correlation
df_unique = df.drop_duplicates('word')
corr_unique = df_unique[['word_length', 'freq_pM']].corr(method = 'spearman').loc['word_length', 'freq_pM']
print('Word length and frequency Spearman r: {:.2f}'.format(corr_unique))
plt.figure()
sns.regplot(x = 'word_length', y = 'freq_pM', data = df_unique)

# ET feature correlation
for gp in gp_list:
    for sp in sp_list:
        if (gp == 'control_2') & (sp != 2):
            break
        corr = np.empty((len(var_list), len(var_list), 0))
        for subj in set(df[df['group'] == gp].subj_code):
            if gp != 'control_2':
                filters = (df['group'] == gp) & (df['subj_code'] == subj) & (df['spacing'] == sp)
                filters_all = (df['group'] == gp) & (df['spacing'] == sp)
            else:
                filters = (df['group'] == gp) & (df['subj_code'] == subj)
                filters_all = (df['group'] == gp)
            corr_df = df[filters][var_list].corr(method = 'spearman')
            corr = np.concatenate([corr, np.expand_dims(corr_df, axis = 2)], axis = 2)
        mean_corr = corr.mean(axis = 2)
        df_mean_corr = pd.DataFrame(mean_corr, index = corr_df.index, columns = corr_df.columns)
        
        plt.figure(figsize=(13,12))
        plt.suptitle('{}  SP{}  Mean of subject level Spearman r'.format(gp, sp))
        ax = sns.heatmap(df_mean_corr, vmin=-1, vmax=1, cmap='RdBu_r', annot=True, fmt=".2f", annot_kws={"fontsize":8})
        
        # Bugfix for matplotlib 3.1.1
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        
        plt.figure(figsize=(16,12))
        plt.suptitle('{}  SP{}  Histograms'.format(gp, sp))
        for i, v in enumerate(var_list):
            row_num = int(np.sqrt(len(var_list)))
            col_num = len(var_list)/row_num
            plt.subplot(row_num, col_num, i+1)
            data = df[filters_all][v].dropna()
            sns.distplot(data, kde = False)
