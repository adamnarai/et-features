import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Params
gp_list = ['control', 'dyslexic', 'control_2']
word_length_min = 5
word_length_max = 13
sp = 5
res_path_1 = 'D:/et_features/data/dys_study/word_info/word_info_all.csv'
res_path_2 = 'D:/et_features/data/dys_contr_2_study/word_info/word_info_all.csv'
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

# Get groups
df = df[df['group'].isin(['control', 'dyslexic'])]

# Filter spacing
df = df[df['spacing'] == sp]

# Get word length range
df = df[(df['word_length'] >= word_length_min) & (df['word_length'] <= word_length_max)]
df['refix_prob'] = (df['fix_num'] > 1).astype(float)


plt.figure()
# FFD
plt.subplot(1,3,1)
sns.pointplot(x='word_length', y='first_fix_dur', hue='group', data=df, dodge=True, ci=95)

# GD
plt.subplot(1,3,2)
sns.pointplot(x='word_length', y='total_reading_time', hue='group', data=df, dodge=True, ci=95)

# REFIX
plt.subplot(1,3,3)
sns.pointplot(x='word_length', y='refix_prob', hue='group', data=df, dodge=True, ci=95)


# Freq measures
plt.figure()
df_unique = df.drop_duplicates('word')
plt.subplot(1,2,1)
sns.pointplot(x='word_length', y='freq_pM', hue='group', data=df, dodge=True, ci=95)
plt.subplot(1,2,2)
sns.pointplot(x='word_length', y='stem_freq_pM', hue='group', data=df, dodge=True, ci=95)
