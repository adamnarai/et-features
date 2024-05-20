import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Params
gp_list = ['control', 'dyslexic']
word_length_min = 5
word_length_max = 13
sp = 2
res_path = 'D:/et_features/data/dys_study/fix_nfo/fix_nfo_all.csv'

# Load data
df = pd.read_csv(res_path, dtype = {'subj_code': str}, encoding = 'unicode_escape')

# Remove nan values
df.dropna(subset = ['currFixDur'], inplace = True)
df = df[df['wordFound'] == 1]

# Filter spacing
df = df[df['spacing'] == sp]

# Get word length range
df = df[(df['length'] >= word_length_min) & (df['length'] <= word_length_max)]


plt.figure()
plt.subplot(1,2,1)
sns.pointplot(x='length', y='currFixDur', hue='first_fix', data=df[df['group'] == 'control'], dodge=True, ci=95)
plt.title('control')

plt.subplot(1,2,2)
sns.pointplot(x='length', y='currFixDur', hue='first_fix', data=df[df['group'] == 'dyslexic'], dodge=True, ci=95)
plt.title('dyslexic')

