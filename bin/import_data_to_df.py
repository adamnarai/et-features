"""
Importing ET and behav. data to pandas df and saving as pkl

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""
import os
import pandas as pd
import yaml
from na_py_tools.defaults import ROOT_DIR, RESULTS_DIR, SETTINGS_DIR

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# %% ET features
df_all = pd.DataFrame()
for study in list(p['studies'].keys()):
    data_path = ROOT_DIR + p['studies'][study]['experiments']['et']['features_path']
    cond = p['studies'][study]['experiments']['et']['conditions']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    spacing_sizes = p['studies'][study]['experiments']['et']['spacing_size']
    
    # Get data
    df = pd.read_excel(data_path, sheet_name=cond, index_col='Measures/Subjects')
    
    # Filter subjects in each condition
    for key, value in df.items():
        df[key] = value.transpose()
        df[key] = df[key].loc[sum([subjects[gp] for gp in list(groups.values())], []), :]
        
    # Concatenate conditions
    df = pd.concat((v.assign(condition=k) for k, v in df.items()))
    df = df.rename_axis('subj_id').reset_index()
    
    # Set spacing size
    df['spacing_size'] = df.condition.replace(spacing_sizes)
    
    # Set group name
    if 'Group' in df:
        df = df.rename(columns={'Group': 'group'})
    elif len(groups) == 1:
        df['group'] = 1
    else:
        raise('Multiple groups without ''Group'' variable.')
    df['group'] = df['group'].replace(groups)
    
    # Save df in pkl and csv format
    out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'et')
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(out_path + '/et_features.pkl')
    df.to_csv(out_path + '/et_features.csv')
    
    # Concatenate conditions
    df['study'] = study
    df_all = pd.concat((df_all, df), sort=False)
    
# Save df_all in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', 'all', 'et')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/et_features.pkl')
df_all.to_csv(out_path + '/et_features.csv')
    
# %% 3DMH
df_all = pd.DataFrame()
for study in ['dys', 'dys_contr_2']:
    data_path = ROOT_DIR + p['studies'][study]['experiments']['3dmh']['data_path']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    
    if study == 'dys':  # Dyslexic study
        # Get data
        df = pd.read_csv(data_path, sep=';', encoding='unicode_escape')
        
        # Rename variables
        df = df.rename(columns={'subid': 'subj_id'})
        
        # Filter subjects
        df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
    elif study == 'dys_contr_2':    # Dyslexic control 2study
        # Get data
        df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')
        info_path = ROOT_DIR + p['studies'][study]['experiments']['3dmh']['info_path']
        df_info = pd.read_csv(info_path, sep=',', encoding='unicode_escape')
        df_info = df_info.rename(columns={'Szkod': 'SzKod'})
        df = df.merge(df_info, on='SzKod', how='inner')
        
        # Rename variables
        df = df.rename(columns={'Nev': 'subj_id'})
        
        # Filter subjects
        df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
        df['subj_id'] = df['subj_id'].apply('{:06d}'.format)
    
    # Set group
    df['group'] = None
    for gp in list(groups.values()):
        df.loc[df['subj_id'].isin(subjects[gp]), 'group'] = gp
    
    # Save df in pkl and csv format
    out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], '3dmh')
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(out_path + '/3dmh.pkl')
    df.to_csv(out_path + '/3dmh.csv')
    
    # Concatenate studies
    df['study'] = study
    df_all = pd.concat((df_all, df), sort=False)
    
# Save df_all in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/3dmh.pkl')
df_all.to_csv(out_path + '/3dmh.csv')
    

# %% WAIS
# Dyslexic study
study = 'dys'
data_path = ROOT_DIR + p['studies'][study]['experiments']['wais']['data_path']
subjects = p['studies'][study]['subjects']
groups = p['studies'][study]['groups']

# Get data
df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')

# Rename variables
df = df.rename(columns={'subject': 'subj_id'})

# Filter subjects
df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]

# Save df in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'wais')
os.makedirs(out_path, exist_ok=True)
df.to_pickle(out_path + '/wais.pkl')
df.to_csv(out_path + '/wais.csv')

# %% VSP
# Dyslexic study
study = 'dys'
df_all = pd.DataFrame()
for var_type in ['ecc', 'est', 'sum']:
    data_path = ROOT_DIR + p['studies'][study]['experiments']['vsp'][f'{var_type}_data_path']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    spacing_sizes = p['studies'][study]['experiments']['vsp']['spacing_size']
    
    # Get data
    df = pd.read_csv(data_path, sep=',', encoding='unicode_escape', index_col=0)
    
    # Rename variables
    df = df.rename(columns={'subid': 'subj_id', 'perf_sum': 'perf', 
                            'value': 'perf', 'eccentricity': 'param',
                            'spacing': 'spacing_size'})
    
    df['spacing_size'] = df['spacing_size'].round(3)
    df['condition'] = df['spacing_size'].replace({v: k for k, v in spacing_sizes.items()})
    
    # Filter subjects
    df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
    
    # Long to wide format
    df = pd.pivot_table(df, values='perf', index=['group', 'condition', 'subj_id', 'spacing_size'], columns=['param']).reset_index()
    
    # Concat variable types
    if df_all.empty:
        df_all = df
    else:
        df_all = df_all.merge(df, how='outer', on=['group', 'condition', 'subj_id', 'spacing_size'])
    
# Save df in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'vsp')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/vsp.pkl')
df_all.to_csv(out_path + '/vsp.csv')
    
# %% Reading perf
df_all = pd.DataFrame()
for study in ['dys', 'dys_contr_2']:
    data_path = ROOT_DIR + p['studies'][study]['experiments']['et']['perf_data_path']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    spacing_sizes = p['studies'][study]['experiments']['et']['spacing_size']
    
    if study == 'dys':  # Dyslexic study
        # Get data
        df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')
        
        # Rename variables
        df = df.rename(columns={'id': 'subj_id', 'gp': 'group', 'sp': 'condition'})
        
        # Set condition
        df['condition'] = 'SP' + df.condition.astype(str)
        df['spacing_size'] = df.condition.replace(spacing_sizes)
        
        # Filter subjects
        df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
    elif study == 'dys_contr_2':    # Dyslexic control 2 study
        # Get data
        df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')
        
        # Rename variables
        df = df.rename(columns={'id': 'subj_id'})
        df['group'] = 'control'
        df['condition'] = 'SP2'
        df['spacing_size'] = 1
        
        # Filter subjects
        df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
        df['subj_id'] = df['subj_id'].apply('{:06d}'.format)
    
    # Save df in pkl and csv format
    out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'et')
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(out_path + '/et_perf.pkl')
    df.to_csv(out_path + '/et_perf.csv')
    
    # Concatenate studies
    df['study'] = study
    df_all = pd.concat((df_all, df), sort=False)
    
# Save df_all in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', 'all', 'et')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/et_perf.pkl')
df_all.to_csv(out_path + '/et_perf.csv')

# %% Word info
df_all = pd.DataFrame()
for study in ['dys', 'dys_contr_2']:
    data_path = ROOT_DIR + p['studies'][study]['experiments']['et']['word_info_path']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    spacing_sizes = p['studies'][study]['experiments']['et']['spacing_size']
    
    # Get data
    df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')
    
    # Rename variables
    df = df.rename(columns={'subj_code': 'subj_id', 'spacing': 'condition'})
    
    if study == 'dys_contr_2':    # Dyslexic control 2 study
        df['subj_id'] = df['subj_id'].apply('{:06d}'.format)
        df['group'] = 'control'
    
    # Set condition
    df['condition'] = 'SP' + df.condition.astype(str)
    df['spacing_size'] = df.condition.replace(spacing_sizes)
    
    # Filter subjects
    df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
    
    # Get mean
    df = df.groupby(['group', 'condition', 'subj_id']).mean()[p['params']['et']['word_info_list']]  
    
    # Save df in pkl and csv format
    out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'et')
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(out_path + '/word_info.pkl')
    df.to_csv(out_path + '/word_info.csv')
    
    # Concatenate studies
    df['study'] = study
    df_all = pd.concat((df_all, df), sort=False)
    
# Save df_all in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', 'all', 'et')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/word_info.pkl')
df_all.to_csv(out_path + '/word_info.csv')

# %% EEG
df_all = pd.DataFrame()
for study in ['dys', 'dys_contr_2']:
    data_path = ROOT_DIR + p['studies'][study]['experiments']['eeg']['data_path']
    subjects = p['studies'][study]['subjects']
    groups = p['studies'][study]['groups']
    
    # Get data
    df = pd.read_csv(data_path, sep=',', encoding='unicode_escape')
    df['condition'] = 'SP2'
    
    if study == 'dys_contr_2':
        df['subj_id'] = df['subj_id'].apply('{:06d}'.format)
        df['group'] = 'control'
    
    # Filter subjects
    df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
    
    # Save df in pkl and csv format
    out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'eeg')
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(out_path + '/eeg_peak_data.pkl')
    df.to_csv(out_path + '/eeg_peak_data.csv')
    
    # Concatenate studies
    df['study'] = study
    df_all = pd.concat((df_all, df), sort=False)
    
# Save df_all in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', 'all', 'eeg')
os.makedirs(out_path, exist_ok=True)
df_all.to_pickle(out_path + '/eeg_peak_data.pkl')
df_all.to_csv(out_path + '/eeg_peak_data.csv')
    
# %% Proofreading (ET features)
study = 'dys'
subjects = p['studies'][study]['subjects']
groups = p['studies'][study]['groups']
data_path = ROOT_DIR + p['studies'][study]['experiments']['proofreading']['data_path']

# Get data
df = pd.read_csv(data_path, index_col='subject').reset_index().rename(columns={'subject':'subj_id', 'label':'group'})
# df.columns = [str(col) + '_proof' if col not in {'study', 'group','subj_id'} else col for col in df.columns]
df['group'].replace({1:'dyslexic', 2:'control'}, inplace=True)
df['study'] = study

# Filter subjects
df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]
  
# Save df in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'proofreading')
os.makedirs(out_path, exist_ok=True)
df.to_pickle(out_path + '/proofreading.pkl')
df.to_csv(out_path + '/proofreading.csv')

# %% Sentence verification (ET features)
study = 'dys'
subjects = p['studies'][study]['subjects']
groups = p['studies'][study]['groups']
data_path = ROOT_DIR + p['studies'][study]['experiments']['sentence_verification']['data_path']

# Get data
df = pd.read_csv(data_path, index_col='subject').reset_index().rename(columns={'subject':'subj_id', 'label':'group'})
# df.columns = [str(col) + '_verif' if col not in {'study', 'group','subj_id'} else col for col in df.columns]
df['group'].replace({1:'dyslexic', 2:'control'}, inplace=True)
df['study'] = study

# Filter subjects
df = df[df['subj_id'].isin(sum([subjects[gp] for gp in list(groups.values())], []))]

  
# Save df in pkl and csv format
out_path = os.path.join(RESULTS_DIR, 'df_data', p['studies'][study]['dir'], 'sentence_verification')
os.makedirs(out_path, exist_ok=True)
df.to_pickle(out_path + '/sentence_verification.pkl')
df.to_csv(out_path + '/sentence_verification.csv')
