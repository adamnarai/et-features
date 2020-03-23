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
    
    # Concatenate conditions
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