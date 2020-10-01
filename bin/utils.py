import os
import yaml
import pandas as pd
from na_py_tools.defaults import RESULTS_DIR, SETTINGS_DIR

def load_params(path=SETTINGS_DIR + '/params.yaml'):
    with open(path) as file:
        p = yaml.full_load(file)
    return p
    
# TODO: Currently loads everything and filters, should not load unnecessary data
def import_et_behav_data(p, experiments=['et', '3dmh', 'wais', 'perf', 'vsp', 'word_info', 'proofreading', 'sentence_verification'],
                         et_feature_list='meas_list'):
    # Read data
    meas_list = dict()
    # ET
    data_et = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
    meas_list['et'] = p['params']['et'][et_feature_list]

    # 3DMH
    data_3dmh = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'all', '3dmh', '3dmh.pkl'))
    meas_list_3dmh = sum(list(p['params']['3dmh']['meas_list'].values()), [])

    # WAIS
    data_wais = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'wais', 'wais.pkl'))
    meas_list['wais'] = p['params']['wais']['meas_list']
    data_wais['study'] = 'dys'

    # VSP (est + sum)
    data_vsp = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'vsp', 'vsp.pkl'))
    data_vsp.columns = data_vsp.columns.astype(str)
    meas_list['vsp'] = p['params']['vsp']['meas_list_all']

    # Reading perf
    data_perf = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_perf.pkl'))
    meas_list['perf'] = ['perf']

    # Word info
    data_word_info = pd.read_pickle(
        os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'word_info.pkl'))
    meas_list['word_info'] = p['params']['et']['word_info_list']
    
    # Merge all data
    data = data_et.merge(data_3dmh, how='outer', on=['study', 'group', 'subj_id'])
    data = data.merge(data_wais, how='outer', on=['study', 'group', 'subj_id'])
    data = data.merge(data_vsp, how='outer', on=['group', 'condition', 'spacing_size', 'subj_id'])
    data = data.merge(data_perf, how='outer', on=['study', 'group', 'condition', 'spacing_size', 'subj_id'])
    data = data.merge(data_word_info, how='outer', on=['study', 'group', 'condition', 'subj_id'])
    
    # Proofreading
    try:
        data_proof = pd.read_pickle(
            os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'proofreading', 'proofreading.pkl'))
        data_proof.columns = ['_'.join([c, 'proof']) if c in p['params']['proofreading'][et_feature_list] else c for c in data_proof.columns]
        meas_list['proofreading'] = ['_'.join([s, 'proof']) for s in p['params']['proofreading'][et_feature_list]]
        data = data.merge(data_proof, how='outer', on=['study', 'group', 'subj_id'])
    except:
        print('Could not load proofreading data.')
    
    # Sentence verification
    try:
        data_verif = pd.read_pickle(
            os.path.join(RESULTS_DIR, 'df_data', 'dys_study', 'sentence_verification', 'sentence_verification.pkl'))
        data_verif.columns = ['_'.join([c, 'verif']) if c in p['params']['sentence_verification'][et_feature_list] else c for c in data_verif.columns]
        meas_list['sentence_verification'] = ['_'.join([s, 'verif']) for s in p['params']['sentence_verification'][et_feature_list]]
        data = data.merge(data_verif, how='outer', on=['study', 'group', 'subj_id'])
    except:
        print('Could not load sentence verification data.')

    # Merge all measure varname
    plot_meas_list = sum([meas_list[exp] for exp in experiments], [])
    
    return (data, plot_meas_list)
