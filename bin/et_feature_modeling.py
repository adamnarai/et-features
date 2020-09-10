"""
ET feature modeling

@author: Adam Narai
@email: narai.adam@gmail.com
@institute: Brain Imaging Centre, RCNS
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
import timeit
from na_py_tools.defaults import RESULTS_DIR
from utils import load_params, import_et_behav_data
import logging
from datetime import datetime
import warnings
from tqdm import tqdm
import pickle

# Load params YAML
p = load_params()
results_dir = os.path.join(RESULTS_DIR, 'regression')
os.makedirs(results_dir, exist_ok=True)
now = datetime.now()
        
# Logging
logging.captureWarnings(True)
log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler('{}/{}{}.log'.format(results_dir, os.path.basename(__file__).replace('.py', '_'), now.strftime('%Y%d%m_%H%M%S')))
fileHandler.setFormatter(log_formatter)
logger.addHandler(fileHandler)
logger.propagate = False

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
logger.addHandler(consoleHandler)

# Params
experiment = 'proofreading'            # et, proofreading, sentence_verification
study_list = list(p['studies'].keys())
gp_list = ['dyslexic', 'control']
cond_list = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'MS', 'NS', 'DS']   # SP1, SP2, SP3, SP4, SP5, MS, NS, DS
dep_var = 'Med_rspeed_wnum'
save_pdf = True
complex_models = True
simple_models = True
et_feature_list = ['meas_list']#['meas_list_min_2', 'meas_list_min']#, 'meas_list']

# Params
experiments = ['proofreading']            # et, proofreading, sentence_verification
study_list = ['dys']
gp_list = ['dyslexic', 'control']
cond_list = ['SP2']   # SP1, SP2, SP3, SP4, SP5, MS, NS, DS
dep_var = 'Med_rspeed_wnum_proof'
save_pdf = True
complex_models = False
simple_models = True
et_feature_list = ['meas_list']

# Model params
cv_in_perm = False
fold_num_list = [10]
p_perm_num = 1000

# ENET params
seed = 42
l1_ratio_N = 20
l1_ratio_min_exp = -2
enet_alpha_N = 20
l1_ratio_list = (1 + 10**l1_ratio_min_exp) - np.logspace(0, l1_ratio_min_exp, l1_ratio_N)
enet_max_iter = 10000

# Ridge params
ridge_alpha_N = 20
alpha_list = np.logspace(-2, 2, ridge_alpha_N)

def create_pdf(exp, study, gp, cond, et_features):
    sub_dir = f'/{study}/{gp}/{cond}/{et_features}'
    os.makedirs(results_dir + sub_dir, exist_ok=True)
    meta_str = f'fold{fold_num}_perm{p_perm_num}_enet_L1N{l1_ratio_N}_alphN{enet_alpha_N}_ridge_alphN{ridge_alpha_N}_cvinperm{int(cv_in_perm)}'
    out_path = results_dir + sub_dir + '/{}_regr_{}'.format(exp.upper(), meta_str)
    pdf = PdfPages(out_path + '.pdf')
    return pdf, out_path
            
def get_model_params(model, X, y, X_names, fold_num, seed):
    params = dict()
    params['coef'] = pd.DataFrame(data=model.coef_, index=X_names)
    params['raw_coef'] = model.coef_
    try:
        params['alpha'] = model.alpha_
        params['l1_ratio'] = model.l1_ratio_
    except:
        pass
    params['mse'] = mean_squared_error(y, model.predict(X))
    params['r2'] = r2_score(y, model.predict(X))
    return params


def run_regressions(df, X_names, y_name, fold_num, seed, l1_ratio_list, alpha_list, study, gp, cond, 
                    cv_in_perm=False, binary_vars=[]):
    logger.info('Modeling {} in study: {}, group: {}, cond: {}'.format(y_name, study, gp, cond))
    
    # Dict of all modeling data
    model_data = dict()
    model_data['ols'] = dict()
    model_data['enet'] = dict()
    model_data['ridge'] = dict()
    model_data['vif'] = dict()
    
    # Get data
    X = df[X_names]
    y = df[y_name]
    
    # Standardize data
    scaler = StandardScaler()
    if not binary_vars:
        X_st = scaler.fit_transform(X)
        X_st = pd.DataFrame(X_st, index=X.index, columns=X.columns)
    else:
        # Leave binary variables as is and divide all others with 2 SD
        cols_to_standard = X.columns.difference(binary_vars)
        X_st = X.copy()
        X_st[cols_to_standard] = scaler.fit_transform(X_st[cols_to_standard]) / 2
    
    # OLS fit fit
    logger.info('OLS fit')
    ols_model = LinearRegression().fit(X_st, y)
    ols_params = get_model_params(ols_model, X_st, y, X_names, fold_num, seed)
    logger.info('DONE.')
    model_data['ols']['model'] = ols_model
    model_data['ols']['params'] = ols_params
    model_data['ols']['beta'] = ols_params['coef']
    
    # ENET hyperparameter (alpha, L1_ratio) selection and fit in one step
    logger.info('ENET hyperparameter selection using {}-fold cross-validation...'.format(fold_num))
    enet_model = ElasticNetCV(cv=fold_num, random_state=seed, l1_ratio=l1_ratio_list, 
                              n_alphas=enet_alpha_N, max_iter=enet_max_iter, n_jobs=6)
    enet_params = get_model_params(enet_model.fit(X_st, y), X_st, y, X_names, fold_num, seed)
    logger.info('Best hyperparameters: alpha = {:.4f}  L1_ratio = {:.2f}'.format(enet_model.alpha_, enet_model.l1_ratio_))
    model_data['enet']['model'] = enet_model
    model_data['enet']['params'] = enet_params
    model_data['enet']['beta'] = enet_params['coef']
    
    # ENET permuted p values
    logger.info('ENET p calculation using {} permutations...'.format(p_perm_num))
    enet_model_simple = ElasticNet(alpha=enet_model.alpha_, l1_ratio=enet_model.l1_ratio_, max_iter=enet_max_iter)
    model_data['enet']['simple_model'] = enet_model_simple
    
    # Permutations
    np.random.seed(seed)
    model_data['enet']['seed'] = seed
    perm_betas = np.empty([X_st.shape[1], p_perm_num])
    perm_betas[:] = np.nan
    for i in range(p_perm_num):
        y_perm = np.random.permutation(y)
        if not cv_in_perm:
            res = enet_model_simple.fit(X_st, y_perm)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                res = enet_model.fit(X_st, y_perm)
            logger.info('Perm {:4} hyperparameters: alpha = {:.4f}  L1_ratio = {:.2f}'
                        .format(i+1, enet_model.alpha_, enet_model.l1_ratio_))
        perm_betas[:,i] = res.coef_

    model_data['enet']['perm_betas'] = perm_betas
    
    # Get p values
    ge = np.sum(perm_betas <= enet_params['raw_coef'][:, None], axis=1)
    le = np.sum(perm_betas >= enet_params['raw_coef'][:, None], axis=1)
    enet_p = np.minimum((2*np.minimum(ge,le)+1)/(p_perm_num+1), np.ones(ge.shape))    
    model_data['enet']['pval'] = enet_p
    logger.info('DONE.')
    
    
    # Ridge hyperparameter (alpha) selection and fit in one step
    logger.info('Ridge hyperparameter selection using {}-fold cross-validation...'.format(fold_num))
    ridge_model = RidgeCV(cv=fold_num, alphas=alpha_list)
    ridge_params = get_model_params(ridge_model.fit(X_st, y), X_st, y, X_names, fold_num, seed)
    logger.info('Best hyperparameters: alpha = {:.4f}'.format(ridge_model.alpha_))
    model_data['ridge']['model'] = ridge_model
    model_data['ridge']['params'] = ridge_params
    model_data['ridge']['beta'] = ridge_params['coef']
    
    # Ridge permuted p values
    logger.info('Ridge p calculation using {} permutations...'.format(p_perm_num))
    ridge_model_simple = Ridge(alpha=ridge_model.alpha_)
    model_data['ridge']['simple_model'] = ridge_model_simple
    
    # Permutations
    np.random.seed(seed)
    model_data['ridge']['seed'] = seed
    perm_betas = np.empty([X_st.shape[1], p_perm_num])
    perm_betas[:] = np.nan
    for i in tqdm(range(p_perm_num)):
        y_perm = np.random.permutation(y)
        if not cv_in_perm:
            res = ridge_model_simple.fit(X_st, y_perm)
        else:
            res = ridge_model.fit(X_st, y_perm)
            logger.info('Perm {:4} hyperparameters: alpha = {:.4f}'.format(i+1, ridge_model.alpha_))
        perm_betas[:,i] = res.coef_
        
    model_data['ridge']['perm_betas'] = perm_betas
    
    # Get p values
    ge = np.sum(perm_betas <= ridge_params['raw_coef'][:, None], axis=1)
    le = np.sum(perm_betas >= ridge_params['raw_coef'][:, None], axis=1)
    ridge_p = (2*np.minimum(ge,le)+1)/(p_perm_num+1)    
    model_data['ridge']['pval'] = ridge_p
    logger.info('DONE.')
    
    # VIF
    model_data['vif'] = [variance_inflation_factor(X_st.values, i) for i in range(X_st.shape[1])]
    
    # Get coeff df
    betas = pd.DataFrame(index = X_names)
    betas['OLS'] = ols_params['coef']
    betas['Ridge'] = ridge_params['coef']
    betas['ENET'] = enet_params['coef']
    
    # Get pval df
    pval = pd.DataFrame(index = X_names)
    pval['ridge_p'] = ridge_p
    pval['enet_p'] = enet_p
    
    # Figure
    fig = plt.figure(figsize=(15,10))
    gs = GridSpec(1, 10, figure=fig)
    
    # Coef heatmap
    fig.add_subplot(gs[0, :3])
    ax = sns.heatmap(betas, cmap = 'RdBu_r', vmin = -1, vmax = 1, annot = True, fmt='.2f')
    ax.set_title('Study: {}  Group: {}  Condition: {} '
                 '\nENET: alpha = {:.4f}  L1 ratio = {:.2f}\nRidge: alpha = {:.4f}'
                  .format(study, gp, cond, enet_params['alpha'], enet_params['l1_ratio'], ridge_params['alpha']), pad=20)
    
    # Pval heatmap
    fig.add_subplot(gs[0, 3:5])
    ax = sns.heatmap(pval, cmap = 'afmhot', vmin = 0, vmax = 0.2, annot = True, fmt='.3f')
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(f'Permuted p values\n(N = {p_perm_num})\nCV in each perm: {cv_in_perm}')
    fig.subplots_adjust(top=0.90, bottom=0.1, wspace=0.4, left=0.1, right=0.9)
    
    # Text subplot
    fig.add_subplot(gs[0, 5:])
    plt.axis('off')
    
    # Prediction errors
    error_txt = (f'Prediction errors (all training data)):'
            f'\n    OLS:   MSE={ols_params["mse"]:.4f}  R2={ols_params["r2"]:.4f}'
            f'\n    Ridge: MSE={ridge_params["mse"]:.4f}  R2={ridge_params["r2"]:.4f}'
            f'\n    ENET:  MSE={enet_params["mse"]:.4f}  R2={enet_params["r2"]:.4f}')
    
    plt.text(0.1, 0.95, error_txt, fontdict={'family':'monospace', 'fontsize':10})
    
    # Stat table
    sm.OLS(y, sm.add_constant(X_st))
    res = sm.OLS(y, sm.add_constant(X_st)).fit()
    plt.text(0.1, 0, res.summary(), fontdict={'family':'monospace', 'fontsize':8})
    model_data['ols']['stats'] = res
    return model_data
    

for et_features, fold_num in product(et_feature_list, fold_num_list):
    start_time = timeit.default_timer()
    # Import data
    data, meas_list = import_et_behav_data(p, experiments=[experiment], et_feature_list=et_features)
    
    # Feature list
    X_names = meas_list.copy()
    X_names.remove(dep_var)
    y_name = dep_var
        
    if simple_models:
        # Modeling each study/group/condition
        for study, gp, cond in product(study_list, gp_list, cond_list):
            df = data[(data['study'] == study) & (data['condition'] == cond) 
                             & (data['group'] == gp)].loc[:, meas_list]
            if df.empty:
                continue
            if save_pdf:
                pdf, out_path = create_pdf(experiment, study, gp, cond, et_features)

            # Run ENET and Ridge regressions
            model_data = run_regressions(df, X_names, y_name, fold_num, seed, 
                            l1_ratio_list, alpha_list, study, gp, cond, cv_in_perm=cv_in_perm)
            
            with open(out_path + '.pkl', 'wb') as file:
                pickle.dump(model_data, file)
            if save_pdf:
                pdf.savefig()
                plt.savefig(out_path + '.png')
                plt.close()
                pdf.close()
        
    if complex_models:
        # Models including spacing
        for study, gp in product(['letter_spacing', 'dys'], gp_list):
            cond = 'ALL'
            df = data[(data['study'] == study) & (data['group'] == gp)].loc[:, ['spacing_size'] + meas_list]
            if df.empty:
                continue
            if save_pdf:
                pdf, out_path = create_pdf(experiment, study, gp, cond, et_features)
            
            # Run ENET and Ridge regression
            model_data = run_regressions(df, ['spacing_size'] + X_names, y_name, fold_num, seed, 
                            l1_ratio_list, alpha_list, study, gp, cond, cv_in_perm=cv_in_perm)
            
            with open(out_path + '.pkl', 'wb') as file:
                pickle.dump(model_data, file)
            if save_pdf:
                pdf.savefig()
                plt.savefig(out_path + '.png')
                plt.close()
                pdf.close()
                        
        # Models including group
        for study, cond in product(['dys'], cond_list):
            gp = 'ALL'
            df = data[(data['study'] == study) & (data['condition'] == cond)].loc[:, ['group'] + meas_list]
            if df.empty:
                continue
            df['group'] = pd.get_dummies(df['group'])['dyslexic']
            for model_type in ['standard', 'interactions']:
                for binary_vars in [['group'], []]:
                    if binary_vars:
                        binary_vars_str = '_nobinstd'
                    else:
                        binary_vars_str = ''
                    if save_pdf:
                        pdf, out_path = create_pdf(experiment, study, gp, cond, et_features + '/' + model_type + binary_vars_str)
                    if model_type == 'standard':
                        # Run standard regression
                        model_data = run_regressions(df, ['group'] + X_names, y_name, fold_num, seed, 
                                        l1_ratio_list, alpha_list, model_type + ': ' + study, gp, cond, 
                                        cv_in_perm=cv_in_perm, binary_vars=binary_vars)
                    elif model_type == 'interactions':
                        # Adding interaction terms with group
                        inter_X_names = []
                        for v in X_names:
                            inter_name = 'group*' + v
                            inter_X_names.append(inter_name)
                            df[inter_name] = df['group'] * df[v]
                            
                        # Run regression with interactions
                        model_data = run_regressions(df, ['group'] + X_names + inter_X_names, y_name, fold_num, seed, 
                                        l1_ratio_list, alpha_list, '(' + model_type + ') ' + study, gp, cond, 
                                        cv_in_perm=cv_in_perm, binary_vars=binary_vars)
                    
                    with open(out_path + '.pkl', 'wb') as file:
                        pickle.dump(model_data, file)
                    if save_pdf:
                        pdf.savefig()
                        plt.savefig(out_path + '.png')
                        plt.close()
                        pdf.close()
                    
        # Models including group and spacing
        for study in ['dys']:
            gp = 'ALL'
            cond = 'ALL'
            df = data[data['study'] == study].loc[:, ['group'] + ['spacing_size'] + meas_list]
            if df.empty:
                continue
            df['group'] = pd.get_dummies(df['group'])['dyslexic']
            
            for model_type in ['standard', 'interactions']:
                for binary_vars in [['group'], []]:
                    if binary_vars:
                        binary_vars_str = '_nobinstd'
                    else:
                        binary_vars_str = ''
                    if save_pdf:
                        pdf, out_path = create_pdf(experiment, study, gp, cond, et_features + '/' + model_type + binary_vars_str)
                    if model_type == 'standard':
                        # Run standard regression
                        model_data = run_regressions(df, ['group'] + ['spacing_size'] + X_names, y_name, fold_num, seed, 
                                        l1_ratio_list, alpha_list, model_type + ': ' + study, gp, cond, 
                                        cv_in_perm=cv_in_perm, binary_vars=binary_vars)
                    elif model_type == 'interactions':
                        # Adding interaction terms with group
                        inter_X_names = []
                        for v in X_names:
                            inter_name = 'group*' + v
                            inter_X_names.append(inter_name)
                            df[inter_name] = df['group'] * df[v]
                            
                        # Run regression with interactions
                        model_data = run_regressions(df, ['group'] + ['spacing_size'] + X_names + inter_X_names, y_name, fold_num, seed, 
                                        l1_ratio_list, alpha_list, '(' + model_type + ') ' + study, gp, cond, 
                                        cv_in_perm=cv_in_perm, binary_vars=binary_vars)
            
                    with open(out_path + '.pkl', 'wb') as file:
                        pickle.dump(model_data, file)
                    if save_pdf:
                        pdf.savefig()
                        plt.savefig(out_path + '.png')
                        plt.close()
                        pdf.close() 
    
    logger.info('Running time: {:d} min {:d} s'.format(
        round((timeit.default_timer() - start_time)//60), 
        round((timeit.default_timer() - start_time)%60)))
