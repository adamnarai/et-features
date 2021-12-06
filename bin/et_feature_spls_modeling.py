import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNetCV
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as datetime

# R API
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
cv_spls = robjects.r('spls::cv.spls')

import time
seed = int(time.time())
start_time = time.time()
print('Script started at {}'.format(datetime.now().strftime("%Y-%d-%m %H:%M:%S")))

r_spls = importr('spls')
r_base = importr('base')

# Read data
results_path = 'D:/temp';
df = pd.read_pickle(results_path+'/df_et.pkl')

# Parameters
gp_list = ['dyslexic']       # Group filter
sp_list = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']   # Spacing filter
sp_list = ['SP2']   # Spacing filter
y_name = 'Med_rspeed_wnum'  # Dependent varable

# X_names = [
#     # "group",
#     # "spacing",
#     "Med_fixdur",
#     "Med_saccamp",
#     "Perc_fsacc",
#     "Perc_sacc_gliss",
#     "Perc_fsacc_fgliss",
#     "Perc_bsacc_fgliss",
#     "Med_fsaccamp",
#     "Med_bsaccamp",
#     "Perc_fgliss",
#     "Perc_fsacc_gliss",
#     "Perc_bsacc_gliss",
#     "Freq_fsacc",
#     "Freq_bsacc",
#     "Mean_meanfixvel",
#     "Mean_stdevfixvel",
#     "Med_saccpdetth",
#     "Med_saccdur",
#     "Med_fsaccdur",
#     "Med_bsaccdur",
#     "Med_glissdur",
#     "Med_fglissdur",
#     "Med_bglissdur",
#     "Med_glissamp",
#     "Med_fglissamp",
#     "Med_bglissamp",
#     "Med_tsnum_wnum",
#     "Med_fsnum_wnum",
#     "Med_bsnum_wnum",
#     "Med_saccpvel",
#     "Perc_lvel_gliss",
#     "Med_glisspvel",
#     "Med_lvglisspvel",
#     "Med_hvglisspvel"
#     ]

X_names = [
#    "spacing",
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

# Filters (group and spacing)
df = df[df.Group.isin(gp_list)]
df = df[df.spacing.isin(sp_list)]

# Change spacing variable to continuous
df['spacing'] = df['spacing'].replace({'SP1':0.707, 'SP2':1, 'SP3':1.3, 'SP4':1.6, 'SP5':1.9})

# Change group names to dummy coding
df = df.rename(columns={'Group': 'group'})
df['group'] = df['group'].replace({'control': 0, 'dyslexic': 1})

# Get data and standardize
X = df[X_names]
y = df[y_name]
X = X.apply(stats.zscore)
data = pd.concat([X, y], axis=1)

# SPLS hyperparameter (eta, K) selection
K_list = np.array(list(range(1, round(min(len(X.columns), len(X.index))*4/5))))
eta_list = np.arange(0, 100)/100
aggr_mspemat = np.empty((len(eta_list), len(K_list), 0))
hypar = pd.DataFrame()
fold_num = 6
rep_num = 20
print('\nSPLS hyperparameter selection on {} group(s) and {} condition(s)...'
      .format(', '.join(gp_list), ', '.join(sp_list)))
print('Using {}-fold cross-validation with {} repetition and mean MSPE map'
      .format(fold_num, rep_num))
for rep in range(rep_num):
    print('SPLS cross-validation repetition {}/{}'.format(rep+1, rep_num))
    seed = int(time.time())
    r_base.set_seed(seed)
    cv_par = cv_spls(X.to_numpy(), y.to_numpy(), fold = fold_num, K = K_list, eta = eta_list)
    cv_mspemat = np.array(cv_par.rx2('mspemat'))
    aggr_mspemat = np.append(aggr_mspemat, np.expand_dims(cv_mspemat, axis=2), axis = 2)
    hypar.loc[rep,'SPLS_K'] = np.array(cv_par.rx2('K.opt'))[0]
    hypar.loc[rep,'SPLS_eta'] = np.array(cv_par.rx2('eta.opt'))[0]

mean_mspemat = np.mean(aggr_mspemat, axis=2)
min_idx = np.unravel_index(np.argmin(mean_mspemat, axis=None), mean_mspemat.shape)
spls_eta_idx = min_idx[0]
spls_eta = eta_list[spls_eta_idx]
spls_K = K_list[min_idx[1]]
print('DONE.')
print('Global min on mean MSPE map: eta = {}  K = {}'.format(spls_eta, spls_K))
print('Mean of hyperpar. within repetitions: eta = {:.2f}  K = {:.1f}'
      .format(np.mean(hypar['SPLS_eta']), np.mean(hypar['SPLS_K'])))

# SPLS fit
print('\nSPLS fit...', end='')
spls_res = r_spls.spls(X.to_numpy(), y.to_numpy(), spls_K, spls_eta, trace = True)
print('DONE.')

# ENET hyperparameter (alpha, L1_ratio) selection and fit in one step
print('\nENET hyperparameter selection and fit on {} group(s) and {} condition(s)...'
      .format(', '.join(gp_list), ', '.join(sp_list)))
print('Using {}-fold cross-validation'.format(fold_num, rep_num))
l1_ratio_list = list(np.arange(0.1, 0.9, 0.1)) + list(np.arange(.9, 1.0, .005)) + [1]
seed = int(time.time())
regr = ElasticNetCV(cv=fold_num, random_state=seed, max_iter=10000, l1_ratio=l1_ratio_list)
regr.fit(X, y)
enet_res = pd.DataFrame(data=regr.coef_, index=X_names)
print('DONE.')
print('Best hyperparameters: alpha = {:.4f}  L1_ratio = {:.2f}'.format(regr.alpha_, regr.l1_ratio_))

# Adding ENET and SPLS LV coeffs to df
beta = pd.DataFrame(index = X_names)
beta['ENET'] = enet_res
for k in range(1, spls_K+1):
    curr_lv = np.array(spls_res.rx2('betamat').rx2(k))
    beta['LV'+str(k)] = curr_lv
    
print('\nTotal script runtime: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))

# %% PLOTS
print('\nPlotting results...', end='')
# Plot eta-K MSPE map
fig_mspe = plt.figure(figsize=(7,5))
ax = sns.heatmap(mean_mspemat, cmap = sns.cm.rocket_r)
ax.set_xlabel('K')
ax.set_ylabel('eta')
ax.set_xticklabels(K_list)
ax.set_yticks(np.array(range(len(eta_list)+1))[0::10])
ax.set_yticklabels(np.append(eta_list[0::10], 1))
ax.set_title('SPLS MSPE,  gp: {}  min at eta = {}  K = {}'.format(','.join(gp_list), spls_eta, spls_K))

# Plot K for selected eta
fig_K = plt.figure(figsize=(6,4))
ax = plt.axes()
plt.plot(K_list, mean_mspemat[spls_eta_idx,])
ax.set_xlabel('K')
ax.set_ylabel('MSPE')
ax.set_xticks(K_list)
ax.set_title('SPLS MSPE,  gp: {}  at eta = {} (min at K = {})'.format(','.join(gp_list), spls_eta, spls_K))

# Plot SPLS and Elastic net results
fig_res = plt.figure(figsize=(9,10))
ax = sns.heatmap(beta, cmap = 'RdBu_r', vmin = -.5, vmax = .5, annot = True)
fig_res.subplots_adjust(top=0.90, bottom=0.15, hspace=0.3, left=0.2, right=0.95)
ax.set_title('Groups: {}  Conditions: {} \nENET: alpha = {:.4f}  L1 ratio = {:.2f}   \nSPLS: K = {}  eta = {}  '
              .format(', '.join(gp_list), ', '.join(sp_list), regr.alpha_, regr.l1_ratio_, spls_K, spls_eta), pad=20)
# Bugfix for matplotlib 3.1.1
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print('DONE.')



# # %% Elastic net cross-validated
# regr = ElasticNetCV(cv=8, random_state=0, max_iter=10000, l1_ratio=list(np.arange(0.1, 0.9, 0.1)) + list(np.arange(.9, 1.0, .005)) + [1])
# regr.fit(X, y)
# print('alpha = ' + str(regr.alpha_))
# print('L1 ratio = ' + str(regr.l1_ratio_))
# res = pd.DataFrame(data=regr.coef_, index=X_names)
# # res = res[res != 0]
# # res = res.reindex(res.abs().sort_values(ascending = False).index)

# plt.figure()
# sns.heatmap(res, cmap = 'RdBu_r', vmin = -.5, vmax = .5, annot = True)

## %% MLR statsmodels OLS
#model = smf.ols(y_name+' ~ '+' + '.join(X_names), data=data)
#results = model.fit()
#print(results.summary())
#
## %% MLR statsmodels Lasso
#alpha = 0.1
#L1_wt = 0.5
#model = smf.ols(y_name+' ~ '+' + '.join(X_names), data=data)
#results = model.fit_regularized(alpha=alpha, L1_wt = L1_wt)
##print(results.summary())
##results.params
#
## Sorted values
#order = results.params.abs().sort_values(ascending = False)
#results.params[order.index]
#
## %% MLR sklearn
#mlr_mod = LinearRegression()
#mlr_mod.fit(X, y)
##print(mlr_mod.intercept_)
##print(mlr_mod.coef_)

# %% PCA
#pc = PCA(X)
#
#plt.figure(figsize=(20, 14))
#sns.heatmap(pc.loadings, 
#            cmap='RdBu_r', 
#            vmin=-1, 
#            vmax=1,
#            annot=True,
#            fmt=".2f",
#            annot_kws={"fontsize":8})