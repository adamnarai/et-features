"""


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

from rpy2.robjects import r, pandas2ri, conversion
from rpy2.robjects.packages import importr
pandas2ri.activate()
r_nlme = importr('nlme')
r_base = importr('base')
r_ez = importr('ez')

# Load params YAML
with open(SETTINGS_DIR + '/params.yaml') as file:
    p = yaml.full_load(file)

# Read data
data = pd.read_pickle(
    os.path.join(RESULTS_DIR, 'df_data', 'all', 'et', 'et_features.pkl'))
data = data[data.study == 'dys']

# ANOVA
res = r_ez.ezANOVA(data = data,
                   dv = r_base.as_symbol('Med_rspeed_wnum'),
                   wid = r_base.as_symbol('subj_id'), 
                   within = r_base.as_symbol('condition'),
                   between = r_base.as_symbol('group'),
                   detailed = True)
anova = pandas2ri.ri2py(res.rx2('ANOVA'))

# LME ANOVA table
res = r.anova(r_nlme.lme(r.formula('Med_rspeed_wnum ~ condition*group'), 
                  random=r.formula('~1|subj_id/condition'), 
                  data=data, 
                  method='ML'))
lme_anova = pandas2ri.ri2py(res)

# LME ANOVA post-hocs
res = r.anova(r_nlme.lme(r.formula('Med_rspeed_wnum ~ condition*group'), 
                  random=r.formula('~1|subj_id/condition'), 
                  data=data, 
                  method='ML'))
lme_anova = pandas2ri.ri2py(res)
