import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadr
import yaml
from matplotlib.backends.backend_pdf import PdfPages

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

# Load params YAML
with open('params.yaml') as file:
    p = yaml.full_load(file)
    
# Create reports dir
os.makedirs(p['reports_dir'], exist_ok=True)

def p2star(p):
    if p < 0.00001:
        return '*****'
    elif p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''
    
# Boxplot generation
def boxplot_condition_group(data, y_label = '', p_values = None, legend_loc=1):
    bp = sns.boxplot(x='factor_1', y='value', hue='factor_2', data = data, showmeans=False, whis = np.inf)
    handles, labels = bp.get_legend_handles_labels()
    plt.legend(loc=legend_loc, borderaxespad=1.5, frameon=False)
    bp.set(xlabel='factor_1', ylabel=y_label);
    
    # Stats
    if p_values:
        y_min = min(data['value'])
        y_max = max(data['value'])
        y = y_min-(y_max-y_min)*0.03
        for i in range(0,3):
            x1, x2 = -0.2+i, 0.2+i
            stat_str = p2star(p_values[i])
            if stat_str:
                plt.plot([x1, x2], [y, y], lw=2, c='k')
                plt.text((x1+x2)*.5, y-(y_max-y_min)*0.01, stat_str, ha='center', va='top', color='k')

# ANOVA table generation            
def gen_anova_table(df):
    # Format table
    df = df.copy()
    df = df.rename(columns={'DFn': 'df'})
    df['df'] = df['df'].astype(int)
    df['F'] = df['F'].map('{:,.2f}'.format)
    df['sign'] = df['p'].apply(p2star)
    df['p'] = df['p'].map('{:,.5f}'.format).replace({'0.00000': '< 0.00001'})
    
    # Plot table
    the_table = plt.table(cellText=df[['df', 'F', 'p', 'sign']].values,
                          colWidths=[0.08, 0.1, 0.12, 0.1],
                          rowLabels=list(df['Effect']),
                          colLabels=['df', 'F', 'p', 'sign'],
                          loc = 'best')
    
    # Format table plot
    cells = the_table.properties()["celld"]
    for i in range(0, df.shape[0]+1):
        cells[i, 3]._loc = 'center'
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(2, 2)

# Post-hoc table generation
def gen_ph_table(df):
    #• Format table
    df = df.copy()
    df = df.rename(columns={'lhs': 'contrast', 'std.error': 'SE', 'statistic': 'z', 'p.value': 'p'})
    df['estimate'] = df['estimate'].map('{:,.4f}'.format)
    df['SE'] = df['SE'].map('{:,.4f}'.format)
    df['z'] = df['z'].map('{:,.4f}'.format)
    df['sign'] = df['p'].map(p2star)
    df['p'] = df['p'].map('{:,.5f}'.format).replace({'0.00000': '< 0.00001'})
    
    # Plot table
    the_table = plt.table(cellText=df[['estimate', 'SE', 'z', 'p', 'sign']].values,
                          colWidths=[0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                          rowLabels=list(df['contrast']),
                          colLabels=['estimate', 'SE', 'z', 'p', 'sign'],
                          loc = 'best')
    
    # Format table plot
    cells = the_table.properties()["celld"]
    for i in range(0, df.shape[0]+1):
        cells[i, 4]._loc = 'center'
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.5, 1.5)

# Load data
data = pd.read_csv(p['data_path'])
data_long = pd.melt(data, id_vars=['factor_1', 'factor_2', 'subj_id'], var_name = 'measure', value_name = 'value')

# %% Generate pdf from figures
for type in range(2):        
    with PdfPages(p['reports_dir'] + '/repeated_ANOVA_report.pdf') as pdf:        
        for meas_name in p['meas_list']:
            # Load R stats
            anova_stats = pyreadr.read_r(p['stats_dir'] + '/mixed_ANOVA_' + meas_name + '.RData')
            
            # Boxplot
            fig = plt.figure(figsize=(16, 13))
            fig.add_axes([0.06, 0.5, 0.4, 0.4])
            y_label = ''
            boxplot_condition_group(data_long[data_long['measure'] == meas_name], y_label, list(anova_stats['ph_summary']['p.value'][:5]))
            plt.title(meas_name)
            
            # ANOVA table
            fig.add_axes([0.06, 0.25, 0.4, 0.18]).axis('off')
            gen_anova_table(anova_stats['ezANOVA_res'])
            
            # Post-hoc table
            fig.add_axes([0.5, 0.25, 0.45, 0.65]).axis('off')
            gen_ph_table(anova_stats['ph_summary'])
            
            # %%Histograms
            for sp, cond in enumerate(['f', 'g', 'no']):
                sp += 1
                ax = fig.add_axes([-0.165+sp*0.2, 0.06, 0.16, 0.16])
                for gp in ['F', 'G']:
                    x = data[(data['factor_1'] == cond) & (data['factor_2'] == gp)][meas_name]
                    sns.distplot(x, bins=8, label=gp, ax=ax)
                    plt.axvline(x.mean(), color='b' if gp == 'F' else 'r', linestyle='dashed', linewidth=1)
                    plt.title('SP'+str(sp))
            
            # Save fig as pdf page
            pdf.savefig()
            plt.close()

