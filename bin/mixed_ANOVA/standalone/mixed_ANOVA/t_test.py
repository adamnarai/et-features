import pandas as pd
import pingouin as pg

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)

sp = 'SP2'
df = pd.read_csv('perf.csv', index_col=0)
df = df[df['condition']==sp]

x = df[df['group']=='control']['perf']
y = df[df['group']=='dyslexic']['perf']

print(df.groupby('group')['perf'].describe()*100)
print(pg.mwu(x, y))
print(pg.ttest(x, y))