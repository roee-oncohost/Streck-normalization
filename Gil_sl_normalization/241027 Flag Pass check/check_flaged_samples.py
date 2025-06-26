import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

raw_df = pd.read_csv('20241022_Adat_data_Samples.csv')
with open('conf.json','r') as f:
    conf_dict = json.load(f)

step_dict = {}
for i, step in enumerate(conf_dict['ReportConfig']['analysisSteps']):
    step_dict[i] = step.copy()

step_df = pd.DataFrame(step_dict).T

df = raw_df[raw_df.Type=='Sample']
#df = df.drop_duplicates('SubjectId')

norm_cols = ['HybControlNormScale',
             'NormScale_20', 'NormScale_0_005', 'NormScale_0_5',
             'ANMLFractionUsed_20', 'ANMLFractionUsed_0_005', 'ANMLFractionUsed_0_5']

th_dict = {}
for col in norm_cols:
    sns.stripplot(df, x='RowCheck', y=col)
    plt.title(col)
    plt.show()
    
    ix = df.RowCheck=='PASS'
    th_dict[col] = {'min':df.loc[ix,col].min(),'max':df.loc[ix,col].max()}
    
th_df = pd.DataFrame(th_dict).T
print(th_df)

### check 1 : check 0.4<scale_cols<2.5 & 0.3<anml_cols<2 

scale_cols = ['HybControlNormScale', 'NormScale_20', 'NormScale_0_005', 'NormScale_0_5']
anml_cols = ['ANMLFractionUsed_20', 'ANMLFractionUsed_0_005', 'ANMLFractionUsed_0_5']             
ix_scale = ((df[scale_cols]>2.5)|(df[scale_cols]<0.4)).any(axis=1)
ix_anml = (df[anml_cols]<0.3).any(axis=1)

df['BenCheck1'] = (ix_scale|ix_anml).map({True:'Fail',False:'Pass'})

for col in norm_cols:
    sns.violinplot(df, x='RowCheck', y=col, hue = 'BenCheck1')
    sns.stripplot(df, x='RowCheck', y=col, hue = 'BenCheck1')
    plt.title(col)
    plt.show()


### other checks
ix1 = df['HybControlNormScale'] > 2
ix2 = df['NormScale_20'] < 0.4
ix3 = df['NormScale_0_5'] < 0.4
ix4 = df['NormScale_0_005'] >2.5
ix5 = df['ANMLFractionUsed_0_5'] < 0.3
ix_list = [ix1,ix2,ix3,ix4,ix5]
df['BenCheck2'] = (ix1|ix2|ix3|ix4|ix5).map({True:'Fail',False:'Pass'})

for col in norm_cols:
    sns.stripplot(df, x='RowCheck', y=col, hue = 'BenCheck2')
    plt.title(col)
    plt.show()

summary_list = []
for r in range(1, len(ix_list)+1):
    combs = combinations(ix_list, r)
    for comb in combs:
        comb_df = pd.concat(comb,axis=1)
        conds = list(comb_df.columns)
        comb_dict = {x:True for x in conds}
        comb_dict['FLAG'] = (df.loc[comb_df.T.any(),'RowCheck']=='FLAG').sum()
        comb_dict['PASS'] = (df.loc[comb_df.T.any(),'RowCheck']=='PASS').sum()
        summary_list.append(comb_dict)

summary_df = pd.DataFrame(summary_list).fillna(False)        
summary_df = summary_df[conds+['FLAG','PASS']]
summary_df['FLAG_p'] = summary_df['FLAG']/(df.RowCheck=='FLAG').sum()
    

# th_dict2 = {}
# for col in scale_cols[1:]:
#     new_col = col +'_BY'
#     df[new_col] = df['HybControlNormScale']*df[col]
#     sns.stripplot(df, x='RowCheck', y=new_col)
#     plt.title(new_col)
#     plt.show()
    
#     ix = df.RowCheck=='PASS'
#     th_dict2[new_col] = {'min':df.loc[ix,new_col].min(),'max':df.loc[ix,new_col].max()}
    
# th_df2 = pd.DataFrame(th_dict2).T
# print(th_df2)

## check if missed samples by check1:
print('\n>>>>>>>> Check 1:')
ix = (df.RowCheck=='FLAG')&(df.BenCheck1=='Pass')
if ix.sum() > 0:
    print('Missed FLAG samples:')
    print(df[ix].T)
else: 
    print('All flagged samples idetified')
    
ix = (df.RowCheck=='PASS')&(df.BenCheck1=='Fail')
if ix.sum() > 0:
    print('Failed PASS samples:')
    print(df[ix].T)
else: 
    print('No PASS samples was failed')
    
        
## check if missed samples by check2:
print('\n>>>>>>>> Check 2:')
ix = (df.RowCheck=='FLAG')&(df.BenCheck2=='Pass')
if ix.sum() > 0:
    print('Missed FLAG samples:')
    print(df[ix].T)
else: 
    print('All flagged samples idetified')
    
ix = (df.RowCheck=='PASS')&(df.BenCheck2=='Fail')
if ix.sum() > 0:
    print('Failed PASS samples:')
    print(df[ix].T)
else: 
    print('No PASS samples was failed')
    
        





