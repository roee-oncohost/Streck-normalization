# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:30:03 2024

@author: BenYellin
"""


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from somadata import read_adat

steps = ['hybNorm','medNormInt','plateScale','calibrate','anmlQC','qcCheck','anmlSMP']
plate = '014'

old_cols = set()
old_index = None
old_meas = None
for i in range(len(steps)+1):
    fn = f'{plate}/OH2024_{plate}.{".".join(steps[:i]+["adat"])}'
    if i>0:
        print(f'##### {i} (step - {steps[i-1]}): {fn}')
    else:
        print(f'##### {i} (First file): {fn}')
    adat = read_adat(fn)
    adat.columns = adat.columns.get_level_values('SeqId')
    
    #### compare index data between frames
    index_df = adat.index.to_frame().reset_index(drop=True)
    current_cols = set(index_df.columns)
    add_cols = current_cols - old_cols
    rem_cols = old_cols - current_cols
    if len(add_cols) > 0:
        print(f'>> new columns:')
        print(', '.join(add_cols))
    if len(rem_cols) > 0:
        print(f'>> columns removed:')
        print(', '.join(rem_cols))
    if (len(add_cols)+len(rem_cols)) ==  0:
        print('XX no cols added or removed')
    
    
    mut_cols = current_cols&old_cols
    print(f'>> mutual_columns: {len(mut_cols)}')    
    if old_index is not None:
        ix = old_index[list(mut_cols)] != index_df[list(mut_cols)]
        if ix.any().any():
            ch = ix.sum()
            ch = ch[ch>0]
            print(ch)
        else:
            print('XX No values were changed between columns')
    
    if 'RowCheck' in index_df.columns:
        flg = (index_df.RowCheck=='FLAG').sum()  
        if flg > 0:
            print(f'@@ {flg} Flagged samples :(')
        else:
            print(f'@@ No flagged samples :)')
            
    old_index = index_df.copy()
    old_cols = current_cols.copy()
    
    
    
    ### compare measurments
    meas_df = adat.reset_index(drop=True)
    if i==0:
        meas0_df = meas_df.copy()
    else:
        ix = meas_df != old_meas
        ix = ix.T.sum()
        changed = index_df.loc[ix>0,'SampleType'].value_counts()
        not_changed = index_df.loc[ix==0,'SampleType'].value_counts()
        
        if len(changed)>0:
            txt = 'Changed meas: '
            txt += ', '.join([f'{x[1].SampleType} {x[1].loc["count"]}' for x in changed.reset_index().iterrows()])
            print(txt)
        if len(not_changed)>0:
            txt = 'Not changed meas: '
            txt += ', '.join([f'{x[1].SampleType} {x[1].loc["count"]}' for x in not_changed.reset_index().iterrows()])
            print(txt)
        
    old_index = index_df.copy()
    old_cols = current_cols.copy()
    old_meas = meas_df.copy()
    
    
print('############## Finished all steps')

