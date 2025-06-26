# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:24:19 2024

@author: GilLoewenthal
"""

import pandas as pd
import numpy as np

#%%

df = pd.read_excel("241027 Flag Pass check/Flagged samples_gl_edit.xlsx", sheet_name='Flagged samples')
normalization_control_cols = ['HybControlNormScale', 'NormScale_20',
                              'NormScale_0_005', 'NormScale_0_5',
                              'ANMLFractionUsed_20', 'ANMLFractionUsed_0_005',
                              'ANMLFractionUsed_0_5']

for col in ['ANMLFractionUsed_20', 'ANMLFractionUsed_0_005', 'ANMLFractionUsed_0_5']:
    df[f'{col}_result'] = df[col].apply(lambda x: True if type(x)==float and x>0.3 else False)
    print(col)
    print(df[f'{col}_result'].value_counts())
    print()
    
    
#%%

adat_path = r"C:\Users\GilLoewenthal\Oncohost DX\Shares - Gil Loewenthal\code\sl_normalization\241027 Flag Pass check\014\OH2024_014.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.anmlSMP.adat"

with open(adat_path, 'r') as f:
    for line_num, line in enumerate(f):
        if 'CalQcRatio' in line:
            CalQcRatio_line = line_num
CalQcRatio = pd.read_csv(adat_path, skiprows=CalQcRatio_line, nrows=1, header=None, sep='\t').T.dropna().iloc[1:]
if np.sum((CalQcRatio > 1.2) | (CalQcRatio < 0.8)).iloc[0] / len(CalQcRatio) < 0.15:
    print('Passed QC Check')
else:
    print('Failed QC Check')

        
RFU_msnAll_ref3 = pd.read_csv(adat_path, skiprows=63, sep='\t') # no change .replace('.adat','.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.anmlSMP.adat')
RFU_msnAll_ref3_meta = RFU_msnAll_ref3[RFU_msnAll_ref3.columns[:37]]


# Checking hybradization normalization
if all((RFU_msnAll_ref3_meta['HybControlNormScale'] > 0.4) & (RFU_msnAll_ref3_meta['HybControlNormScale']<2.5)):
    print('Passed hybradization control test')
else:
    print('Falied hybradization control test')

(RFU_msnAll_ref3_meta['ANMLFractionUsed_0_005'].dropna() > 0.3).sum() - len(RFU_msnAll_ref3_meta['ANMLFractionUsed_0_005'].dropna())

RFU_msnAll_ref3_meta['ANMLFractionUsed_0_5'].dropna() > 0.3

RFU_msnAll_ref3_meta['ANMLFractionUsed_20'].dropna() > 0.3

(RFU_msnAll_ref3_meta['NormScale_0.005'] > 0.4) & (RFU_msnAll_ref3_meta['NormScale_0.005'] < 2.5)
(RFU_msnAll_ref3_meta['NormScale_0.5'] > 0.4) & (RFU_msnAll_ref3_meta['NormScale_0.5'] < 2.5)
len(RFU_msnAll_ref3_meta['NormScale_20']) - ((RFU_msnAll_ref3_meta['NormScale_20'] > 0.4) & (RFU_msnAll_ref3_meta['NormScale_20'] < 2.5)).sum() # This is the failed test 0_16


RFU_msnAll_ref3_meta.loc[~((RFU_msnAll_ref3_meta['NormScale_20'] > 0.4) & (RFU_msnAll_ref3_meta['NormScale_20'] < 2.5))].index
#%%

