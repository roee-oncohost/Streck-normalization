# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:19:47 2025

@author: GilLoewenthal
"""

import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['figure.dpi'] = 300
import seaborn as sns

#%%

def create_anml_ref_df(df_anml_ref_sample, df_calib_meas):
    anml_ref_new = df_anml_ref_sample.copy().set_index('SeqId')
    anml_ref_new['RefMedian'] = df_calib_meas.median()
    anml_ref_new['RefSD'] =  df_calib_meas.std()
    anml_ref_new['RefLog10Median'] = np.log10(df_calib_meas).median()
    anml_ref_new['RefLog10MAD'] = stats.median_abs_deviation(np.log10(df_calib_meas), scale='normal')
    anml_ref_new['SeqId'] = anml_ref_new.index
    anml_ref_new = anml_ref_new[df_anml_ref_sample.columns].set_index('Target')
    return anml_ref_new
    


#%% Loading the data

df_prot_calib = pd.read_csv(r"C:\Users\GilLoewenthal\Oncohost DX\Shares - R&D\Or\proteomic_data_in_calibrator_norm_for_GL\calibrator_norm_proteomics_partial_set.csv")
pattern = re.compile(r'^\d+-\d+$')
prot_cols = [x for x in df_prot_calib.columns if pattern.match(x)]
df_prot_calib[prot_cols].median()
anml_ref = pd.read_csv('SD4.1ReV_Plasma_ANML.txt',delimiter='\t')
anml_ref_nsclc = anml_ref.copy().set_index('SeqId')
anml_ref_nsclc['RefMedian'] = df_prot_calib[prot_cols].median()
anml_ref_nsclc['RefSD'] =  df_prot_calib[prot_cols].std()
anml_ref_nsclc['RefLog10Median'] = np.log10(df_prot_calib[prot_cols]).median()
anml_ref_nsclc['RefLog10MAD'] = stats.median_abs_deviation(np.log10(df_prot_calib[prot_cols]), scale='normal')
anml_ref_nsclc['SeqId'] = anml_ref_nsclc.index
anml_ref_nsclc = anml_ref_nsclc[anml_ref.columns].set_index('Target')
# anml_ref_nsclc.to_csv('OH_NSCLC_ANML_ref.txt', sep='\t')

create_anml_ref_df(anml_ref, df_prot_calib[prot_cols]) == anml_ref_nsclc


plt.figure()
plt.plot(anml_ref_nsclc['RefLog10MAD'], anml_ref['RefLog10MAD'], 'x')
plt.plot(anml_ref_nsclc['RefLog10MAD'], anml_ref_nsclc['RefLog10MAD'], '--')
plt.show()

df_hnc_calib = pd.read_csv(r"C:\Users\GilLoewenthal\Oncohost DX\Shares - R&D\Or\proteomic_data_in_calibrator_norm_for_GL\HNC\HNC_proteomics_calibrate_measurements.csv", index_col=0)

anml_ref_hnc = create_anml_ref_df(anml_ref, df_hnc_calib)
anml_ref_hnc.to_csv('OH_HNC_ANML_ref.txt', sep='\t')
