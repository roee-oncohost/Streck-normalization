# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:11:39 2024

@author: GilLoewenthal following: https://www.biorxiv.org/content/10.1101/2024.02.09.579724v1.full.pdf
except for ANML normalization
"""

# TODO: remove hybradization control later

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#%% functions per normalization steps
    
## I assume there is one plate in each adat from now on - if not, the functuin should be run per plate and merge all the plates

def hyb_norm(RFU, somamers):
    # 1st normalization - Hybridization control normalization

    sel_HCE_seqid = list(somamers[somamers['Type'] == "Hybridization Control Elution"].index)
    n_HCE = len(sel_HCE_seqid)
    
    
    RFU_HCE_i = RFU.loc[:, sel_HCE_seqid]
    RFU_HCEref_i = RFU_HCE_i.median()
    # np.median(RFU_HCEref_i / RFU.loc[i, sel_HCE_seqid])
    
    # hyb_factor_per_well = (RFU_HCEref_i / RFU.loc[:, sel_HCE_seqid]).median(axis=1)
    hyb_factor_per_well = (RFU_HCEref_i / RFU_HCE_i).median(axis=1)
    RFU_hyb = RFU.mul(hyb_factor_per_well, axis=0)
    RFU_hyb = RFU_hyb.apply(lambda x: round(x, 1))
    
    return RFU_hyb


def msnCal_norm(RFU_hyb, somamers, sampl, dil_lab=("0", "0_005", "0_5", "20")):
    # 2nd normalization - Median signal normalization on calibrators
    RFU_msnCal = RFU_hyb.copy()
    
    for sampl_type in ['Calibrator', 'Buffer']: # Only calibrators according to the paper
        sel2 = sampl['SampleType'] == sampl_type
        indx_sampl = sampl.index[sel2]
        for dil in dil_lab:
                somamers_tmp = somamers[somamers['Dilution'] == dil].index
                # RFU_msnCal.loc[indx_sampl, somamers_tmp].median()
                rgd_ialpha = RFU_msnCal.loc[indx_sampl, somamers_tmp] / RFU_msnCal.loc[indx_sampl, somamers_tmp].median()
                SFgd_ialpha = 1 / rgd_ialpha.median(axis=1)
                RFU_msnCal.loc[indx_sampl, somamers_tmp] = RFU_msnCal.loc[indx_sampl, somamers_tmp].mul(SFgd_ialpha, axis=0)
    RFU_msnCal = RFU_msnCal.apply(lambda x: round(x, 1))
    return RFU_msnCal


def ps_norm(RFU_msnCal, sampl, ps_ref_path='SD4.1ReV_Plasma_Calibrator_200169.txt'):
    # 3rd normalization - Plate scale normalization
    ser_ps_ref = pd.read_csv(ps_ref_path, sep='\t', index_col=0)['Reference']
    RFU_ps = RFU_msnCal.copy()
    
    sel2 = sampl['SampleType'] == 'Calibrator'
    indx_sampl = sampl.index[sel2]
    SF = (ser_ps_ref / RFU_msnCal.loc[indx_sampl].median()).median()

    RFU_ps = RFU_ps * SF
    RFU_ps = RFU_ps.apply(lambda x: round(x, 1))
    return RFU_ps


def cal_norm(RFU_ps, sampl, ps_ref_path):
    # 4th normalization - Inter-plate calibration
    ser_ps_ref = pd.read_csv(ps_ref_path, sep='\t', index_col=0)['Reference']
    RFU_cal = RFU_ps.copy()
     
    sel2 = sampl['SampleType'] == 'Calibrator'
    indx_sampl = sampl.index[sel2]
    SF_i = (ser_ps_ref / RFU_cal.loc[indx_sampl].median())
    RFU_cal = RFU_cal * SF_i
    RFU_cal = RFU_cal.apply(lambda x: round(x, 1))
    return RFU_cal


def msnAll_norm(RFU_ps, somamers, sampl, dil_lab=("0", "0_005", "0_5", "20")):
    # 5th normalization - Median signal normalization on all sample types
    # note it is different than SL algo, but yields similar results according to Candia paper
    RFU_msnAll = RFU_ps.copy()
    
    for sampl_type in range(len(RFU_msnAll)): #['Sample', 'Calibrator', 'Buffer', 'QC']: # Only calibrators according to the paper
        print(sampl_type)
        # sel2 = sampl['SampleType'] == sampl_type
        indx_sampl = [sampl_type]#sampl.index[sel2]
        for dil in dil_lab:
                somamers_tmp = somamers[somamers['Dilution'] == dil].index
                # RFU_msnCal.loc[indx_sampl, somamers_tmp].median()
                rgd_ialpha = RFU_msnAll.loc[indx_sampl, somamers_tmp] / RFU_msnAll.loc[indx_sampl, somamers_tmp].median()
                SFgd_ialpha = 1 / rgd_ialpha.median(axis=1)#(axis=1)
                RFU_msnAll.loc[indx_sampl, somamers_tmp] = RFU_msnAll.loc[indx_sampl, somamers_tmp].mul(SFgd_ialpha, axis=0)
    RFU_msnAll = RFU_msnAll.apply(lambda x: round(x, 1))
    return RFU_msnAll

#%% Loading the data and using the Candia notations

adat_path = r"C:\Users\GilLoewenthal\Oncohost DX\Shares - Gil Loewenthal\code\sl_normalization\241027 Flag Pass check\014\OH2024_014.adat" # Flagged samples [3, 8, 12, 19, 51, 55, 56, 60, 66, 68, 86, 88, 91, 95]
# somamer_data_path = r"C:\Users\GilLoewenthal\Oncohost DX\Shares - Gil Loewenthal\general_data\SomaScan v4.1_7K_Annotated Content 27073023xlsx.xlsx"
somamer_data_path = r"C:\Users\GilLoewenthal\Oncohost DX\Shares - Gil Loewenthal\general_data\SomaScan_V4.1_7K_Annotated_Content_20241119.xlsx"

sampl = pd.read_csv(adat_path, skiprows=43, sep='\t')
sampl_cols = ['PlateId', 'PlatePosition', 'SampleId', 'SampleType',
              'SampleMatrix', 'Barcode', 'ExtIdentifier',]
sampl = sampl[sampl_cols]

with open(adat_path, 'r') as file:
    lines = file.readlines()
    if len(lines) >= 32:
        seqid = lines[31].strip()
seqid_order_list = seqid.split('\t')[1:]



somamers = pd.read_excel(somamer_data_path, skiprows=8, sheet_name='Annotations')
somamers_cols = ['SeqId', 'SomaId', 'Target Name', 'UniProt ID',
                 'Entrez Gene ID', 'Entrez Gene Name', 'Type',
                 'Organism', 'Dilution', ]
somamers = somamers[somamers_cols]
somamers = somamers.rename(columns={'Target Name': 'Target',
                                    'UniProt ID': 'UniProt', 
                                    'Entrez Gene Name': 'EntrezGeneSymbol',
                                    'Entrez Gene ID': 'EntrezGeneID'})
somamers['Dilution'] = somamers['Dilution'].apply(lambda x: x[:-1].replace('.','_'))
somamers = somamers.sort_values(by='SeqId')
somamers = somamers.set_index('SeqId')
assert(all(somamers.index == seqid_order_list))

RFU = pd.read_csv(adat_path, skiprows=43, sep='\t')
RFU = RFU[RFU.columns[30:]]
RFU = RFU.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU.columns)})

#%% Basic parameters

norm = ["raw", "hyb", "hyb.msnCal", "hyb.msnCal.ps", "hyb.msnCal.ps.cal", "hyb.msnCal.ps.cal.msnAll"]

n_norm = len(norm)
n_sampl = sampl.shape[0]
n_somamer = somamers.shape[0]
plate = sampl["PlateId"].unique()  # Assuming "PlateId" is in the header
n_plate = len(plate)
dil = [0, 0.005, 0.5, 20]
dil_lab = ["0", "0_005", "0_5", "20"]
n_dil = len(dil)
sample_types = ['Sample', 'Buffer', 'Calibrator', 'QC']

#%% Additional parameters

ps_ref_path = 'SD4.1ReV_Plasma_Calibrator_200169.txt'

#%% 1 hyb reference

RFU_hyb_ref = pd.read_csv(adat_path.replace('.adat','.hybNorm.adat'), skiprows=44, sep='\t')
RFU_hyb_ref_meta = RFU_hyb_ref[RFU_hyb_ref.columns[:32]]
RFU_hyb_ref = RFU_hyb_ref[RFU_hyb_ref.columns[32:]]
RFU_hyb_ref = RFU_hyb_ref.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_hyb_ref.columns)})
hyb_ratio = RFU_hyb_ref.values / RFU.values

# plt.hist(hyb_ratio.flatten())


#%% 1 hyb  

## I assume there is one plate in each adat from now on - if not, each step should be done per plate

RFU_hyb = hyb_norm(RFU, somamers)


# Checking the error of the reverse engineering
(RFU_hyb / RFU_hyb_ref).max().max()
(RFU_hyb / RFU_hyb_ref).min().min() # I got error of up to 3%

(RFU_hyb / RFU_hyb_ref)

z = RFU_hyb / RFU

x = (RFU_hyb / RFU_hyb_ref)

y = (x <= 0.99) | (x>=1.01)
y = y.apply(sum, axis=1) # TODO: Two samples are not good, the rest are fine, why? Not calibratorts/qc/buffer (66, 13 are not calibrated well - 66 is flagged later not by this step)
# TODO: check if bad samples are flagged
# TODO: start from good plates
# TODO: % vs. error for sample

z = (np.abs(x-1)).apply(sum, axis=1)
plt.plot(z)
#%% 2 hyb.msnCal reference

RFU_msnCal_ref = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.adat'), skiprows=46, sep='\t')
RFU_msnCal_ref_meta = RFU_msnCal_ref[RFU_msnCal_ref.columns[:35]]
RFU_msnCal_ref = RFU_msnCal_ref[RFU_msnCal_ref.columns[35:]]
RFU_msnCal_ref = RFU_msnCal_ref.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_msnCal_ref.columns)})
msnCal_ratio = RFU_msnCal_ref / RFU_hyb_ref

plt.hist(msnCal_ratio.values.flatten())

#%% 2 hyb.msnCal without propgating error # Now almost perfect (less than 1% maximal error)

RFU_msnCal = msnCal_norm(RFU_hyb_ref, somamers, sampl, dil_lab)

            
(RFU_msnCal / RFU_msnCal_ref).min().min()
(RFU_msnCal / RFU_msnCal_ref).max().max()

x = (RFU_msnCal / RFU_msnCal_ref)
y = (x < 0.99) | (x>1.01)
y = y.apply(sum, axis=1)

x.max().max()
x.min().min()



#%% 2 hyb.msnCal with propgating error

RFU_msnCal = msnCal_norm(RFU_hyb, somamers, sampl, dil_lab)

            
(RFU_msnCal / RFU_msnCal_ref).min().min()
(RFU_msnCal / RFU_msnCal_ref).max().max()

#%% 3 hyb.msnCal.ps reference

RFU_ps_ref = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.plateScale.adat'), skiprows=50, sep='\t')
RFU_ps_ref_meta = RFU_ps_ref[RFU_ps_ref.columns[:35]]
RFU_ps_ref = RFU_ps_ref[RFU_ps_ref.columns[35:]]
RFU_ps_ref = RFU_ps_ref.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_ps_ref.columns)})
ps_ratio = RFU_ps_ref / RFU_msnCal_ref

plt.hist(ps_ratio.values.flatten())


#%% 3 hyb.msnCal.ps with propgating error


RFU_ps = ps_norm(RFU_msnCal, sampl, ps_ref_path)


(RFU_ps / RFU_ps_ref).max().max()
(RFU_ps / RFU_ps_ref).min().min()

x = RFU_ps / RFU_ps_ref


#%% 3 hyb.msnCal.ps without propgating error - perfect!


RFU_ps = ps_norm(RFU_msnCal_ref, sampl, ps_ref_path)


(RFU_ps / RFU_ps_ref).max().max()
(RFU_ps / RFU_ps_ref).min().min()

x = RFU_ps / RFU_ps_ref

#%% 4 hyb.msnCal.ps.cal reference

RFU_cal_ref = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.plateScale.calibrate.adat'), skiprows=58, sep='\t')
RFU_cal_ref_meta = RFU_cal_ref[RFU_cal_ref.columns[:35]]
RFU_cal_ref = RFU_cal_ref[RFU_cal_ref.columns[35:]]
RFU_cal_ref = RFU_cal_ref.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_cal_ref.columns)})
ps_ratio = RFU_cal_ref / RFU_ps_ref

plt.hist(ps_ratio.values.flatten(), bins=30)

#%% 4 hyb.msnCal.ps.cal with propgating error

RFU_cal = cal_norm(RFU_ps, sampl, ps_ref_path)

(RFU_cal / RFU_cal_ref).max().max()
(RFU_cal / RFU_cal_ref).min().min()

#%% 4 hyb.msnCal.ps.cal without propgating error - almost perfect. max relative error 0.0025

RFU_cal = cal_norm(RFU_ps_ref, sampl, ps_ref_path)

(RFU_cal / RFU_cal_ref).max().max()
(RFU_cal / RFU_cal_ref).min().min()

#%% 5 hyb.msn.Cal.ps.cal.msnAll semi reference
RFU_msnAll_ref1 = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.plateScale.calibrate.anmlQC.adat'), skiprows=58, sep='\t')
RFU_msnAll_ref1_meta = RFU_msnAll_ref1[RFU_msnAll_ref1.columns[:35]]
RFU_msnAll_ref1 = RFU_msnAll_ref1[RFU_msnAll_ref1.columns[38:]]
RFU_msnAll_ref1 = RFU_msnAll_ref1.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_msnAll_ref1.columns)})
ps_ratio = RFU_msnAll_ref1 / RFU_cal_ref # This stage changes QC

plt.hist(ps_ratio.values.flatten(), bins=30)

RFU_msnAll_ref2 = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.adat'), skiprows=63, sep='\t') # no change
RFU_msnAll_ref2_meta = RFU_msnAll_ref2[RFU_msnAll_ref2.columns[:35]]
RFU_msnAll_ref2 = RFU_msnAll_ref2[RFU_msnAll_ref2.columns[38:]]
RFU_msnAll_ref2 = RFU_msnAll_ref2.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_msnAll_ref2.columns)})
ps_ratio = RFU_msnAll_ref2 / RFU_msnAll_ref1

plt.hist(ps_ratio.values.flatten(), bins=30)

RFU_msnAll_ref3 = pd.read_csv(adat_path.replace('.adat','.hybNorm.medNormInt.plateScale.calibrate.anmlQC.qcCheck.anmlSMP.adat'), skiprows=63, sep='\t') # no change
RFU_msnAll_ref3_meta = RFU_msnAll_ref3[RFU_msnAll_ref3.columns[:35]]
RFU_msnAll_ref3 = RFU_msnAll_ref3[RFU_msnAll_ref3.columns[38:]]
RFU_msnAll_ref3 = RFU_msnAll_ref3.rename(columns = {x: seqid_order_list[ind]  for ind, x in enumerate(RFU_msnAll_ref3.columns)})
ps_ratio = RFU_msnAll_ref3 / RFU_msnAll_ref2 # This stage changes samples only

plt.hist(ps_ratio.values.flatten(), bins=30)

# TODO: check the order. The order matters?


#%% 5 hyb.msn.Cal.ps.cal.msnAll

RFU_msnAll = msnAll_norm(RFU_cal, somamers, sampl, dil_lab)

(RFU_msnAll / RFU_msnAll_ref3).max().max()
(RFU_msnAll / RFU_msnAll_ref3).min().min()


#%% Trying to replicate ANML
df_anml = pd.read_csv('SD4.1ReV_Plasma_ANML.txt', sep='\t')
ref_med_col = 'RefMedian' # 'Reference'
# df_anml = pd.read_csv('SD4.1ReV_Plasma_QC_ANML_200170.txt', sep='\t')
# df_anml = pd.merge(left=pd.read_csv('SD4.1ReV_Plasma_ANML.txt', sep='\t'), right=df_anml, on='SeqId', how='inner')
df_anml = df_anml.set_index('SeqId')
num_anml_iterations = 10#200

dil_lab= ("20", "0", "0_005", "0_5", )
RFU_anmlCal = RFU_cal_ref.copy() #RFU_cal.copy()


res = {}
for sampl_type in ['Sample']: # Only calibrators according to the paper
    sel2 = sampl['SampleType'] == sampl_type
    indx_sampl_list = sampl.index[sel2]
    for ind, indx_sampl in enumerate(indx_sampl_list):
        print(ind)
        res[indx_sampl] = {}
        for dil in dil_lab:
            for ind in range(num_anml_iterations):
                somamers_dil = somamers[somamers['Dilution'] == dil].index
                somamers_dil_in_range = (RFU_anmlCal.loc[indx_sampl, somamers_dil] >= (df_anml.loc[somamers_dil, ref_med_col] - 2*df_anml.loc[somamers_dil, 'RefSD'])) & (RFU_anmlCal.loc[indx_sampl, somamers_dil] <= (df_anml.loc[somamers_dil, ref_med_col] + 2*df_anml.loc[somamers_dil, 'RefSD'])) 
                somamers_dil_in_range = somamers_dil_in_range.loc[somamers_dil_in_range].index
                # RFU_msnCal.loc[indx_sampl, somamers_tmp].median()
                rgd_ialpha = RFU_anmlCal.loc[indx_sampl, somamers_dil_in_range] / df_anml.loc[somamers_dil_in_range, ref_med_col]
                SFgd = 1 / rgd_ialpha.median()
                RFU_anmlCal.loc[indx_sampl, somamers_dil] = RFU_anmlCal.loc[indx_sampl, somamers_dil] * SFgd
                RFU_anmlCal.loc[indx_sampl, somamers_dil] = RFU_anmlCal.loc[indx_sampl, somamers_dil].apply(lambda x: round(x,1))
                if ind == 0:
                    res[indx_sampl][dil] = (len(somamers_dil_in_range)/len(somamers_dil), SFgd)
                else:
                    res[indx_sampl][dil] = (len(somamers_dil_in_range)/len(somamers_dil), res[indx_sampl][dil][1]*SFgd)
            # if res[indx_sampl][dil][1] < 0.3 or res[indx_sampl][dil][1] > 2.5 or res[indx_sampl][dil][0] < 0.3:
            #     RFU_anmlCal.loc[indx_sampl, somamers_dil] = RFU_cal.loc[indx_sampl, somamers_dil]



(RFU_anmlCal / RFU_msnAll_ref3).max().max()
(RFU_anmlCal / RFU_msnAll_ref3).min().min()
t = (RFU_anmlCal / RFU_cal_ref)

x = RFU_anmlCal / RFU_msnAll_ref3

x.max().max()
x.min().min()
(x.iloc[:,0] < 1.05).sum() / len(x)

x = (RFU_anmlCal / RFU_msnAll_ref3)
x = np.abs(x - 1)
# y = (x < 0.99) | (x>1.01) #(x < 0.99) | (x>1.01)
y = x>0.01
y = y.apply(sum, axis=1) # Works perfect on Calibrator and Buffer, 
z = x.apply(sum, axis=1)

by_seqid = pd.merge(left=pd.DataFrame(x.mean(), columns=['Error']), right=somamers, left_index=True, right_index=True)
by_seqid.groupby('Dilution')['Error'].mean()
#%%
sampl['SampleType'].value_counts()
sampl[sampl['SampleType'] == 'Calibrator']
sampl[sampl['SampleType'] == 'Buffer']
sampl[sampl['SampleType'] == 'QC']

sampl[sampl['SampleType'].isin(['Calibrator', 'Buffer'])]
#%%

import json

# Path to your JSON file
file_path = "241027 Flag Pass check/014/OH2024_014.json"

# Open and load the JSON file into a dictionary
with open(file_path, "r") as json_file:
    data = json.load(json_file)