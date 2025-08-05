import os
import numpy as np
import pandas as pd


def truncate(number, decimals):
    multiplier = 10**decimals
    number = np.floor(number * multiplier)/multiplier
    return number


def hyb_norm(RFU, control_rfu, somamers):
    # 1st normalization - Hybridization control normalization
    # roee: from each well take only the readings for the 12 hybridization control elutions
    # roee: then take the 12 medians of these readings from the entire plate
    sel_HCE_seqid = list(somamers[somamers['Type'] == "Hybridization Control Elution"].index)
    # n_HCE = len(sel_HCE_seqid)
    # Calibrator, Buffer, and QC
    
    RFU_HCE_i = control_rfu.loc[:, sel_HCE_seqid] # RFU.loc[:, sel_HCE_seqid] # 11X12 readings across the entire plate
    RFU_HCEref_i = RFU_HCE_i.median() # 12 medians across entire plate (96 wells)
    # np.median(RFU_HCEref_i / RFU.loc[i, sel_HCE_seqid])
    
    # hyb_factor_per_well = (RFU_HCEref_i / RFU.loc[:, sel_HCE_seqid]).median(axis=1)
    hyb_factor_per_well = (RFU_HCEref_i / RFU_HCE_i).median(axis=1) # 96X12 ratios then the median of 12 ratios per well (so 1 per well) 
    # hyb_factor_per_well = hyb_factor_per_well.apply(lambda x: truncate(x, 2))
    scaling_factors = (RFU_HCEref_i/RFU[sel_HCE_seqid]).median(axis=1)
    RFU_hyb = RFU.mul(scaling_factors, axis=0)
    # RFU_hyb = RFU.mul(hyb_factor_per_well, axis=0) # each 
    RFU_hyb = RFU_hyb.apply(lambda x: round(x, 1))
    # RFU_hyb = RFU_hyb.apply(lambda x: truncate(x, 1))
    return RFU_hyb


def msnCal_norm(RFU_hyb, somamers, sampl, dil_lab=("0", "0_005", "0_5", "20")):
    # 2nd normalization - Median signal normalization on calibrators
    RFU_msnCal = RFU_hyb.copy()
    
    for sampl_type in ['Calibrator', 'Buffer']: #, 'QC']: #['Calibrator']: #['Calibrator', 'Buffer']: # Only calibrators according to the paper (and SomaLogic's documentation)
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
     
    sel2 = sampl['SampleType'].isin(['Calibrator']) 
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



def ANML_norm(RFU_cal_ref, somamers, sampl, num_anml_iterations=200, dil_lab=("0", "0_005", "0_5", "20"), ref_files_path='../'):
    # 5th normalization - ANML normalization
    ref_med_col = 'RefLog10Median' # 'RefMedian' # 'Reference'
    ref_sd_col = 'RefLog10MAD' #
    # factor = 1.4826*3 #2 
    factor_dict = {'20': 2, '0': 2, '0_005': 2, '0_5': 2,} #{'20': 1.4826*3, '0': 1.4826*1.5, '0_005': 1.4826*2, '0_5': 1.4826*2,}
    df_anml = pd.read_csv(os.path.join(ref_files_path, 'SD4.1ReV_Plasma_QC_ANML_200170.txt'), sep='\t')
    df_anml = df_anml.rename(columns={'Reference': 'RefMedianQC'})
    df_anml['RefLog10MedianQC'] = np.log10(df_anml['RefMedianQC'])
    df_anml = pd.merge(left=pd.read_csv(os.path.join(ref_files_path, 'SD4.1ReV_Plasma_ANML.txt'), sep='\t'), right=df_anml, on='SeqId', how='inner')
    df_anml['RefSDLog10'] = np.log10(df_anml['RefSD'])
    df_anml = df_anml.set_index('SeqId')
    # num_anml_iterations = 200
    # min_fraction_to_use = 0.4
    log_multiplier = False
    
    # different factor per dilution (higher for 20 than the other dilutions)
    
    dil_lab= ("20", "0", "0_005", "0_5", )
    RFU_anmlCal = RFU_cal_ref.copy() #RFU_cal.copy()
    RFU_anmlCal_log = RFU_anmlCal.apply(np.log10).copy()
    
    res = {}
    for sampl_type in ['QC', 'Sample']: # Only calibrators according to the paper
        if sampl_type == 'Sample':
            ref_med_col_sampl = 'RefLog10Median'
            ref_sd_col_sampl = 'RefLog10MAD'
        elif sampl_type == 'QC':
            ref_med_col_sampl ='RefLog10MedianQC'
            ref_sd_col_sampl = 'RefSDLog10'
        sel2 = sampl['SampleType'] == sampl_type
        indx_sampl_list = sampl.index[sel2]
        for ind, indx_sampl in enumerate(indx_sampl_list):
            if ind%10 == 0:
                print(ind)
            res[indx_sampl] = {}
            for dil in dil_lab:
                factor = factor_dict[dil]

                for ind in range(num_anml_iterations):
                    somamers_dil = somamers[somamers['Dilution'] == dil].index
                    somamers_dil_in_range = (RFU_anmlCal_log.loc[indx_sampl, somamers_dil] >= (df_anml.loc[somamers_dil, ref_med_col_sampl] - factor*df_anml.loc[somamers_dil, ref_sd_col_sampl])) & (RFU_anmlCal_log.loc[indx_sampl, somamers_dil] <= (df_anml.loc[somamers_dil, ref_med_col_sampl] + factor*df_anml.loc[somamers_dil, ref_sd_col_sampl])) 
                    somamers_dil_in_range = somamers_dil_in_range.loc[somamers_dil_in_range].index
                    fraction_in_range = len(somamers_dil_in_range)/len(somamers_dil)
                    
                    if log_multiplier:
                        rgd_ialpha = RFU_anmlCal_log.loc[indx_sampl, somamers_dil_in_range] - (df_anml.loc[somamers_dil_in_range, ref_med_col_sampl])
                        inv_log_SFgd = round(rgd_ialpha.mean(), 6)
                        SFgd = round(1 / 10 ** inv_log_SFgd, 6)
                    else:
                        rgd_ialpha = RFU_anmlCal.loc[indx_sampl, somamers_dil_in_range] / (10**df_anml.loc[somamers_dil_in_range, ref_med_col_sampl])
                        SFgd = 1 / rgd_ialpha.median()
                    RFU_anmlCal.loc[indx_sampl, somamers_dil] = RFU_anmlCal.loc[indx_sampl, somamers_dil] * SFgd
                    RFU_anmlCal.loc[indx_sampl, somamers_dil] = RFU_anmlCal.loc[indx_sampl, somamers_dil]  #.apply(lambda x: round(x,1))
                    RFU_anmlCal_log.loc[indx_sampl, somamers_dil] = RFU_anmlCal.loc[indx_sampl, somamers_dil].apply(np.log10)
                    if ind == 0:
                        res[indx_sampl][dil] = (fraction_in_range, SFgd)
                    else:
                        iter_res = (fraction_in_range, res[indx_sampl][dil][1]*SFgd)
                        if iter_res == res[indx_sampl][dil]:
                            break
                        res[indx_sampl][dil] = (fraction_in_range, res[indx_sampl][dil][1]*SFgd)
                    if res[indx_sampl][dil][0] >= 1:
                        break

    return RFU_anmlCal, res
