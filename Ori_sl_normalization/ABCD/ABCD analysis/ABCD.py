import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

def transform_concentration(stri:str):
    stri2 = stri.split()
    units2=stri2[-1][0]
    multiplier = 10 if units2=='n' else 10**4 if units2=='µ' else 10**7 if units2=='m' else 0
    c = float(stri2[0])*multiplier
    return c


plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Aptos"
lw = plt.rcParams['lines.linewidth']

# define params
latest_date_str = '20250629' # '20250518'
refrence_method = 'EDTA plasma'
load_file = True # False # True

# define aptamer groups and params
sheba_df = pd.read_csv('data/SeqId Annotations Secreted & Sheba Filters.csv',index_col=0)
raps_df = pd.read_csv('data/v2_raps_extended_metadata.csv',index_col=0)

aptamers_sheba = sheba_df.index[sheba_df['Sheba p-value']>0.05].to_list()
aptamers_7k = sheba_df.index[(sheba_df['Organism']=='Human')&(sheba_df['Type']=='Protein')].to_list()
aptamers_raps = raps_df.index.to_list()

aptamers_group_names = ['all 7K','Sheba-filtered','388 RAPs']
aptamers_groups = [aptamers_7k,aptamers_sheba,aptamers_raps]
best_range_dict = {"Pearson's r":[0.8,1],"Spearman's r":[0.8,1],'R^2':[0.8,1],'Slope':[0.75,1.25],'Intercept':[-2,2],"Wald's P-value":[0,0.05],"Wald's Q-value":[0,0.05]}
xlim_dict = {"Pearson's r":[0,1],"Spearman's r":[0,1],'R^2':[0,1],'Slope':[0,2],'Intercept':[-10,10],"Wald's P-value":[0,1],"Wald's Q-value":[0,1]}
tested_methods_list = ['EDTA plasma (consecutive runs)','CPT Plasma','STRECK-PROT+ DS','STRECK-cfDNA DS','STRECK-PROT+','Serum']

tab10_colors = plt.get_cmap('tab10')
colors_dict = {mthd:tab10_colors(num) for num,mthd in enumerate(tested_methods_list)}
colors_dict.update({grp:tab10_colors(len(tested_methods_list)+num) for num,grp in enumerate(aptamers_group_names)})

# load aptamers annotations file
concentrations_df = pd.read_csv('data/proteins_concentrations.csv',index_col=0)
concentrations_df['Concentration'] = concentrations_df['Concentration'].apply(transform_concentration)
annotations_df = pd.read_excel('data/SL00000571_SomaScan_7K_v4.1_Plasma_Serum_Annotated_Menu.xlsx', skiprows = 8, index_col=0, sheet_name='Annotations')[['Target Name','Dilution','Correlation Plasma']]

# load data
adats_path = r'C:\Users\RoeeOrland\Oncohost DX\Shares - R&D\Data Analysis\Experiments\ADAT_extraction\Output' # r'~/Oncohost DX/Shares - R&D/Data Analysis/Experiments/ADAT_extraction/Output'
samples_fname = '_adat_data_samples_with_sample_info.csv'
measurements_fname = '_adat_data_measurements.csv'
labguru_fname = '_samples_from_labguru.xlsx'
get_fname = lambda fnme: os.path.join(os.path.expanduser(adats_path),f'{latest_date_str}{fnme}')
samples_df = pd.read_csv(get_fname(samples_fname), index_col=0, low_memory=False)
measurements_df = pd.read_csv(get_fname(measurements_fname), index_col=0, low_memory=False)
labguru_df = pd.read_excel(get_fname(labguru_fname), index_col=1)
serum_results_fname = 'data/Supplementary Table S9 - Correction factors by cohort.csv'
serum_results_df = pd.read_csv(serum_results_fname, index_col=0, low_memory=False)
# load plasma stability measurements from analytical validity
plasma_consecutive_fname = 'data/classifier_df_for_dry03.csv'
stability_pairs_fname = 'data/stability_pairs.csv'
plasma_consecutive_df = pd.read_csv(plasma_consecutive_fname, index_col=0,low_memory=False)
stability_pairs_df = pd.read_csv(stability_pairs_fname,index_col=0)
abcd_clinical_fname = 'data/ONCOHOST Sample Manifest _07APR2025.xlsx'
abcd_clinical_df = pd.read_excel(abcd_clinical_fname, skiprows=5)

# transform data
samples_df = samples_df.loc[(pd.to_datetime(samples_df['DateOfRun']).dt.year==2025)&(samples_df['Indication']=='HEALTHY')]
labguru_df = labguru_df.loc[samples_df['Optional1'].unique()]
labguru_df['SubjectId'] = labguru_df['sample_name'].apply(lambda stri: stri.split('_')[1])
samples_df = samples_df.rename({'SubjectId':'Adat_SubjectId'},axis=1).join(labguru_df[['material_collected','SubjectId']],on='Optional1')
samples_df['material_collected'] = samples_df['material_collected'].str.replace('STRECK-PROT+DS','STRECK-PROT+ DS').str.replace('STRECK-cfDNA-DS','STRECK-cfDNA DS')
measurements_df = measurements_df.loc[samples_df.index]
abcd_clinical_df = abcd_clinical_df.iloc[1:,:]
abcd_clinical_df['Oncohost Donor ID'] = abcd_clinical_df['Oncohost Donor ID'].str[-4:]
abcd_clinical_df = abcd_clinical_df.loc[abcd_clinical_df['Additional Donor Info']=='EDTA'].rename({'Gender':'Sex'},axis=1).rename({'Oncohost Donor ID':'SubjectId'},axis=1).set_index('SubjectId')

# transform serum data
serum_results_df = serum_results_df[[col for col in serum_results_df.columns if '_CohortA' in col]]
serum_results_df = serum_results_df.rename({col:col.replace('_CohortA','') for col in serum_results_df.columns}, axis=1).rename({'Pearson':"Pearson's r",'Spearman':"Spearman's r"},axis=1)
serum_results_df['R^2'] = serum_results_df["Pearson's r"].apply(lambda num: num**2)
serum_results_df = serum_results_df[["Pearson's r","Spearman's r",'R^2','Slope','Intercept']]
serum_results_df['Method']='Serum'

# transform plasma stability
for col in ['SubjectId','Sample1','Sample2']:
    stability_pairs_df[col] = stability_pairs_df[col].str.replace('OH','IL').str.replace('-T0','')
stability_pairs_df = stability_pairs_df.groupby('SubjectId').first().reset_index(drop=False)
plasma_consecutive_df.index = plasma_consecutive_df.index.str.replace('OH','IL').str.replace('-T0','')
plasma_consecutive_df = plasma_consecutive_df.loc[plasma_consecutive_df.index.isin(stability_pairs_df.Sample1.to_list()+stability_pairs_df.Sample2.to_list())]
plasma_df_tmp1 = stability_pairs_df[['SubjectId','Sample1','Plate1','Date1']].rename({'Sample1':'MeasureId','Plate1':'PlateCode','Date1':'DateOfRun'},axis=1).set_index('MeasureId')
plasma_df_tmp1['material_collected'] = 'EDTA plasma'
plasma_df_tmp2 = stability_pairs_df[['SubjectId','Sample2','Plate2','Date2']].rename({'Sample2':'MeasureId','Plate2':'PlateCode','Date2':'DateOfRun'},axis=1).set_index('MeasureId')
plasma_df_tmp2['material_collected'] = 'EDTA plasma (consecutive runs)'
samples_df = pd.concat([samples_df,plasma_df_tmp1,plasma_df_tmp2],axis=0)
measurements_df = pd.concat([measurements_df,plasma_consecutive_df],axis=0)

# # save files
samples_df.iloc[:,-2:].join(abcd_clinical_df, on='SubjectId').join(measurements_df).to_csv('abcd_data.csv')

# log2
for prot in measurements_df.columns:
    measurements_df[prot] = measurements_df[prot].apply(np.log2)

if load_file==False:
    # run analysis and figure plots for each tested_method compared with refrence_method
    stats_df_dict={}
    best_range_percent_df_ls=[]
    for tested_method in tested_methods_list:
        # calculate correlation measures (or get from file if tested_method is "Serum")
        if tested_method=='Serum':
            stats_df = serum_results_df
        else:
            stats_df = pd.DataFrame(index=measurements_df.columns, columns = ["Pearson's r","Spearman's r",'R^2','Slope','Intercept',"Wald's P-value","Wald's Q-value",'Method'], dtype=float)
            stats_df['Method'] = tested_method
            for prot in measurements_df.columns:
                data_a = measurements_df.loc[samples_df['material_collected']==refrence_method,[prot]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
                data_b = measurements_df.loc[samples_df['material_collected']==tested_method,[prot]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
                data_a = data_a.loc[data_b.index]  
                ## added by Ro'ee:
                data_a = data_a[~data_a.index.duplicated(keep='first')]              
                ## data_a: protein values for the reference method (EDTA) FOR SAMPLES COLLECTED ALSO WITH TESTED METHOD (one of the 6 methods) 
                tmp_corr_df = pd.concat([data_a,data_b],axis=1)
                stats_df.loc[prot,"Pearson's r"] = tmp_corr_df.corr(method='pearson').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='pearson')
                stats_df.loc[prot,"Spearman's r"] = tmp_corr_df.corr(method='spearman').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='spearman')
                # reg = LinearRegression()
                # reg = reg.fit(X=data_a, y=data_b)
                # stats_df.loc[prot,'Intercept'] = reg.intercept_
                # stats_df.loc[prot,'Slope'] = reg.coef_[0]
                # stats_df.loc[prot,'R^2'] = reg.score(data_a, data_b)
                res = linregress(x=data_a[prot], y=data_b[prot], alternative='greater')
                stats_df.loc[prot,'Intercept'] = res.intercept
                stats_df.loc[prot,'Slope'] = res.slope
                stats_df.loc[prot,'R^2'] = res.rvalue**2
                stats_df.loc[prot,"Wald's P-value"] = res.pvalue
            res = multipletests(stats_df["Wald's P-value"],method='fdr_bh')
            stats_df["Wald's Q-value"] = res[1]
                
            print(f"{tested_method}: {data_a.shape[0]} subjects")
        stats_df_dict.update({tested_method:stats_df.copy()})

        best_range_percent_df_method = pd.DataFrame(columns=stats_df.columns.to_list(), index=aptamers_group_names)
        best_range_percent_df_method['Method'] = tested_method
        for group_name,group in zip(aptamers_group_names,aptamers_groups):
            for stat in stats_df.columns:
                if stat=='Method':
                    continue
                plt_df = stats_df.loc[group,stat].copy()
                best_range = best_range_dict[stat]
                best_range_percent = round(100*plt_df.between(left=best_range[0], right=best_range[1], inclusive='both').apply(int).mean())
                ## best_range_percent: for the group (7K, Sheba, RAPs) and statistic (Pearson's, Spearman's, slope, intercept, R^2, p-value, q-value) what fraction of values are in the best range (defined in best_range_dict)
                best_range_percent_df_method.loc[group_name,stat] = best_range_percent
        best_range_percent_df_ls.append(best_range_percent_df_method.copy().reset_index(drop=False).rename({'index':'Aptamers'},axis=1).set_index('Aptamers'))
    best_range_percent_df = pd.concat(best_range_percent_df_ls, axis=0)
    stats_df_all = pd.concat(list(stats_df_dict.values()), axis=0)
    best_range_percent_df.to_csv('best_range_percent_df.csv')
    stats_df_all.to_csv('stats_df_all.csv')
else:
    best_range_percent_df = pd.read_csv('best_range_percent_df.csv',index_col=0,low_memory=False)
    stats_df_all = pd.read_csv('stats_df_all.csv',index_col=0,low_memory=False)
# Use Somalogic's EDTA plasma consecutive runs correlation
# stats_df_all.loc[stats_df_all['Method']==tested_methods_list[0],"Spearman's r"] = annotations_df.loc[stats_df_all.index[stats_df_all['Method']==tested_methods_list[0]],'Correlation Plasma']

stable_prots={}
for mthd in tested_methods_list:
    # stable_prots_mthd = stats_df_dict[mthd].index[stats_df_dict[mthd]['Intercept'].between(left=best_range_dict['Intercept'][0], right=best_range_dict['Intercept'][1], inclusive='both')&stats_df_dict[mthd]['Slope'].between(left=best_range_dict['Slope'][0], right=best_range_dict['Slope'][1], inclusive='both')].to_list()
    stable_prots_mthd = stats_df_all.index[(stats_df_all["Spearman's r"]>=best_range_dict["Spearman's r"][0])&(stats_df_all.Method==mthd)].to_list()
    stable_prots.update({mthd:stable_prots_mthd})

# methods_to_compare = ['EDTA plasma (consecutive runs)','STRECK-PROT+ DS']
methods_to_compare = ['EDTA plasma (consecutive runs)','STRECK-cfDNA DS']
# methods_to_compare = ['STRECK-PROT+ DS','STRECK-PROT+']

## Figures

c = 0
for stat in best_range_percent_df.columns:
    if stat=='Method':
        continue
    c += 1
    # summary barplot #1
    if stat in ['Slope','Intercept']:
        condition_str = f"between {str(best_range_dict[stat])}"
    elif stat=="Wald's P-value":
         condition_str = f"below {best_range_dict[stat][1]}"
    else:
        condition_str = f"above {best_range_dict[stat][0]}"
    
    plt_df = best_range_percent_df[[stat,'Method']].reset_index(drop=False)
    palette = [colors_dict[grp] for grp in aptamers_group_names]
    g = sns.barplot(plt_df,y=stat, x='Method', order=tested_methods_list, hue='Aptamers', hue_order=aptamers_group_names, palette=palette)
    g.set_xticklabels([val._text for val in g.get_xticklabels()],rotation=30)
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle(stat)
    plt.ylabel(f'Percent of aptamers {condition_str}')
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.savefig(f"results/fig{c}.0 - {stat} in best range (by method).png")
    plt.close()
    
    # summary barplot #2
    palette = [colors_dict[mthd] for mthd in tested_methods_list]
    g = sns.barplot(plt_df,y=stat, x='Aptamers', order=aptamers_group_names, hue='Method', hue_order=tested_methods_list, palette=palette)
    plt.suptitle(stat)
    plt.ylabel(f'Percent of aptamers {condition_str}')
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"results/fig{c}.1 - {stat} in best range (by aptamers group).png")
    plt.close()

# CDF/KDE per stat per aptamers group with 5 lines each, describing the 5 methods.
plt.rcParams['lines.linewidth'] = 2
# smooth=False
for group_name, group in zip(aptamers_group_names,aptamers_groups):
    for stat in stats_df_all.columns:
        if stat=='Method':
            continue
        if 'Q' in stat:
            st=0
        for methods100 in[tested_methods_list,methods_to_compare]:
            c+=1
            palette = [colors_dict[mthd] for mthd in methods100]
            plt_df = stats_df_all.loc[group,[stat,'Method']]
            plt_df = plt_df.loc[plt_df.Method.isin(methods100)].reset_index()
            method_str = 'select methods' if len(methods100)>2 else methods100[1]
            if stat in ["Pearson's r","Spearman's r",'R^2',"Wald's P-value","Wald's Q-value"]:
                figname = f"{stat} CDF between {refrence_method} and {method_str} among {group_name}"
                ax = sns.ecdfplot(data=plt_df,x=stat, hue='Method', hue_order=methods100, zorder=10, palette=palette)
                ylimits = list(ax.get_ylim())
                xlimits = xlim_dict[stat]
                dashed_ylimits=ylimits

                if len(methods100)<=3:
                    y80_ls = []
                    for num in range(len(methods100)):
                        y80 = (plt_df.loc[plt_df['Method']==methods100[num],stat]<best_range_dict[stat][0]).apply(int).mean()
                        ax = sns.lineplot(x=[xlim_dict[stat][0],best_range_dict[stat][0]],y=[y80,y80], color='k', linestyle='--',linewidth=1.5, ax=ax)
                        y80_ls.append(y80)
                    dashed_ylimits = [ylimits[0],max(y80_ls)]
                plt.ylabel('Aptamers CDF')
                if stat=="Wald's P-value":
                    plt.plot([best_range_dict[stat][1],best_range_dict[stat][1]], dashed_ylimits, '--k', linewidth=1.5)
            else:
                figname = f"{stat} PDF between {refrence_method} and {method_str} among {group_name}"
                ax = sns.histplot(data=plt_df,x=stat, hue='Method', hue_order=methods100, element='poly', common_norm=False, stat='probability', palette=palette)
                plt.ylabel('Aptamers PDF')
                ylimits = list(ax.get_ylim())
                dashed_ylimits=ylimits
                plt.plot([best_range_dict[stat][1],best_range_dict[stat][1]], dashed_ylimits, '--k', linewidth=1.5)
            plt.plot([best_range_dict[stat][0],best_range_dict[stat][0]], dashed_ylimits, '--k', linewidth=1.5)

            plt.xlabel(stat)
            plt.xlim(xlim_dict[stat])
            ax.set_ylim(ylimits)
            plt.suptitle(figname)
            sns.move_legend(ax, "upper center", bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(f"results/fig{c} - {figname}.png")
            plt.close()
        
plt.rcParams['lines.linewidth'] = lw


# # scatter to compare selected methods
# for group,group_name in zip(aptamers_groups,aptamers_group_names):
#     for plot_kind in ['reg']:#['scatter','KDE','reg']:
#         for stat in stats_df_all.columns:
#             if stat=='Method':
#                 continue
#             c+=1
#             stats_a = stats_df_all.loc[stats_df_all.Method==methods_to_compare[0]].loc[group,[stat]].rename({stat:methods_to_compare[0]},axis=1)
#             stats_b = stats_df_all.loc[stats_df_all.Method==methods_to_compare[1]].loc[group,[stat]].rename({stat:methods_to_compare[1]},axis=1)
#             group_df = pd.DataFrame(data='Other',columns=['Aptamers Group'], index=stats_a.index)
#             for num in range(len(aptamers_group_names)):
#                 group_df.loc[group_df.index.isin(aptamers_groups[num])] = aptamers_group_names[num]
            
            
#             plt_df = pd.concat([stats_a,stats_b,group_df], axis=1)
#             marginal_kws= {'common_norm':False}
#             joint_kws = {'color':'.3','line_kws':{'color':'r'}}#{'zorder':0}

#             g = sns.jointplot(data=plt_df, x=methods_to_compare[0], y=methods_to_compare[1], xlim=xlim_dict[stat], ylim=xlim_dict[stat], kind=plot_kind.lower(), marker='x', marginal_kws=marginal_kws, joint_kws=joint_kws)
#             pk=plot_kind
#             plot_kind = plot_kind.replace('reg','scatter')
#             # if plot_kind=='scatter':
#             #     legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
#             #     for num in range(1,len(aptamers_group_names)):
#             #         g.ax_joint = sns.scatterplot(data=plt_df.loc[plt_df['Aptamers Group']==aptamers_group_names[num]], x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, marker='+', zorder=num, ax=g.ax_joint)#, x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, xlim=xlim, ylim=xlim)
#             #     g.ax_joint.legend(legend_handles, legend_labels)

#             plt.suptitle(f"{stat} {plot_kind} among {group_name}")
#             plt.subplots_adjust(top=0.95, left=0.15)
#             plt.savefig(f"results/fig{c} - {stat} comparison {plot_kind} between {methods_to_compare[0]} and {methods_to_compare[1]} among {group_name}.png")
#             plt.close()
#             plot_kind=pk

chosen_method = methods_to_compare[1]
stable_prots_to_check = set(stable_prots[chosen_method]).intersection(aptamers_groups[1])
for tested_method in methods_to_compare:
    c+=1
    plt_df = stats_df_all.loc[(stats_df_all.Method==tested_method)&(stats_df_all.index.isin(stable_prots_to_check)),['Slope','Intercept']]
    g = sns.jointplot(data=plt_df, x='Intercept',y='Slope', ratio=2, kind='scatter', marker='x')
    plt.suptitle(f"Regression factors of {tested_method}")
    plt.xlim(xlim_dict['Intercept'])
    plt.ylim(xlim_dict['Slope'])
    plt.subplots_adjust(top=0.95, left=0.15)

    plt.savefig(f"results/fig{c} - regression factors between {tested_method} and {refrence_method} among stable aptamers in {chosen_method}.png")
    plt.close()


# investors slide correlation scatter of 1 RAP
selected_aptamers = ['2573-20','3580-25','13113-7','9021-1','13701-2']
stats_to_check = ["Pearson's r","Spearman's r"]
log2_str = '(LOG2(RFU))'
for apt in selected_aptamers:
    data_a = samples_df.loc[samples_df['material_collected']==refrence_method].iloc[:,-2:].rename({'material_collected':'Method'},axis=1).join(measurements_df[apt].apply(np.log2)).set_index('SubjectId').sort_index()
    data_b = samples_df.loc[samples_df['material_collected']==chosen_method].iloc[:,-2:].rename({'material_collected':'Method'},axis=1).join(measurements_df[apt].apply(np.log2)).set_index('SubjectId').sort_index()
    data_a = data_a.loc[data_b.index]
    ## Ro'ee added:
    data_a = data_a[~data_a.index.duplicated(keep='first')]  
    ## End of Ro'ee added   
    plt_df = pd.concat([data_a.rename({apt:f"{refrence_method} {log2_str}"},axis=1),data_b.rename({apt:f"{chosen_method} {log2_str}"},axis=1)],axis=1)

    apt_name = f"{raps_df.loc[apt,'Target Full Name']} ({raps_df.loc[apt,'Entrez Gene Name']})"
    stat_str_ls=[f"{stat} = {round(stats_df_all.loc[stats_df_all['Method']==chosen_method].loc[apt,stat],2)}" for stat in stats_to_check]
    stat_str = " ; ".join(stat_str_ls)
    if len(stats_to_check)>0:
        stat_str = "\n"+stat_str
    
    ax = sns.regplot(data=plt_df, x=f"{refrence_method} {log2_str}", y=f"{chosen_method} {log2_str}")
    ax.set_aspect('equal')
    xlimits = ax.get_xlim()
    ylimits = ax.get_ylim()
    new_lims = [min(xlimits[0],ylimits[0]),max(xlimits[1],ylimits[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    plt.subplots_adjust(bottom=0.15)
    plt.suptitle(apt_name+stat_str)

    plt.savefig(f"results/fig{c} - regression scatter of {chosen_method} vs {refrence_method} for {apt}.png")
    plt.close()
        

# prophet: histogram of PROphet scores for the healthy EDTA samples

# scatter of Slope vs. Calculated concentration (concentration*dilution)
all_subjects = samples_df.SubjectId.sort_values().unique().tolist()[:20]
annotations_df = annotations_df.loc[aptamers_7k]
annotations_df = annotations_df.join(raps_df['Repeats']).join(concentrations_df['Concentration'], on='Target Name').join(stats_df_all.loc[stats_df_all.Method==methods_to_compare[1]])
annotations_df['Repeats'] = annotations_df['Repeats'].fillna(0)

annotations_df['Dilution'] = annotations_df['Dilution'].apply(lambda stri: float(stri[:-1])/100)
annotations_df['Diluted Concentration'] = annotations_df['Concentration']*annotations_df['Dilution']

pca = PCA(n_components=1)
pca_df = measurements_df.loc[(samples_df['material_collected']=='EDTA plasma')&(samples_df['SubjectId'].isin(all_subjects)),aptamers_7k]
pca_df_mean = pca_df.mean()
pca_df_std = pca_df.std()
for prot in pca_df.columns:
    pca_df[prot] = (pca_df[prot]-pca_df_mean[prot])/pca_df_std[prot]
pca.fit(pca_df)
annotations_df.loc[aptamers_7k,'PCA component']=pca.components_[0,:]#**2
annotations_df['Mean Log2(RFU)'] = measurements_df.loc[samples_df['material_collected']=='EDTA plasma'].mean()
annotations_df['Median Log2(RFU)'] = measurements_df.loc[samples_df['material_collected']=='EDTA plasma'].median()
annotations_df['STD of Log2(RFU)'] = measurements_df.loc[samples_df['material_collected']=='EDTA plasma'].std()

# ax = sns.pairplot(data=annotations_df.loc[annotations_df.index.isin(aptamers_sheba)],vars=["Spearman's r",'Correlation Plasma'], kind='reg', corner=True, plot_kws={'marker':'+'})
# ax = sns.jointplot(data=annotations_df.loc[annotations_df.index.isin(aptamers_sheba)],x='Correlation Plasma',y="Spearman's r", marker='+', kind='reg')
# ax.ax_joint.set_aspect('equal', adjustable='datalim')
# plt.show()

annotations_df['Aptamers group'] = annotations_df.reset_index(drop=False)['SeqId'].isin(aptamers_sheba).map({True:aptamers_group_names[1],False:aptamers_group_names[0]}).values
grp_order = aptamers_group_names[:2]
for stat in stats_df_all.columns:
    if stat=='Method':
        continue
    c += 1
    cc = 0
    plt_df = annotations_df.loc[annotations_df.index.isin(aptamers_7k)]
    ax = sns.regplot(data=plt_df.dropna(subset=['Concentration']), x='Concentration',y=stat, marker='+', logx=True, scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]].dropna(subset=['Concentration']), x='Concentration',y=stat, hue='Aptamers group', hue_order=grp_order, marker='+', ax=ax, legend=True, linewidth=1)
    plt.xscale('log')
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs Concentration.png")
    plt.close()

    ax = sns.regplot(data=plt_df.dropna(subset=['Diluted Concentration']), x='Diluted Concentration',y=stat, marker='+', logx=True, scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]].dropna(subset=['Diluted Concentration']), x='Diluted Concentration',y=stat, hue='Aptamers group', hue_order=grp_order, marker='+',ax=ax, legend=True, linewidth=1)
    plt.xscale('log')
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs Diluted Concentration.png")
    plt.close()

    ax = sns.regplot(data=plt_df, x='PCA component',y=stat, marker='+', scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]], x='PCA component',y=stat, hue='Aptamers group', hue_order=grp_order, marker='+', ax=ax, legend=True, linewidth=1)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs PCA component.png")
    plt.close()

    # boxplot of Slope for each dilution group
    ax = sns.boxplot(data=plt_df, x='Dilution',y=stat)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs Dilution.png")
    plt.close()


    # scatter of Slope vs. Median/Mean/STD of RFU
    ax = sns.regplot(data=plt_df, x='Mean Log2(RFU)', y=stat, marker='+', scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]], x='Mean Log2(RFU)', y=stat, hue='Aptamers group', hue_order=grp_order, marker='+', ax=ax, legend=True, linewidth=1)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs average RFU.png")
    plt.close()

    ax = sns.regplot(data=plt_df, x='Median Log2(RFU)', y=stat, marker='+', scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]], x='Median Log2(RFU)', y=stat, hue='Aptamers group', hue_order=grp_order, marker='+', ax=ax, legend=True, linewidth=1)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs median RFU.png")
    plt.close()

    ax = sns.regplot(data=plt_df, x='STD of Log2(RFU)', y=stat, marker='+', scatter_kws={"linewidths": 1})
    ax = sns.scatterplot(data=plt_df.loc[annotations_df['Aptamers group']==grp_order[1]], x='STD of Log2(RFU)', y=stat, hue='Aptamers group', hue_order=grp_order, marker='+', ax=ax, legend=True, linewidth=1)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs STD of RFU.png")
    plt.close()


    # prophet: scatter between RAP repetition and correlation
    th_ls=[1,10,40,81]
    plt_df = annotations_df.loc[aptamers_raps,['Repeats',stat]]
    plt_df['Repetitions'] = ''
    for num in range(len(th_ls)-1):
        num_range_str = f'{th_ls[num]}-{th_ls[num+1]-1}'
        plt_df.loc[plt_df['Repeats'].between(left=th_ls[num], right=th_ls[num+1],inclusive='left'),'Repetitions']=num_range_str

    sns.scatterplot(data=plt_df, x='Repeats',y=stat, marker='+', linewidth=1)
    plt.xlabel('Repetitions')
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs RAP Repetitions.png")
    plt.close()
    
    rep_order = plt_df.sort_values('Repeats')['Repetitions'].unique().tolist()
    sns.boxplot(plt_df,x='Repetitions',y=stat, order=rep_order)
    plt.subplots_adjust(bottom=0.12)
    cc+=1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs RAP Repetitions group.png")
    plt.close()

    th_ls=[1,10,20,30,40,50,60,70,81]
    plt_df['Repetitions'] = ''
    for num in range(len(th_ls)-1):
        plt_df.loc[plt_df['Repeats'].between(left=th_ls[num], right=th_ls[num+1],inclusive='left'),'Repetitions']=f'{th_ls[num]}-{th_ls[num+1]-1}'
    any_above = (plt_df[stat]>best_range_dict[stat][1]).apply(int).sum()>0
    any_below = (plt_df[stat]<best_range_dict[stat][0]).apply(int).sum()>0
    str_good = f"Between [{best_range_dict[stat][0]},{best_range_dict[stat][1]}]" if any_above and any_below else f"Above {best_range_dict[stat][0]}" if any_below else f"Below {best_range_dict[stat][1]}"
    str_bad = f"Out of [{best_range_dict[stat][0]},{best_range_dict[stat][1]}]" if any_above and any_below else f"Below {best_range_dict[stat][0]}" if any_below else f"Above {best_range_dict[stat][1]}"
    
    plt_df[stat] = plt_df[stat].between(left=best_range_dict[stat][0],right=best_range_dict[stat][1]).map({True:str_good,False:str_bad})
    vc = plt_df[['Repetitions',stat]].value_counts().reset_index(drop=False).set_index('Repetitions')
    vc1 = vc.loc[vc[stat]==str_good,['count']].rename({'count':str_good},axis=1)
    vc2 = vc.loc[vc[stat]==str_bad,['count']].rename({'count':str_bad},axis=1)
    vc3 = pd.concat([vc1,vc2],axis=1).sort_index().fillna(0)

    
    ax1 = vc3.plot.bar(stacked=True, color=['tab:blue', 'tab:red'])
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h, l,loc='upper center')
    ax2 = ax1.twinx()
    ax1.set_ylabel('Number of RAPs')
    vc3[f"Percentage {str_good.lower()}"] = [100*vc3.loc[idx,str_good]/(vc3.loc[idx,str_good]+vc3.loc[idx,str_bad]) for idx in vc3.index]
    vc3[f"Percentage {str_good.lower()}"].plot.line(color='k', ax=ax2)
    ax2.set_ylabel(f'Percent of RAPs {str_good.lower()}')
    ax2.set_ylim([0,101])
    plt.subplots_adjust(bottom=0.2)
    plt.title(f"{stat} by RAPs repetition")    
    cc += 1
    plt.savefig(f"results/fig{c}.{cc} - {stat} of {chosen_method} vs RAP Repetitions quantile.png")
    plt.close()





# bootstrap sets of 10 patients of a single aptamer to see how mean and std of Slope and Intercept are affected
selected_aptamer_to_test_ls=[]
for p in [0.74,0.75,0.76]:
    selected_aptamer_to_test1 = stats_df_all.loc[stats_df_all.Method==methods_to_compare[1],'Intercept'].sort_values().index[round(0.75*stats_df_all.loc[stats_df_all.Method==methods_to_compare[1],'Slope'].shape[0])]
    selected_aptamer_to_test2 = stats_df_all.loc[stats_df_all.Method==methods_to_compare[1],'Slope'].sort_values().index[round(0.25*stats_df_all.loc[stats_df_all.Method==methods_to_compare[1],'Slope'].shape[0])]
    selected_aptamer_to_test_ls.extend([selected_aptamer_to_test1,selected_aptamer_to_test2])
c += 1
cc = 0
seed = 0
for selected_aptamer_to_test in selected_aptamer_to_test_ls:
    boot_df = pd.DataFrame(columns=stats_df_all.columns[:-1],index=range(len(stable_prots_to_check)), dtype=float)
    for tested_method in methods_to_compare:
        for k in range(len(stable_prots_to_check)):
            data_a = measurements_df.loc[(samples_df['material_collected']=='EDTA plasma'),[selected_aptamer_to_test]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
            data_b = measurements_df.loc[(samples_df['material_collected']==tested_method),[selected_aptamer_to_test]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
            data_a = data_a.loc[data_b.index]
            seed+=1
            random.seed(seed)
            pat_sample = random.choices(data_a.index.to_list(),k=10)
            data_a=data_a.loc[pat_sample]
            data_b = data_b.loc[pat_sample]
            
            tmp_corr_df = pd.concat([data_a,data_b],axis=1)
            boot_df.loc[k,"Pearson's r"] = tmp_corr_df.corr(method='pearson').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='pearson')
            boot_df.loc[k,"Spearman's r"] = tmp_corr_df.corr(method='spearman').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='spearman')

            # reg = LinearRegression()
            # reg = reg.fit(X=data_a, y=data_b)
            # boot_df.loc[k,'Intercept'] = reg.intercept_
            # boot_df.loc[k,'Slope'] = reg.coef_[0]
            # boot_df.loc[k,'R^2'] = reg.score(data_a, data_b)
            res = linregress(x=data_a[selected_aptamer_to_test], y=data_b[selected_aptamer_to_test], alternative='greater')
            boot_df.loc[k,'Intercept'] = res.intercept
            boot_df.loc[k,'Slope'] = res.slope
            boot_df.loc[k,'R^2'] = res.rvalue**2
            boot_df.loc[k,"Wald's P-value"] = res.pvalue
        res = multipletests(boot_df["Wald's P-value"],method='fdr_bh')
        boot_df["Wald's Q-value"] = res[1]

        cc+=1
        g = sns.jointplot(data=boot_df[['Intercept','Slope']], x='Intercept',y='Slope', ratio=2, kind='scatter', marker='x')
        g.ax_marg_x.axvline(x=stats_df_all.loc[(stats_df_all.Method==tested_method)].loc[selected_aptamer_to_test,'Intercept'], ls="--", color="k", label="wo bootstrapping")
        g.ax_marg_y.axhline(y=stats_df_all.loc[(stats_df_all.Method==tested_method)].loc[selected_aptamer_to_test,'Slope'], ls="--", color="k", label="wo bootstrapping")

        plt.suptitle(f"Regression factors bootstrapping {selected_aptamer_to_test}")
        plt.xlim(xlim_dict['Intercept'])
        plt.ylim(xlim_dict['Slope'])
        plt.subplots_adjust(top=0.95, left=0.15)

        plt.savefig(f"results/fig{c}.{cc} - bootstrapped regression factors between {tested_method} and {refrence_method} among stable aptamers in {chosen_method}.png")
        plt.close()


print('ALL DONE!!!')
