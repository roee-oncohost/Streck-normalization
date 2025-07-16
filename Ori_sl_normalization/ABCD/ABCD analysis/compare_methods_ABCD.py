import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.dpi"] = 300
# plt.rcParams["font.family"] = "Aptos"

# define params
latest_date_str = '20250629' # '20250427'
refrence_method = 'EDTA plasma'
print('13')
print('hello')
# load data
adats_path = r'C:\Users\RoeeOrland\Oncohost DX\Shares - R&D\Data Analysis\Experiments\ADAT_extraction\Output' # '~\Oncohost DX\Shares - R&D\Data Analysis\Experiments\ADAT_extraction\Output'
samples_fname = '_adat_data_samples_with_sample_info.csv'
measurements_fname = '_adat_data_measurements.csv'
labguru_fname = '_samples_from_labguru.xlsx'
get_fname = lambda fnme: os.path.join(os.path.expanduser(adats_path),f'{latest_date_str}{fnme}')
samples_df = pd.read_csv(get_fname(samples_fname), index_col=0, low_memory=False)
measurements_df = pd.read_csv(get_fname(measurements_fname), index_col=0, low_memory=False)
labguru_df = pd.read_excel(get_fname(labguru_fname), index_col=1)
serum_results_fname = 'data/Supplementary Table S9 - Correction factors by cohort.csv'
serum_results_df = pd.read_csv(serum_results_fname, index_col=0, low_memory=False)

# transform data
samples_df = samples_df.loc[(pd.to_datetime(samples_df['DateOfRun']).dt.year==2025)&(samples_df['Indication']=='HEALTHY')]
labguru_df = labguru_df.loc[samples_df['Optional1'].unique()]
labguru_df['SubjectId'] = labguru_df['sample_name'].apply(lambda stri: stri.split('_')[1])
samples_df = samples_df.rename({'SubjectId':'Adat_SubjectId'},axis=1).join(labguru_df[['material_collected','SubjectId']],on='Optional1')
measurements_df = measurements_df.loc[samples_df.index]
serum_results_df = serum_results_df[[col for col in serum_results_df.columns if '_CohortA' in col]]
serum_results_df = serum_results_df.rename({col:col.replace('_CohortA','') for col in serum_results_df.columns}, axis=1)
serum_results_df['R^2'] = serum_results_df['Pearson'].apply(lambda num: num**2)
serum_results_df = serum_results_df[['Pearson','Spearman','R^2','Slope','Intercept']]

for prot in measurements_df.columns:
    measurements_df[prot] = measurements_df[prot].apply(np.log2)

# define aptamer groups
sheba_df = pd.read_csv('data/SeqId Annotations Secreted & Sheba Filters.csv',index_col=0)
raps_df = pd.read_csv('data/v2_raps_extended_metadata.csv',index_col=0)

aptamers_sheba = sheba_df.index[sheba_df['Sheba p-value']>0.05].to_list()
aptamers_7k = sheba_df.index[(sheba_df['Organism']=='Human')&(sheba_df['Type']=='Protein')].to_list()
aptamers_raps = raps_df.index.to_list()

aptamers_group_names = ['all 7K','Sheba-filtered','388 RAPs']
aptamers_groups = [aptamers_7k,aptamers_sheba,aptamers_raps]
best_range_dict = {'Pearson':[0.8,1],'Spearman':[0.8,1],'R^2':[0.8,1],'Slope':[0.75,1.25],'Intercept':[-2,2]}

# run analysis and figure plots for each tested_method compared with refrence_method
best_range_percent_df_ls=[]
tested_methods_list = ['CPT Plasma','STRECK-PROT+DS','STRECK-cfDNA-DS','STRECK-PROT+','Serum']
cc=0
bins_dict={}
stats_df_dict={}
for tested_method in tested_methods_list:
    # calculate correlation measures (or get from file if tested_method is "Serum")
    if tested_method=='Serum':
        stats_df = serum_results_df
    else:
        stats_df = pd.DataFrame(index=measurements_df.columns, columns = ['Pearson','Spearman','R^2','Slope','Intercept'], dtype=float)
        for prot in measurements_df.columns:
            data_a = measurements_df.loc[samples_df['material_collected']==refrence_method,[prot]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
            data_b = measurements_df.loc[samples_df['material_collected']==tested_method,[prot]].join(samples_df[['SubjectId']]).set_index('SubjectId').sort_index()
            ## Ro'ee added:
            data_a = data_a[~data_a.index.duplicated(keep='first')]  
            ## End of Ro'ee added  

            tmp_corr_df = pd.concat([data_a,data_b],axis=1)
            stats_df.loc[prot,'Pearson'] = tmp_corr_df.corr(method='pearson').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='pearson')
            stats_df.loc[prot,'Spearman'] = tmp_corr_df.corr(method='spearman').iloc[0,1]#data_a.corrwith(data_b[prot].values, method='spearman')

            reg = LinearRegression()
            reg = reg.fit(X=data_a, y=data_b)
            stats_df.loc[prot,'Intercept'] = reg.intercept_
            stats_df.loc[prot,'Slope'] = reg.coef_[0]
            stats_df.loc[prot,'R^2'] = reg.score(data_a, data_b)
    stats_df_dict.update({tested_method:stats_df.copy()})
    c=0
    cc+=1
    best_range_percent_df_method = pd.DataFrame(columns=['Method']+stats_df.columns.to_list(), index=aptamers_group_names)
    best_range_percent_df_method['Method'] = tested_method
    
    for group_name,group in zip(aptamers_group_names,aptamers_groups):
        for stat in stats_df.columns:
            c+=1
            # histogram
            figname = f"{stat} probability between {refrence_method} and {tested_method} among {group_name}"
            plt_df = stats_df.loc[group,stat].copy()
            bins_dict_key = f"{group_name}-{stat}"
            if bins_dict_key in bins_dict.keys():
                bin_edges = bins_dict[bins_dict_key]
            else:
                _, bin_edges = np.histogram(plt_df, bins='auto')
                bins_dict.update({bins_dict_key:bin_edges})
            ax = sns.histplot(data=plt_df, stat='probability', bins=bin_edges)
            plt.xlabel(stat)
            xlim = [-1,1]
            if stat=='Slope':
                xlim=[0,2]
            if stat=='Intercept':
                xlim=[-10,10]
            if stat=='R^2':
                xlim=[0,1]
            plt.xlim(xlim)
            best_range=best_range_dict[stat]
            ylimits = ax.get_ylim()
            
            lw = plt.rcParams['lines.linewidth']
            plt.rcParams['lines.linewidth'] = 1
            ax = sns.lineplot(x=[best_range[0],best_range[0]], y=ylimits, color='r', linestyle='--', ax=ax)
            if stat not in ['Pearson','Spearman','R^2']:
                ax = sns.lineplot(x=[best_range[1],best_range[1]], y=ylimits, color='r', linestyle='--', ax=ax)
                condition_str = f"between {str(best_range)}"
            else:
                condition_str = f"above {best_range[0]}"
            ax.set_ylim(ylimits)
            best_range_percent = round(100*plt_df.between(left=best_range[0], right=best_range[1], inclusive='both').apply(int).mean())
            best_range_percent_df_method.loc[group_name,stat] = best_range_percent
            #plt.title(figname)
            plt.title(f"{tested_method} ({best_range_percent}% {condition_str})")
            plt.savefig(f"results/fig{c}.{cc}.0 - {figname}.png")
            plt.close()
            
            # KDE CDF
            plotkde = False
            if plotkde:#stat in ['Pearson','Spearman','R^2']:
                plt.rcParams['lines.linewidth'] = 2
                figname = f"{stat} CDF between {refrence_method} and {tested_method} among {group_name}"
                ax = sns.kdeplot(data=plt_df, cumulative=True, fill=False)
                plt.xlabel(stat)
                plt.xlim(xlim)
                ylimits = ax.get_ylim()
                ax = sns.lineplot(x=[best_range[0],best_range[0]], y=ylimits, color='r', linestyle='--', ax=ax)
                ax.set_ylim(ylimits)
                plt.title(f"{tested_method} ({best_range_percent}% {condition_str})")
                plt.ylabel('Aptamers CDF')
                plt.savefig(f"results/fig{c}.{cc}.1 - {figname}.png")
                plt.close()
            plt.rcParams['lines.linewidth'] = lw
            
    best_range_percent_df_ls.append(best_range_percent_df_method.copy().reset_index(drop=False).rename({'index':'Aptamers'},axis=1).set_index(['Aptamers','Method']))
print('line 145')
best_range_percent_df = pd.concat(best_range_percent_df_ls, axis=0)
print('line 147')
stable_prots={}
for mthd in tested_methods_list:
    print('line 150')
    stable_prots_mthd = stats_df_dict[mthd].index[stats_df_dict[mthd]['Intercept'].between(left=best_range_dict['Intercept'][0], right=best_range_dict['Intercept'][1], inclusive='both')&stats_df_dict[mthd]['Slope'].between(left=best_range_dict['Slope'][0], right=best_range_dict['Slope'][1], inclusive='both')].to_list()
    stable_prots.update({mthd:stable_prots_mthd})
stable_prots_wo_DS = stable_prots['STRECK-PROT+']
print('154')
for stat in best_range_percent_df.columns:
    print(f"c: {c}")
    c+=1
    # summary barplot #1
    if stat not in ['Pearson','Spearman','R^2']:
        condition_str = f"between {str(best_range_dict[stat])}"
    else:
        condition_str = f"above {best_range_dict[stat][0]}"
    
    plt_df = best_range_percent_df[[stat]].reset_index(drop=False)
    sns.barplot(plt_df,y=stat, x='Method', order=tested_methods_list, hue='Aptamers', hue_order=aptamers_group_names)
    plt.title(stat)
    plt.ylabel(f'Percent of aptamers {condition_str}')
    plt.savefig(f"results/fig{c}.0 - {stat} in best range (by method).png")
    plt.close()
    
    # summary barplot #2
    g = sns.barplot(plt_df,y=stat, x='Aptamers', order=aptamers_group_names, hue='Method', hue_order=tested_methods_list)
    g.set_xticklabels([val._text for val in g.get_xticklabels()],rotation=30)
    plt.title(stat)
    plt.ylabel(f'Percent of aptamers {condition_str}')
    plt.savefig(f"results/fig{c}.1 - {stat} in best range (by aptamers group).png")
    plt.close()

# CD/KDE per stat per aptamers group with 5 lines each, describing the 5 methods.
plt.rcParams['lines.linewidth'] = 2
for group_name,group in zip(aptamers_group_names,aptamers_groups):
    for stat in stats_df.columns:
        c+=1
        plt_df = stats_df.loc[group,stat].copy()
        figname = f"{stat} CDF between {refrence_method} and {tested_method} among {group_name}"
        ax = sns.kdeplot(data=plt_df, cumulative=True, fill=False)
        plt.xlabel(stat)
        plt.xlim(xlim)
        ylimits = ax.get_ylim()
        ax = sns.lineplot(x=[best_range[0],best_range[0]], y=ylimits, color='r', linestyle='--', ax=ax)
        ax.set_ylim(ylimits)
        plt.title(f"{tested_method} ({best_range_percent}% {condition_str})")
        plt.ylabel('Aptamers CDF')
        plt.savefig(f"results/fig{c} - {figname}.png")
        plt.close()
        
plt.rcParams['lines.linewidth'] = lw


# scatter to compare selected methods
methods_to_compare = ['STRECK-PROT+DS','STRECK-PROT+']
for plot_kind in ['scatter','KDE']:
    for stat in stats_df_dict[methods_to_compare[0]].columns:
        c+=1
        stats_a = stats_df_dict[methods_to_compare[0]][[stat]].rename({stat:methods_to_compare[0]},axis=1)
        stats_b = stats_df_dict[methods_to_compare[1]][[stat]].rename({stat:methods_to_compare[1]},axis=1)
        group_df = pd.DataFrame(data='Other',columns=['Aptamers Group'], index=stats_a.index)
        for num in range(len(aptamers_group_names)):
            group_df.loc[group_df.index.isin(aptamers_groups[num])] = aptamers_group_names[num]
        
        xlim = [-1,1]
        if stat=='Slope':
            xlim=[0,2]
        if stat=='Intercept':
            xlim=[-10,10]
        if stat=='R^2':
            xlim=[0,1]
        plt_df = pd.concat([stats_a,stats_b,group_df], axis=1)
        marginal_kws={'common_norm':False, 'cumulative':True, 'fill':False} if stat in ['Pearson','Spearman','R^2'] else {'common_norm':False}

        g = sns.jointplot(data=plt_df, x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, xlim=xlim, ylim=xlim, kind=plot_kind.lower(), marker='+', marginal_kws=marginal_kws, joint_kws={'zorder':0}, marginal_ticks=True)
        if plot_kind=='scatter':
            legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
            for num in range(1,len(aptamers_group_names)):
                g.ax_joint = sns.scatterplot(data=plt_df.loc[plt_df['Aptamers Group']==aptamers_group_names[num]], x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, marker='+', zorder=num, ax=g.ax_joint)#, x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, xlim=xlim, ylim=xlim)
            g.ax_joint.legend(legend_handles, legend_labels)

        prob_func = 'CDF' if stat in ['Pearson','Spearman','R^2'] else 'PDF'
        plt.suptitle(f"{stat} {plot_kind} with {prob_func}'s")
        plt.subplots_adjust(top=0.95, left=0.15)
        plt.savefig(f"results/fig{c} - {stat} comparison {plot_kind} between {methods_to_compare[0]} and {methods_to_compare[1]}.png")
        plt.close()


aptamers_groups = [aptamers_groups[0],stable_prots_wo_DS]
aptamers_group_names = [aptamers_group_names[0],'Stable aptamers in STRECK-PROT+']

for plot_kind in ['scatter','KDE','reg']:
    for stat in stats_df_dict[methods_to_compare[0]].columns:
        c+=1
        stats_a = stats_df_dict[methods_to_compare[0]][[stat]].rename({stat:methods_to_compare[0]},axis=1)
        stats_b = stats_df_dict[methods_to_compare[1]][[stat]].rename({stat:methods_to_compare[1]},axis=1)
        group_df = pd.DataFrame(data='Other',columns=['Aptamers Group'], index=stats_a.index)
        for num in range(len(aptamers_group_names)):
            group_df.loc[group_df.index.isin(aptamers_groups[num])] = aptamers_group_names[num]
        
        xlim = [-1,1]
        if stat=='Slope':
            xlim=[0,2]
        if stat=='Intercept':
            xlim=[-10,10]
        if stat=='R^2':
            xlim=[0,1]
        plt_df = pd.concat([stats_a,stats_b,group_df], axis=1)
        marginal_kws={'common_norm':False, 'cumulative':True, 'fill':False} if stat in ['Pearson','Spearman','R^2'] else {'common_norm':False}

        g = sns.jointplot(data=plt_df, x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, xlim=xlim, ylim=xlim, kind=plot_kind.lower().replace('reg','scatter'), marker='+', marginal_kws=marginal_kws, joint_kws={'zorder':0}, marginal_ticks=True)
        if plot_kind=='scatter':
            legend_handles, legend_labels = g.ax_joint.get_legend_handles_labels()
            for num in range(1,len(aptamers_group_names)):
                g.ax_joint = sns.scatterplot(data=plt_df.loc[plt_df['Aptamers Group']==aptamers_group_names[num]], x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, marker='+', zorder=num, ax=g.ax_joint)#, x=methods_to_compare[0], y=methods_to_compare[1], hue='Aptamers Group', hue_order=aptamers_group_names, xlim=xlim, ylim=xlim)
            g.ax_joint.legend(legend_handles, legend_labels)
        prob_func = 'CDF' if stat in ['Pearson','Spearman','R^2'] else 'PDF'
        plt.suptitle(f"{stat} {plot_kind} with {prob_func}'s")
        plt.subplots_adjust(top=0.95, left=0.15)
        plt.savefig(f"results/fig{c} - {stat} comparison {plot_kind} between {methods_to_compare[0]} and {methods_to_compare[1]}.png")
        plt.close()

print('ALL DONE!!!')
