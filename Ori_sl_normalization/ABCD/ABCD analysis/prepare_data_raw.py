import pandas as pd
import os

normalized_fname = 'abcd_data.csv'

raw_dir = './data/15052025/raw/'
fname1 = '20250515_adat_data_samples.csv'
fname2 = '20250515_adat_data_measurements.csv'


df_norm = pd.read_csv(normalized_fname,index_col=0,low_memory=False)
df1 = pd.read_csv(os.path.join(raw_dir,fname1),index_col=0,low_memory=False)
df2 = pd.read_csv(os.path.join(raw_dir,fname2),index_col=0,low_memory=False)

df_norm = df_norm.loc[df_norm.SubjectId.str.startswith('D0'),df_norm.columns[:2].to_list()+df_norm.columns[14:].to_list()]

plates_list = df1.loc[df_norm.index,'PlateId'].unique()
df1 = df1.loc[df1['PlateId'].isin(plates_list)&(df1['SubType'].isin(['Buffer','Calibrator'])|df1.index.isin(df_norm.index))]
df1 = df1[['SubType','DateOfRun','PlateId','PlatePosition','Barcode','SampleNumber','SysId','SampleId']]
df1 = df1.join(df_norm.iloc[:,:2])
df2=df2.loc[df1.index]

df_norm.to_csv('to SL/abcd_part1_normalized_data.csv')
df1.join(df2).to_csv('to SL/abcd_part1_raw_data.csv')


print('ALL DONE!!!')
