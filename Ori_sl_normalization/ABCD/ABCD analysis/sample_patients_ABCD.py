import pandas as pd
from sklearn.model_selection import train_test_split

seed=42
sample_size=6

abcd_clinical_fname = 'data/ONCOHOST Sample Manifest _07APR2025.xlsx'
abcd_clinical_df = pd.read_excel(abcd_clinical_fname, skiprows=5)

abcd_clinical_df = abcd_clinical_df.iloc[1:,:]
abcd_clinical_df['Oncohost Donor ID'] = abcd_clinical_df['Oncohost Donor ID'].str[-4:]
abcd_clinical_df = abcd_clinical_df.loc[abcd_clinical_df['Additional Donor Info']=='EDTA'].rename({'Gender':'Sex'},axis=1).rename({'Oncohost Donor ID':'SubjectId'},axis=1).set_index('SubjectId')
abcd_clinical_df = abcd_clinical_df.sort_index().iloc[:20,:]
abcd_clinical_df['Age Category'] = abcd_clinical_df['Age'].apply(lambda x:x>abcd_clinical_df['Age'].median())
strata = ['Sex','Age Category']#,'Smoking Status']

abcd_clinical_df = abcd_clinical_df.drop(index=['D006'])
set75, set25 = train_test_split(abcd_clinical_df,test_size=sample_size/20,random_state=seed, stratify=abcd_clinical_df[strata])
print(set25.sort_index().index)

print('ALL DONE!!!')