import pandas as pd

filepath ='/Users/dhruv100691/Documents/cs598-health data analysis/project/UCSF_ADNI/'

#extracts the mapping of features from MappedValues to rename ADNI data columns
#mapping is done manually in the file
def create_adni_column_remap_dict():
    rename_dict={}
    adni_image_dict = pd.read_csv(filepath+'MappedValues.csv')
    for i,j in zip(adni_image_dict['FLDNAME'].tolist(),adni_image_dict['TBLNAME'].tolist()):
        if j!= 'UCSFFSX51':
            rename_dict[i] = j
    return rename_dict

def clean_and_merge():
    #reads the Freesurfer output for PPMI images
    ppmi_image_asegs = pd.read_csv(filepath+'aseg_stats.txt',delimiter='\t')
    ppmi_image_asegs=ppmi_image_asegs.rename(columns={'Measure:volume':'RID'})
    #print (ppmi_image_asegs)

    ppmi_image_thickness_lh = pd.read_csv(filepath+'aparc_thickness_lh.txt',delimiter='\t')
    ppmi_image_thickness_lh=ppmi_image_thickness_lh.rename(columns={'lh.aparc.thickness':'RID'})
    #print (ppmi_image_aparc_lh)

    ppmi_image_thickness_rh = pd.read_csv(filepath+'aparc_thickness_rh.txt',delimiter='\t')
    ppmi_image_thickness_rh=ppmi_image_thickness_rh.rename(columns={'rh.aparc.thickness':'RID'})
    #print (ppmi_image_aparc_rh)

    ppmi_image_area_rh = pd.read_csv(filepath+'aparc_area_rh.txt',delimiter='\t')
    ppmi_image_area_rh=ppmi_image_area_rh.rename(columns={'rh.aparc.area':'RID'})
    #print (ppmi_image_aparc_rh)

    ppmi_image_area_lh = pd.read_csv(filepath+'aparc_area_lh.txt',delimiter='\t')
    ppmi_image_area_lh=ppmi_image_area_lh.rename(columns={'lh.aparc.area':'RID'})
    #print (ppmi_image_aparc_rh)

    ppmi_image_volume_rh = pd.read_csv(filepath+'aparc_volume_rh.txt',delimiter='\t')
    ppmi_image_volume_rh=ppmi_image_volume_rh.rename(columns={'rh.aparc.volume':'RID'})
    #print (ppmi_image_aparc_rh)

    ppmi_image_volume_lh = pd.read_csv(filepath+'aparc_volume_lh.txt',delimiter='\t')
    ppmi_image_volume_lh=ppmi_image_volume_lh.rename(columns={'lh.aparc.volume':'RID'})
    #print (ppmi_image_aparc_rh)

    ppmi_image_thicknessstd_lh = pd.read_csv(filepath+'aparc_stdthickness_lh.txt',delimiter='\t')
    ppmi_image_thicknessstd_lh=ppmi_image_thicknessstd_lh.rename(columns={'lh.aparc.thicknessstd':'RID'})
    #print (ppmi_image_thicknessstd_lh)

    ppmi_image_thicknessstd_rh = pd.read_csv(filepath+'aparc_stdthickness_rh.txt',delimiter='\t')
    ppmi_image_thicknessstd_rh=ppmi_image_thicknessstd_rh.rename(columns={'rh.aparc.thicknessstd':'RID'})
    #print (ppmi_image_thicknessstd_rh)

    #Merges all the output files into a single dataframe
    ppmi_image_stats = pd.merge(ppmi_image_asegs,ppmi_image_thickness_lh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_thickness_rh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_area_rh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_area_lh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_volume_rh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_volume_lh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_thicknessstd_lh)
    ppmi_image_stats = pd.merge(ppmi_image_stats,ppmi_image_thicknessstd_rh)
    ppmi_image_stats.insert(0,'COLPROT','PPMI')
    ppmi_image_stats.insert(2,'Diagnosis','PD')

    ppmi_image_stats.to_csv('merged_ppmi_stats.csv')
    #read the adni preprocessed image data and rename feature columns
    adni_image_stats = pd.read_csv(filepath+'UCSFFSX51_08_01_16.csv')
    adni_image_stats = select_relevant_adni_patients(adni_image_stats)
    adni_image_stats = adni_image_stats.rename(columns=create_adni_column_remap_dict())
    #print(adni_image_stats.dtypes)

    #making the data type same for both adni and ppmi, Maybe not required
    for col in adni_image_stats.columns:
        if col in ppmi_image_stats.columns:
            adni_image_stats[col] = adni_image_stats[col].astype(ppmi_image_stats[col].dtype)
    adni_image_stats.to_csv('modified_adni.csv')
    #print(adni_image_stats.info())

    #creating a final merged table of all patients and features
    merged_patient_stats=pd.concat([ppmi_image_stats,adni_image_stats],ignore_index=True,join='inner')
    print (merged_patient_stats.info())
    merged_patient_stats.to_csv('merged_pateint_stats.csv')
    return merged_patient_stats

def select_relevant_adni_patients(adni_df):
    #selecting only baseline images
    adni_df = adni_df[adni_df['COLPROT'].isin(['ADNI2','ADNIGO'])]
    adni_df = adni_df[adni_df['VISCODE'].isin(['sc','scmri','bl','v01','v02','v03'])]
    adni_df = adni_df[(adni_df['IMAGETYPE'] == 'Non-Accelerated T1')
                      & (adni_df['OVERALLQC'] == 'Pass') & (adni_df['STATUS'] == 'complete')]

    # only choosing baseline diagnosis
    #DXCHANGE/Diagnosis: 1=Stable:NL to NL, 2=Stable:MCI to MCI,
    #3=Stable:AD to AD, 4=Conv:NL to MCI, 5=Conv:MCI to
    #AD, 6=Conv:NL to AD, 7=Rev:MCI to NL, 8=Rev:AD to
    #MCI, 9=Rev:AD to NL.
    adni_diagnostic_summ = pd.read_csv(filepath+'DXSUM_PDXCONV_ADNIALL.csv')
    adni_diagnostic_summ_future = pd.read_csv(filepath + 'DXSUM_PDXCONV_ADNIALL.csv')
    adni_diagnostic_summ = adni_diagnostic_summ[(adni_diagnostic_summ['Phase'].isin(['ADNI2','ADNIGO']))
                                                &(adni_diagnostic_summ['VISCODE'].isin(['sc','bl','v03']))]
    adni_diagnostic_summ = adni_diagnostic_summ.loc[:,['Phase','RID','DXCHANGE']]
    adni_diagnostic_summ = adni_diagnostic_summ.rename(columns={'Phase':'COLPROT'})

    adni_df = pd.merge(adni_df,adni_diagnostic_summ)
    adni_df.insert(2, 'Diagnosis', adni_df['DXCHANGE'])
    del adni_df['DXCHANGE']
    #print(adni_df.info())
    #selecting diagnosis at year 2
    adni_diagnostic_summ_future = adni_diagnostic_summ_future[(adni_diagnostic_summ_future['Phase'].isin(['ADNI2', 'ADNIGO']))
                                                & (adni_diagnostic_summ_future['VISCODE'].isin(['m18', 'v21']))]
    adni_diagnostic_summ_future = adni_diagnostic_summ_future.loc[:, ['Phase', 'RID', 'DXCHANGE']]
    adni_diagnostic_summ_future = adni_diagnostic_summ_future.rename(columns={'Phase': 'COLPROT',
                                                                              'DXCHANGE':'Future Diagnosis'})
    adni_diagnostic_summ= pd.merge(adni_diagnostic_summ,adni_diagnostic_summ_future)
    adni_diagnostic_summ = adni_diagnostic_summ.rename(columns={'DXCHANGE': 'Baseline Diagnosis',
                                                                'RID':'PAT_ID'})
    diagnosis_dict={1:'Normal_CI',2:'MCI',3:'Dementia',4:'MCI',5:'Dementia',6:'Dementia',7:'Normal_CI',
                    8:'MCI',9:'Normal_CI'}
    adni_diagnostic_summ['Baseline Diagnosis']=adni_diagnostic_summ['Baseline Diagnosis'].apply(lambda x: diagnosis_dict[x])
    adni_diagnostic_summ['Future Diagnosis'] = adni_diagnostic_summ['Future Diagnosis'].apply(lambda x: diagnosis_dict[x])


    adni_moca =pd.read_csv(filepath + 'MOCA.csv')
    adni_moca_future = pd.read_csv(filepath + 'MOCA.csv')
    adni_moca = adni_moca[(adni_moca['Phase'].isin(['ADNI2','ADNIGO'])) &(adni_moca['VISCODE'].isin(['sc','bl','v03']))]
    adni_moca = calulate_moca(adni_moca)
    adni_moca = adni_moca.loc[:,['Phase','RID','MOCA_Calc']]
    adni_moca['MOCA_Calc'] = adni_moca['MOCA_Calc'].apply(lambda x: calulate_diagnosis(x))
    adni_moca= adni_moca.rename(columns={'MOCA_Calc': 'Baseline Diagnosis Moca'})

    adni_moca_future = adni_moca_future[(adni_moca_future['Phase'].isin(['ADNI2', 'ADNIGO']))
                                        & (adni_moca_future['VISCODE'].isin(['m18', 'v21']))]
    adni_moca_future = calulate_moca(adni_moca_future)
    adni_moca_future = adni_moca_future.loc[:, ['Phase', 'RID', 'MOCA_Calc']]
    adni_moca_future['MOCA_Calc'] = adni_moca_future['MOCA_Calc'].apply(lambda x: calulate_diagnosis(x))
    adni_moca_future = adni_moca_future.rename(columns={'MOCA_Calc': 'Future Diagnosis Moca'})
    adni_moca = pd.merge(adni_moca,adni_moca_future)
    adni_moca = adni_moca.rename(columns={'Phase': 'COLPROT','RID':'PAT_ID'})
    adni_diagnostic_summ = pd.merge(adni_diagnostic_summ,adni_moca)
    adni_diagnostic_summ.to_csv('adni_patient_labels.csv')

    #for col in range(len(adni_moca.columns)):
    #    print ("col",adni_moca.columns[col],":",adni_moca.iloc[0,col])

    #adni_moca.to_csv('MOCA_calc.csv')

    return adni_df

def calulate_moca(adni_moca):
    adni_moca['MOCA_Calc'] = adni_moca['TRAILS'] + adni_moca['CUBE'] + adni_moca['CLOCKCON'] + adni_moca['CLOCKNO'] \
                             + adni_moca['CLOCKHAN'] \
                             + adni_moca['LION'] + adni_moca['RHINO'] + adni_moca['CAMEL'] + adni_moca['DIGFOR'] \
                             + adni_moca['DIGBACK'] + adni_moca['REPEAT1'] + adni_moca['REPEAT2'] + adni_moca['ABSTRAN'] \
                             + adni_moca['ABSMEAS'] + adni_moca['MONTH'] + adni_moca['YEAR'] \
                             + adni_moca['PLACE'] + adni_moca['CITY']
    # print ("first calc",adni_moca['MOCA_Calc'])
    adni_moca['MOCA_Calc'] += adni_moca['LETTERS'].apply(lambda x: 1 if x < 2 else 0)
    adni_moca['SERIAL'] = adni_moca['SERIAL1'] + adni_moca['SERIAL2'] + adni_moca['SERIAL3'] + adni_moca['SERIAL4'] + \
                          adni_moca['SERIAL5']
    adni_moca['MOCA_Calc'] += adni_moca['SERIAL'].apply(lambda x: serial_calc(x))
    adni_moca['MOCA_Calc'] += adni_moca['FFLUENCY'].apply(lambda x: 1 if x >= 11 else 0)
    adni_moca['MOCA_Calc'] += adni_moca['DELW1'].apply(lambda x: 1 if x == 1 else 0)
    adni_moca['MOCA_Calc'] += adni_moca['DELW2'].apply(lambda x: 1 if x == 1 else 0)
    adni_moca['MOCA_Calc'] += adni_moca['DELW3'].apply(lambda x: 1 if x == 1 else 0)
    adni_moca['MOCA_Calc'] += adni_moca['DELW4'].apply(lambda x: 1 if x == 1 else 0)
    adni_moca['MOCA_Calc'] += adni_moca['DELW5'].apply(lambda x: 1 if x == 1 else 0)
    adni_moca['day_calc'] = adni_moca['DAY'] + adni_moca['DATE']
    adni_moca['MOCA_Calc'] += adni_moca['day_calc'].apply(lambda x: 1 if x == 2 else 0)

    return adni_moca

def calulate_diagnosis(x):
    if x > 26:
        return 'Normal_CI'
    elif x>=23 and x<=26:
        return 'MCI'
    else:
        return 'Dementia'

def serial_calc(x):
    if x==1:
        return 1
    elif x>=2 and x<=3:
        return 2
    elif x>=4 and x<=5:
        return 3
    else:
        return 0

def select_pd_patients_moca():
    #eddie's patient file
    #ppmi_moca_stats = pd.read_csv(filepath + 'pd_moca_scores.csv')

    #extracting the MOCA scores of patient IDs provided by Abhishek
    pd_biospec_dat= pd.read_csv(filepath+'pd_biospecimen.csv')
    print(len(set(pd_biospec_dat['PATNO_PPMI'])))
    #get_pd_labels_from_moca(get_moca_scores(list(pd_biospec_dat['PATNO_PPMI'])),'pd_biospec_moca.csv')

    #extracting moca scores for which mri data is available from above list
    #3T scans
    pd_mri_3t = pd.read_csv(filepath+'filtered_mri_ppmi_data_3T.csv')
    print(len(set(pd_mri_3t['Subject ID'])))
    get_pd_labels_from_moca(get_moca_scores(list(pd_mri_3t['Subject ID'])),'pd_biospec_mri_moca.csv')

def get_pd_labels_from_moca(ppmi_moca_stats,out_file_name):
    ppmi_moca_stats.insert(1,'Baseline Diagnosis','PD')
    ppmi_moca_stats.insert(3,'Future Diagnosis','PD')
    ppmi_moca_stats.insert(0,'COLPROT','PPMI')
    bs_pd_mci=0
    fu_pd_mci=0
    bs_pdd =0
    fu_pdd =0
    mci_to_pdd =0
    stable_pdd =0
    normal_to_pdd =0
    bs_normal=0
    fu_normal=0
    for index,row in ppmi_moca_stats.iterrows():
        if row['SC_MCATOT'] > 26:
            ppmi_moca_stats.loc[index,'Baseline Diagnosis'] = 'Normal_CI'
            bs_normal+=1
        elif row['SC_MCATOT'] >= 23 and row['SC_MCATOT'] <=26:
            ppmi_moca_stats.loc[index, 'Baseline Diagnosis'] = 'MCI'
            bs_pd_mci +=1
        else:
            ppmi_moca_stats.loc[index, 'Baseline Diagnosis'] = 'Dementia'
            bs_pdd+=1

        if row['V06_MCATOT'] > 26:
            ppmi_moca_stats.loc[index,'Future Diagnosis'] = 'Normal_CI'
            fu_normal+=1
        elif row['V06_MCATOT'] >= 23 and row['V06_MCATOT'] <=26:
            ppmi_moca_stats.loc[index, 'Future Diagnosis'] = 'MCI'
            fu_pd_mci+=1
        else:
            ppmi_moca_stats.loc[index, 'Future Diagnosis'] = 'Dementia'
            fu_pdd+=1
            if ppmi_moca_stats.loc[index, 'Baseline Diagnosis'] == 'MCI':
                mci_to_pdd +=1
            if ppmi_moca_stats.loc[index, 'Baseline Diagnosis'] == 'Dementia':
                stable_pdd +=1
            if ppmi_moca_stats.loc[index, 'Baseline Diagnosis'] == 'Normal_CI':
                normal_to_pdd+=1
    print("baseline mci=",bs_pd_mci,"future mci=",fu_pd_mci,"baseline pdd=",bs_pdd,"future pdd=",
          fu_pdd,"baseline cognitive normal=",bs_normal,"future cognitive normal=",fu_normal)
    print ("mci_to_pdd=",mci_to_pdd,"stable pdd=",stable_pdd,"normal_to_pdd=",normal_to_pdd)
    ppmi_moca_stats.to_csv(out_file_name)

def get_moca_scores(patient_list):
    ppmi_moca_data = pd.read_csv(filepath + 'Montreal_Cognitive_Assessment__MoCA_.csv')
    ppmi_moca_data = ppmi_moca_data[ppmi_moca_data['PATNO'].isin(patient_list)]
    #Screening scores
    ppmi_moca_data_bl = ppmi_moca_data[ppmi_moca_data['EVENT_ID'].isin(['SC'])].loc[:, ['PATNO', 'MCATOT']].rename(
                                                                        columns={'MCATOT': 'SC_MCATOT'})
    #24month visit scores
    ppmi_moca_data_v06 = ppmi_moca_data[ppmi_moca_data['EVENT_ID'].isin(['V06'])].loc[:, ['PATNO', 'MCATOT']].rename(
                                                                        columns={'MCATOT': 'V06_MCATOT'})
    ppmi_moca_final = pd.merge(ppmi_moca_data_bl, ppmi_moca_data_v06, on='PATNO')
    #print(len(set(ppmi_moca_final['PATNO'])))
    #print(list(ppmi_moca_final['PATNO']))
    #ppmi_moca_final.to_csv('filtered_ppmi_moca.csv')
    return ppmi_moca_final


if __name__ == '__main__':
    clean_and_merge()
    select_pd_patients_moca()