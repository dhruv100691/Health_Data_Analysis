"""
    This script assembles all data from different modalities
    into one single dataframe, to be used as training/validation
    /testing set for our classification model.

    Input Files:
        UCSF_ADNI/reduced_feature_images.csv -> feature images in PCA (3D), ICA (2D) and NMF (2D/3D) spaces for PPMI+ADNI patients
        UCSF_ADNI/patients_biospecimen.csv -> genetic(APOE) and biospecimen(CSF alpha-sync, CSF t-tau, CSF p-tau, Abeta 42) of PPMI patients
        UCSF_ADNI/adni_biomarker_features.csv -> genetic(APOE) and biospecimen(CSF t-tau, CSF p-tau, Abeta 42) of ADNI patients
        UCSF_ADNI/pd_biospec_mri_moca.csv -> baseline diagnosis and future diagnosis (MOCA scores) of PPMI patients
        src/ad_pd_analysis.py -> dictionary for labelling of ADNI patients in reduced_feature_images.csv

    Output Files:
        UCSF_ADNI/assembled_data.csv -> to be used for training/validation/testing of classification model

    The script has been made verbose for easy usage and
    each column name in assembled_data.csv is also verbose for
    easy interpretation.
"""

import pandas as pd

# Create a new dataframe
# We choose age, gender, 1 genetic feature, 3 biospecimen and 3-D image features for prediction
# COLPROT : ADNI/PPMI
# PATID : patient id in its COLPROT
# DIAGNOSIS : actual diagnosis
# AGE : in years
# GENDER : male/female
# APOE : genetic
# T-Tau : biospecimen
# P-Tau : biospecimen
# ABeta-42 : biospecimen
# IMAGE_1 : reduced feature from freesurfer-preprocessed structural MRI images
# IMAGE_2 : reduced feature from freesurfer-preprocessed structural MRI images
# IMAGE_3 : reduced feature from freesurfer-preprocessed structural MRI images
# BASELINE_DiagnosisL : PD_NC/PD_MCI/PDD at baseline (note we assume Screening ~ baseline)
# FUTURE_DiagnosisL : PD_NC/PD_MCI/PDD after 2 years
# OUTPUT_LABEL : NC/sMCI/pMCI -> depicts progression of dementia (assigned by comparing BASELINE_LABEL_INTERPRETED and FUTURE_LABEL_INTERPRETED)
def merge_ad_pd_patients():
    image_data_frame = pd.read_csv("../../UCSF_ADNI/reduced_feature_images.csv")
    ppmi_biospecimen_data_frame = pd.read_csv("../../UCSF_ADNI/patients_biospecimen.csv")
    adni_biospecimen_data_frame = pd.read_csv("../../UCSF_ADNI/adni_biomarker_features.csv")
    ppmi_label_data_frame = pd.read_csv("../../UCSF_ADNI/pd_biospec_mri_moca.csv")

    ppmi_label_data_frame = ppmi_label_data_frame.rename(columns={'PATNO':'PAT_ID'})
    #TODO:why this merge doesnt work without providing keys??
    #merging image dataframe and PD label data frame
    ppmi_feature_df = pd.merge(ppmi_label_data_frame,image_data_frame,on=['PAT_ID','COLPROT'])
    ppmi_feature_df= ppmi_feature_df.drop(columns=['Unnamed: 0_x','Unnamed: 0_y','Diagnosis',
                                                  'ICA_1','ICA_2','NMF_2_1','NMF_2_2','NMF_3_1','NMF_3_2','NMF_3_3',
                                                  'V06_MCATOT'])

    ppmi_biospecimen_data_frame.insert(0,'COLPROT','PPMI')
    #merge biospecimen dataframe
    ppmi_biospecimen_data_frame=ppmi_biospecimen_data_frame.rename(columns={'PATNO_PPMI':'PAT_ID'})
    ppmi_feature_df=pd.merge(ppmi_feature_df,ppmi_biospecimen_data_frame)
    ppmi_feature_df.insert(0, 'Baseline Diagnosis Moca', ppmi_feature_df['SC_MCATOT'])
    ppmi_feature_df.insert(0, 'Future Diagnosis Moca', ppmi_feature_df['Future Diagnosis'])
    ppmi_feature_df=ppmi_feature_df.drop(columns=['DIAGNOSIS','CSF Alpha-synuclein','SC_MCATOT'])
    #merge adni image and biospecimen dataframe
    adni_feature_df=image_data_frame[image_data_frame['COLPROT'].isin(['ADNIGO',"ADNI2"])]
    adni_biospecimen_data_frame=adni_biospecimen_data_frame.rename(columns={'RID':'PAT_ID'})
    #print(adni_biospecimen_data_frame[adni_biospecimen_data_frame.duplicated(keep=False)])
    adni_feature_df=pd.merge(adni_feature_df,adni_biospecimen_data_frame)
    adni_feature_df=adni_feature_df.drop(columns=['Unnamed: 0','Diagnosis',
                                                  'ICA_1','ICA_2','NMF_2_1','NMF_2_2','NMF_3_1','NMF_3_2','NMF_3_3'])
    adni_feature_df=adni_feature_df.rename(columns={'PTGENDER':'GENDER','APOE':'ApoE Genotype',
                                                    'TAU':'Total tau','PTAU':'p-Tau181P','ABETA':'Abeta 42'})
    #merge adni labels
    adni_feature_df=pd.merge(pd.read_csv("../../UCSF_ADNI/adni_patient_labels.csv"),adni_feature_df)
    adni_feature_df=adni_feature_df.drop(columns=['Unnamed: 0'])
    #concatenate pd and ad features
    pd.concat([ppmi_feature_df,adni_feature_df],ignore_index=True).to_csv('ad_pd_features.csv')

    #adni_feature_df.to_csv('adni_feature.csv')
    #ppmi_feature_df.to_csv('ppmi_final_features.csv')

if __name__ == '__main__':
    merge_ad_pd_patients()


