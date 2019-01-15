""" 
    This script assembles all data from different modalities
    into one single dataframe, to be used as training/validation
    /testing set for our classification model.

    Input Files:
        UCSF_ADNI/reduced_feature_images.csv -> feature images in PCA (3D), ICA (2D) and NMF (2D/3D) spaces for PPMI+ADNI patients
        UCSF_ADNI/patients_biospecimen.csv -> genetic(APOE) and biospecimen(CSF alpha-sync, CSF t-tau, CSF p-tau, Abeta 42) of PPMI patients
        UCSF_ADNI/adni_biomarker_features.csv -> genetic(APOE) and biospecimen(CSF t-tau, CSF p-tau, Abeta 42) of ADNI patients
        UCSF_ADNI/pd_biospec_moca.csv -> baseline diagnosis and future diagnosis (MOCA scores) of PPMI patients
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
# BASELINE_LABEL_ORIGINAL : PD_NC/PD_MCI/PDD at baseline (note we assume Screening ~ baseline)
# BASELINE_LABEL_INTERPRETED : NC/MCI/DEMENTIA at baseline (note we assume Screening ~ baseline)
# FUTURE_LABEL_ORIGINAL : PD_NC/PD_MCI/PDD after 2 years
# FUTURE_LABEL_INTERPRETED : NC/MCI/DEMENTIA after 2 years (should be prediction label)
# OUTPUT_LABEL : NC/sMCI/pMCI -> depicts progression of dementia (assigned by comparing BASELINE_LABEL_INTERPRETED and FUTURE_LABEL_INTERPRETED)
relevant_columns = ["COLPROT", "PATID", "DIAGNOSIS", "AGE", "GENDER", "APOE", "T-tau", "P-Tau", "ABeta-42", "IMAGE_1", "IMAGE_2", "IMAGE_3", "BASELINE_LABEL_ORIGINAL", "BASELINE_LABEL_INTERPRETED", "FUTURE_LABEL_ORIGINAL", "FUTURE_LABEL_INTERPRETED", "OUTPUT_LABEL"]
new_data_frame = pd.DataFrame(index=list(range(0,900)),columns=relevant_columns)

image_data_frame = pd.read_csv("../../UCSF_ADNI/reduced_feature_images.csv")
ppmi_biospecimen_data_frame = pd.read_csv("../../UCSF_ADNI/patients_biospecimen.csv")
adni_biospecimen_data_frame = pd.read_csv("../../UCSF_ADNI/adni_biomarker_features.csv")
ppmi_label_data_frame = pd.read_csv("../../UCSF_ADNI/pd_biospec_moca.csv")

# We assume that reduced_feature_images.csv has the smallest subset of patients (ADNI+PPMI)
# on which we have to work on
count = 0
for index, row in image_data_frame.iterrows():
    # PPMI patient
    if(row["COLPROT"] == "PPMI"):
        # COLPROT
        new_data_frame.loc[count, "COLPROT"] = row["COLPROT"]    
        # PATID
        new_data_frame.loc[count, "PATID"] = row["PAT_ID"][3:7]
        # DIAGNOSIS
        new_data_frame.loc[count, "DIAGNOSIS"] = row["Diagnosis"]
        
        # AGE
        # TODO : access clinical data to get age
        # may be important for prediction
        
        # GENDER
        # TODO : access clinical data to get gender
        # may be important for prediction
        
        # APOE, T-tau, P-tau, Abeta-42
        # go to ppmi_biospecimen
        x = int(row["PAT_ID"][3:7]) 
        temp = ppmi_biospecimen_data_frame.loc[ppmi_biospecimen_data_frame["PATNO_PPMI"] == x, ["ApoE Genotype", "Total tau", "p-Tau181P", "Abeta 42"]]
        if(temp.empty is False):
            temp = temp.iloc[0]
            new_data_frame.loc[count, "APOE"] = temp["ApoE Genotype"]
            new_data_frame.loc[count, "T-tau"] = temp["Total tau"]
            new_data_frame.loc[count, "P-Tau"] = temp["p-Tau181P"]
            new_data_frame.loc[count, "ABeta-42"] = temp["Abeta 42"]
        
        # IMAGE features
        # Considering PCA for now
        new_data_frame.loc[count, "IMAGE_1"] = row["PCA_1"]
        new_data_frame.loc[count, "IMAGE_2"] = row["PCA_2"]
        new_data_frame.loc[count, "IMAGE_3"] = row["PCA_3"]
        
        # BASELINE, FUTURE
        # go to ppmi_label_data_frame
        temp = ppmi_label_data_frame.loc[ppmi_label_data_frame["PATNO"] == x, ["Baseline Diagnosis", "Future Diagnosis"]]
        if(temp.empty is False):
            temp = temp.iloc[0]
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = temp["Baseline Diagnosis"]
            if(temp["Baseline Diagnosis"] == "PD_MCI"):
                new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "MCI"
            elif(temp["Baseline Diagnosis"] == "Normal_CI"):
                new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "NC"
            elif(temp["Baseline Diagnosis"] == "PDD"):
                new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "DEMENTIA"
            else:
                print("Undefined class label!")

            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = temp["Future Diagnosis"]
            if(temp["Future Diagnosis"] == "PD_MCI"):
                new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "MCI"
            elif(temp["Future Diagnosis"] == "Normal_CI"):
                new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "NC"
            elif(temp["Future Diagnosis"] == "PDD"):
                new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "DEMENTIA"
            else:
                print("Undefined class label!")
        
        # OUTPUT LABEL
        # TODO: determine mapping
        # NC - NC => NC
        # NC - MCI => pMCI
        # NC - DEMENTIA => pMCI
        # MCI - NC => ?
        # MCI - MCI => sMCI
        # MCI - DEMENTIA => pMCI
        # DEMENTIA - NC => ?
        # DEMENTIA - MCI => ?
        # DEMENTIA - DEMENTIA => sMCI ? 
    else:
        # COLPROT
        new_data_frame.loc[count, "COLPROT"] = row["COLPROT"]    
        # PATID
        new_data_frame.loc[count, "PATID"] = row["PAT_ID"]
        # DIAGNOSIS
        new_data_frame.loc[count, "DIAGNOSIS"] = row["Diagnosis"]
        
        # AGE
        # TODO : access clinical data to get age
        # may be important for prediction
        # GENDER
        # TODO : access clinical data to get gender
        # may be important for prediction
        
        # APOE, T-tau, P-tau, Abeta-42
        # go to adni_biospecimen
        x_1 = int(row["PAT_ID"]) 
        temp_1 = adni_biospecimen_data_frame.loc[adni_biospecimen_data_frame["RID"] == x_1, ["APOE", "TAU", "PTAU", "ABETA"]]
        if(temp_1.empty is False):
            temp_1 = temp_1.iloc[0]
            new_data_frame.loc[count, "APOE"] = temp_1["APOE"]
            new_data_frame.loc[count, "T-tau"] = temp_1["TAU"]
            new_data_frame.loc[count, "P-Tau"] = temp_1["PTAU"]
            new_data_frame.loc[count, "ABeta-42"] = temp_1["ABETA"]

        # IMAGE features
        # Considering PCA for now
        new_data_frame.loc[count, "IMAGE_1"] = row["PCA_1"]
        new_data_frame.loc[count, "IMAGE_2"] = row["PCA_2"]
        new_data_frame.loc[count, "IMAGE_3"] = row["PCA_3"]

        # BASELINE, FUTURE
        # Diagnosis: 1=Stable:NL to NL, 2=Stable:MCI to MCI,
        # 3=Stable:AD to AD, 4=Conv:NL to MCI, 5=Conv:MCI to
        # AD, 6=Conv:NL to AD, 7=Rev:MCI to NL, 8=Rev:AD to
        # MCI, 9=Rev:AD to NL
        # Reference : src/ad_pd_analysis.py
        # TODO : confirm that this labelling is baseline->future (2 years)
        diagnosis = int(float(row["Diagnosis"]))
        if(diagnosis == 1):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "NC"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "NC"
        elif(diagnosis == 2):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "MCI"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "MCI"
        elif(diagnosis == 3):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "DEMENTIA" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "DEMENTIA"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "DEMENTIA" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "DEMENTIA"
        elif(diagnosis == 4):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "NC"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "MCI"
        elif(diagnosis == 5):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "MCI"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "AD" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "DEMENTIA"
        elif(diagnosis == 6):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "NC"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "AD" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "DEMENTIA"
        elif(diagnosis == 7):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "MCI"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "NC"
        elif(diagnosis == 8):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "AD" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "DEMENTIA"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "MCI" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "MCI"
        elif(diagnosis == 9):
            new_data_frame.loc[count, "BASELINE_LABEL_ORIGINAL"] = "AD" 
            new_data_frame.loc[count, "BASELINE_LABEL_INTERPRETED"] = "DEMENTIA"
            new_data_frame.loc[count, "FUTURE_LABEL_ORIGINAL"] = "NL" 
            new_data_frame.loc[count, "FUTURE_LABEL_INTERPRETED"] = "NC"
        else:
            print("Undefined class label!")

        # OUTPUT LABEL
        # TODO: determine mapping

    count += 1

# Write data to csv
#new_data_frame.to_csv("../UCSF_ADNI/assembled_data.csv", index=False)
new_data_frame.to_csv("../../UCSF_ADNI/assembled_data.csv", index=True)


