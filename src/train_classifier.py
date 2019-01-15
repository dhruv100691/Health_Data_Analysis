### Author: Edward Huang

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.utils import resample
from sklearn.decomposition import PCA
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import tree
import graphviz
import shap

### Splits dataset into folds, trains a classifier.
NUM_FOLDS = 5
# Change this dictionary to map different progressions to different class labels.
CLASS_LABELS = {'Normal_CI-Normal_CI':2, 'MCI-MCI':1, 'Dementia-Dementia':0,
    'Normal_CI-MCI':1, 'MCI-Dementia':0, 'Normal_CI-Dementia':0, 'Dementia-Normal_CI':3,
    'Dementia-MCI':3, 'MCI-Normal_CI':3}
CLASS_LABELS_MOCA = {'Normal_CI-Normal_CI':2, 'MCI-MCI':1, 'Dementia-Dementia':0,
    'Normal_CI-MCI':1, 'MCI-Dementia':0, 'Normal_CI-Dementia':0, 'Dementia-Normal_CI':2,
    'Dementia-MCI':1, 'MCI-Normal_CI':2}
def read_feature_matrix():
    '''
    Preprocessing the feature matrix. Generates class labels, and removes
    strings from float fields.
    '''
    fname = '../../UCSF_ADNI/ad_pd_features.csv'
    df = pd.read_csv(fname)
    # First column is not useful.
    df.drop(columns='Unnamed: 0', inplace=True)
    # Remove any less than or greater than signs from the numerical fields.
    for numerical_col in ['Total tau', 'p-Tau181P', 'Abeta 42']:
        df[numerical_col] = df[numerical_col].str.replace('>', '')
        df[numerical_col] = df[numerical_col].str.replace('<', '')
        df[numerical_col] = df[numerical_col].astype(float)
    #copying baseline diagnosis, to be considered as a feature also
    #df.insert(7, 'Baseline_diag', df['Baseline Diagnosis'])
    # Factorize gender and the ApoE genotype, baseline diagnosis .
    cat_col = ['GENDER', 'ApoE Genotype']
    df[cat_col] = df[cat_col].apply(lambda x: pd.factorize(x)[0])

    # Combine the baseline and future diagnoses to get a class.
    df['label'] = df[['Baseline Diagnosis', 'Future Diagnosis']].apply(lambda x: '-'.join(x),
        axis=1)
    # Re-map the class label to an integer.
    df = df.replace({'label': CLASS_LABELS})
    cols = df.columns.tolist()
    cols.insert(-2, cols.pop(cols.index('ApoE Genotype')))
    cols.insert(-3, cols.pop(cols.index('Abeta 42')))
    cols.insert(0, cols.pop(cols.index('PAT_ID')))
    #cols.insert(-1, cols.pop(cols.index('Abeta 42')))
    #cols.insert(-4, cols.pop(cols.index('ApoE Genotype')))
    #cols.insert(-5, cols.pop(cols.index('GENDER')))
    #cols.insert(-6, cols.pop(cols.index('Baseline_diag')))
    #cols.insert(0, cols.pop(cols.index('PAT_ID')))
    df = df.reindex(columns=cols)
    # Remove duplicates.
    df.drop_duplicates(subset='PAT_ID', inplace=True)
    # Drop patients with nan values.
    df = df.dropna(axis=0, how='any')
    #not including patients that get cured/become better
    df=df[df['label'].isin([0,1,2])]
    #df['label'] = df[['Baseline Diagnosis', 'Future Diagnosis Moca']].apply(lambda x: '-'.join(x),
    #                                                                   axis=1)
    #df = df.replace({'label': CLASS_LABELS_MOCA})
    df.to_csv('read_features.csv')
    return df

def main():
    df = read_feature_matrix()

    lb=LabelBinarizer()
    lb.fit(df['label'])
    #minority class of demntia patients
    df_minority = df[df.label ==0]
    #majority class of MCI/Normal CI patients
    df_majority = df[df.label != 0]

    #df.to_csv('read_features.csv')
    #oversampling demntia patients to 120
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=100,  # to match majority class
                                     random_state=123)  # reproducible results
    #downsampling MCI/CI to a combined count of 250
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=250,  # to match minority class
                                       random_state=123)  # reproducible results

    # Combine original majority class with upsampled minority class only for now
    #df = pd.concat([df_majority, df_minority_upsampled])
    #print("sizes", df.groupby(['label']).size())

    # Split the dataframe into PPMI and ADNI patients.
    adni_df = df.loc[df['COLPROT'] == 'ADNI2']
    ppmi_df = df.loc[df['COLPROT'] == 'PPMI']

    #get_group_statistics(adni_df,'ADNI')
    #get_group_statistics(ppmi_df, 'PPMI')
    get_group_statistics(df,'Total')
    # Split each dataframe into folds.
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True,random_state=42)

    # Get the folds for each dataframe.
    adni_kf = kf.split(adni_df)
    ppmi_kf = kf.split(ppmi_df)

    acc_metric_lst = []
    f1_macro_metric_list = []
    f1_micro_metric_list = []
    precision_metric_list = []
    recall_metric_list = []
    auc_metric_list =[]
    fpr_metric_list =[]
    tpr_metric_list = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    feature_imp_list = []

    for i in range(NUM_FOLDS):
        # Get the train-test split for ADNI.
        adni_idx = next(adni_kf)
        train_adni_df = adni_df.iloc[adni_idx[0]]
        test_adni_df = adni_df.iloc[adni_idx[1]]

        # Get the train-test split for PPMI.
        ppmi_idx = next(ppmi_kf)
        train_ppmi_df = ppmi_df.iloc[ppmi_idx[0]]
        test_ppmi_df =  ppmi_df.iloc[ppmi_idx[1]]

        # Concatenate back the PPMI and ADNI dataframes.
        train_df = pd.concat([train_adni_df, train_ppmi_df])
        test_df = pd.concat([test_adni_df, test_ppmi_df])

        # Get the feature matrix and the class labels.
        features = ['GENDER', 'PCA_1', 'PCA_2', 'PCA_3', 'Total tau', 'Abeta 42',
                    'ApoE Genotype', 'p-Tau181P']
        #print("features",features)
        # Training set
        X_train = train_df[features]
        y_train = train_df['label']
        # Test set
        X_test = test_df[features]
        y_test = test_df['label']

        # TODO: Change classifier.
        clf = RandomForestClassifier(n_jobs=-1,n_estimators=500,max_features=0.6)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #tree_graph = tree.export_graphviz(clf, out_file=None, feature_names=features)
        #graphviz.Source(tree_graph)
        #clf_auc = RandomForestClassifier(n_jobs=-1, random_state=100)
        #clf_auc.fit(X_train, lb.transform(y_train))
        y_pred_prob = clf.predict_proba(X_test)

        #permutation importance: another metric to compute feature importances(based on accuracy score?)
        #perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
        #print(perm.feature_importances_)


        print ("feature importances",clf.feature_importances_)
        feature_dict=dict()
        for index,imp in zip(features,clf.feature_importances_):
            feature_dict[index] = imp
        feature_imp_list += [feature_dict]
        acc_metric_lst += [accuracy_score(y_test, y_pred)]
        #macro gives equal importance to all labels, takes average of all classes
        f1_macro_metric_list +=[f1_score(y_test,y_pred,average='macro')]
        precision_metric_list += [precision_score(y_test, y_pred, average='macro')]
        recall_metric_list+=[recall_score(y_test,y_pred,average='macro')]
        #micro scores will be higher since we are predicting correctly for majority class Normal CI??
        f1_micro_metric_list += [f1_score(y_test, y_pred, average='micro')]
        #print (confusion_matrix(y_test,y_pred))

        for i in range(len(lb.classes_)):
            #calculating false postive/true postive rates and corresponding AUC scores for each class and per fold
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i],pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr_metric_list+=[fpr]
        tpr_metric_list+=[tpr]
        auc_metric_list+=[roc_auc]

    row_to_show =5
    data_for_prediction = X_test.iloc[row_to_show]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(clf)

    # Calculate Shap values
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

    print ("Accuracy :",np.mean(acc_metric_lst))
    print ("Macro F1 :",np.mean(f1_macro_metric_list))
    print ("Precison :",np.mean(precision_metric_list))
    print ("Recall :",np.mean(recall_metric_list))
    print ("Micro F1 :", np.mean(f1_micro_metric_list))
    for i in range(len(lb.classes_)):
        print ("AUC for class ", i , " : ",np.mean([fold[i] for fold in auc_metric_list]))
    mean_feature_dict=dict()
    for key,val in feature_imp_list[0].items():
        mean_feature_dict[key] = np.mean([fold[key] for fold in feature_imp_list])
        #print ("mean importance of ",key," : ",mean_feature_dict[key])
    plt.bar(mean_feature_dict.keys(),mean_feature_dict.values())
    #plot_auc_curves(fpr_metric_list,tpr_metric_list,len(lb.classes_))
    #visualize_data(df)

def get_group_statistics(df,name):
    print("============ ",name," Statistics ===========")
    print ("Total patients :",len(df))
    base_group=df.groupby('Baseline Diagnosis')
    #useful for histo plots
    #print(base_group.hist())
    print(base_group.size())
    future_group= df.groupby('Future Diagnosis')
    print(future_group.size())
    trans_group=df.groupby(['Baseline Diagnosis','Future Diagnosis'])
    print(trans_group.size())
    #trans_group.hist()

def plot_auc_curves(fpr_metric_list,tpr_metric_list,n_class):
    plt.figure()
    lw = 2
    for i in range(n_class):
        tprs=[]
        mean_fpr = np.linspace(0,1,100)
        for fold_fpr,fold_tpr in zip(fpr_metric_list,tpr_metric_list):
            tprs.append(interp(mean_fpr, fold_fpr[i], fold_tpr[i]))
        mean_tpr =np.mean(tprs,axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr,mean_tpr ,lw=lw,
             label='ROC curve class %d (area = %0.2f)' % (i,auc(mean_fpr,mean_tpr)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating Curve (AP -> AP)')
    plt.legend(loc="lower right")
    plt.show()

def visualize_data(normalized_df):
    df_reduced_columns = ['COLPROT', 'PAT_ID', 'Diagnosis', 'PCA_1', 'PCA_2', 'PCA_3', 'ICA_1', 'ICA_2', 'NMF_2_1',
                          'NMF_2_2',
                          'NMF_3_1', 'NMF_3_2', 'NMF_3_3']
    df_reduced = pd.DataFrame(index=normalized_df.index, columns=df_reduced_columns)

    features = ['GENDER', 'PCA_1', 'PCA_2', 'PCA_3', 'Total tau', 'Abeta 42',
                'ApoE Genotype', 'p-Tau181P']
    # feature reduction algos : PCA/NMF/ICA
    model_PCA = PCA(n_components=3)
    df_reduced[['PCA_1', 'PCA_2', 'PCA_3']] = model_PCA.fit_transform(normalized_df.loc[:, features])
    # print(model_PCA.explained_variance_ratio_)

    normalized_df['label'] = normalized_df['label'].astype('str')
    df_reduced['label_colour'] = normalized_df[['COLPROT', 'label']].apply(lambda x: '-'.join(x),axis=1)
    normalized_df['label'] = normalized_df['label'].astype('int64')
    color_dict = {'PPMI-0': 'red', 'PPMI-1': 'blue', 'PPMI-2': 'black', 'ADNI2-0': 'yellow', 'ADNI2-1': 'green', 'ADNI2-2':'orange'}
    plt.figure(1, figsize=(60, 40))
    #plt.gcf().set_size_inches(50, 40)
    #plt.subplot(3, 2, 1)

    # TODO: find better way to plot colored points
    for index, row in df_reduced.iterrows():
        plt.scatter(row[['PCA_2']], row[['PCA_3']], c=color_dict[row['label_colour']])
    plt.title('Dimension reduction with PCA')

    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
    p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='orange')
    p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
    plt.legend((p1, p2, p3, p4, p5, p6), ('PPMI Dementia', 'PPMI MCI', 'ADNI MCI', 'ADNI Dementia', 'ADNI Control', 'PPMI Control'), loc='best');
    plt.show()
if __name__ == '__main__':
    main()
