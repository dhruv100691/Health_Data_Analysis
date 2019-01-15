import process_ppmi_image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def merged_data_analysis ():
    merged_df=process_ppmi_image.clean_and_merge()
    #useful for finding out missing values
    #merged_df.describe().to_csv('data_describe.csv')

    #TODO: move this data cleaning to preprocess script. For some reason it is not working there
    del merged_df['5th-Ventricle']  # column has too many missing values
    merged_df.fillna(merged_df.mean(), inplace=True) #replace missing values with mean column value
    #print(merged_df['Right-vessel'].isna().sum())

    normalized_df = min_max_normalize(merged_df)
    #cannot try NMF with zscore
    #normalized_df = z_score_normalize(merged_df)
    #normalized_df.to_csv('normalized_data.csv')

    df_reduced_columns = ['COLPROT','PAT_ID','Diagnosis','PCA_1', 'PCA_2', 'PCA_3','ICA_1', 'ICA_2', 'NMF_2_1', 'NMF_2_2',
                   'NMF_3_1', 'NMF_3_2', 'NMF_3_3']
    df_reduced = pd.DataFrame(index=normalized_df.index, columns=df_reduced_columns)

    #feature reduction algos : PCA/NMF/ICA
    model_PCA = PCA(n_components=3)
    df_reduced[['PCA_1', 'PCA_2', 'PCA_3']] = model_PCA.fit_transform(normalized_df.loc[:,normalized_df.columns[3:]])
    print("PCA explained variance",model_PCA.explained_variance_ratio_)

    model_NMF = NMF(n_components=2, init='nndsvda', max_iter=200)
    model_NMF3 = NMF(n_components=3, init='nndsvda', max_iter=200)
    df_reduced[['NMF_2_1', 'NMF_2_2']] = model_NMF.fit_transform(normalized_df.loc[:,normalized_df.columns[3:]])
    df_reduced[['NMF_3_1', 'NMF_3_2', 'NMF_3_3']] = model_NMF3.fit_transform(normalized_df.loc[:,normalized_df.columns[3:]])

    model_ICA = FastICA(n_components=2)
    df_reduced[['ICA_1', 'ICA_2']] = model_ICA.fit_transform(normalized_df.loc[:,normalized_df.columns[3:]])

    #tsne = TSNE(n_components=3, verbose=1, perplexity=40)
    #df_reduced[['TSNE_1', 'TSNE_2', 'TSNE_3']] = tsne.fit_transform(normalized_df.loc[:,normalized_df.columns[3:]])

    df_reduced['Diagnosis'] = normalized_df['Diagnosis']
    df_reduced['PAT_ID'] = normalized_df['RID'].astype(str).str.replace("sub","").str.replace("/","")
    df_reduced['COLPROT'] = normalized_df['COLPROT']
    df_reduced.to_csv("reduced_feature_images.csv")

    #plotting
    # Diagnosis: 1=Stable:NL to NL, 2=Stable:MCI to MCI,
    # 3=Stable:AD to AD, 4=Conv:NL to MCI, 5=Conv:MCI to
    # AD, 6=Conv:NL to AD, 7=Rev:MCI to NL, 8=Rev:AD to
    # MCI, 9=Rev:AD to NL
    color_dict = {1.0: 'red', 2.0: 'blue', 'PD': 'black', 3.0: 'yellow', 4.0: 'green', 7.0: 'purple'}
    plt.figure(1, figsize=(36, 36))
    plt.subplot(3, 2, 1)

    #TODO: find better way to plot colored points
    for index,row in df_reduced.iterrows():
        plt.scatter(row[['PCA_1']], row[['PCA_2']],c=color_dict[row['Diagnosis']])
    plt.title('Dimension reduction with PCA')

    plt.subplot(3, 2, 2)
    for index,row in df_reduced.iterrows():
        plt.scatter(-row[['NMF_2_1']], row[['NMF_2_2']],c=color_dict[row['Diagnosis']])
    plt.title('Dimension reduction with NMF')

    plt.show()

def min_max_normalize(df):
    result = df.copy()
    for col in df.columns[3:]:
        #print (col,df[col].min(),df[col].max())
        result[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return result

def z_score_normalize(df):
    result=df.copy()
    for col in df.columns[3:]:
        #print (col,df[col].min(),df[col].max())
        result[col] = (df[col] - df[col].mean())/(df[col].std(ddof=0))
    return result


if __name__ == '__main__':
    merged_data_analysis()