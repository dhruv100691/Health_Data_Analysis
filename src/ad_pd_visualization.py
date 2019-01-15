import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

df = pd.read_csv('read_features.csv')
columns=['PCA_1', 'PCA_2', 'PCA_3', 'Total tau', 'Abeta 42','p-Tau181P']

#print(sns.axes_style())
#sns.set_style({"legend.frameon": True,"legend.numpoints":1})
plt.figure(1)
for i,col in enumerate(columns):
    plt.subplot(4, 2, i+1)
    sns.boxplot(x="Future Diagnosis",y=col,data=df,hue='COLPROT',width=0.5)
    #sns.barplot(x="Future Diagnosis",y=col,data=df,hue='COLPROT',width=0.5)
    plt.legend(loc='lower left', bbox_to_anchor=(0.73, 2.05,0.05,0.05),
          ncol=3, fancybox=False, shadow=False)

 #g = sns.FacetGrid(df, row='COLPROT', col='Baseline Diagnosis')
#g= g.map(plt.hist, "PCA_1")
#sns.plt.show()

df_reduced_columns = ['PCA_1', 'PCA_2', 'PCA_3', 'TSNE_1', 'TSNE_2','TSNE_3', 'NMF_2_1','NMF_2_2',
                      'NMF_3_1', 'NMF_3_2', 'NMF_3_3','patient_label']
df_reduced = pd.DataFrame(index=df.index, columns=df_reduced_columns)

#tsne = TSNE(n_components=3, verbose=1, perplexity=40)
#df_reduced[['TSNE_1', 'TSNE_2', 'TSNE_3']] = tsne.fit_transform(df.loc[:,columns])

# feature reduction algos : PCA/NMF/ICA
df_reduced['patient_label'] = df[['COLPROT', 'Future Diagnosis']].apply(lambda x: '-'.join(x),axis=1)
model_PCA = PCA(n_components=3)
df_reduced[['PCA_1', 'PCA_2', 'PCA_3']] = model_PCA.fit_transform(df.loc[:,columns])
kmeans = KMeans(n_clusters=2, random_state=0).fit(df_reduced[['PCA_1', 'PCA_2', 'PCA_3']])
#kmeans = KMeans(n_clusters=2, random_state=0).fit(df_reduced[['TSNE_1', 'TSNE_2', 'TSNE_3']])

print ("explained variance",model_PCA.explained_variance_ratio_)
#print ("KL divergence",TSNE.kl_divergence_)
df['clustered_labels'] = kmeans.labels_
columns_new=['PCA_1', 'PCA_2', 'PCA_3', 'Total tau', 'Abeta 42','p-Tau181P','COLPROT','clustered_labels']
clustered_group = df[columns_new].groupby(['COLPROT','clustered_labels'])
#print(clustered_group.hist())
print (clustered_group.size())
df['clustered_labels'] = df['clustered_labels'].astype(str)
df['final_clustered_label'] = df[['COLPROT', 'clustered_labels']].apply(lambda x: '-'.join(x),
        axis=1)
df.drop(columns='clustered_labels', inplace=True)
print (df.columns)
columns_new=['PCA_1', 'PCA_2', 'PCA_3', 'Total tau', 'Abeta 42','p-Tau181P']
df=df[df['final_clustered_label'].isin(['PPMI-0','ADNI2-0'])]
fig = plt.figure(2)
for i,col in enumerate(columns_new):
    plt.subplot(4, 2, i+1)
    sns.boxplot(x="Future Diagnosis",y=col,data=df,hue='final_clustered_label',width=0.5)
    plt.legend(loc='lower left', bbox_to_anchor=(0.73, 2.05,0.05,0.05),
          ncol=3, fancybox=False, shadow=False)


centers = kmeans.cluster_centers_
print (centers)
fig = plt.figure(3)
ax = Axes3D(fig)
color_dict = {'PPMI-Dementia': 'red', 'PPMI-MCI': 'blue', 'PPMI-Normal_CI': 'black', 'ADNI2-Dementia': 'yellow',
              'ADNI2-MCI': 'green', 'ADNI2-Normal_CI':'orange'}
for index, row in df_reduced.iterrows():
    ax.scatter(row['PCA_1'],row['PCA_2'],row['PCA_3'],color=color_dict[row['patient_label']])
    #if row['patient_label'] != 'PPMI-Normal_CI' and row['patient_label'] != 'ADNI2-Normal_CI':
    #    ax.scatter(row['TSNE_1'], row['TSNE_2'], row['TSNE_3'], color=color_dict[row['patient_label']])

ax.scatter(centers[:,0],centers[:,1],centers[:,2],c='orange', s=200, alpha=0.5)
plt.title('Patient Feature Space')
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='orange')
p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
plt.legend((p1, p2, p3, p4, p5, p6), ('PPMI Dementia', 'PPMI MCI', 'ADNI MCI', 'ADNI Dementia', 'ADNI Control', 'PPMI Control'), loc='best')
plt.show()

df.to_csv('new_analyzed.csv')
'''
df['Disease Progression (Baseline -> 24 Months)'] = df[['Baseline Diagnosis','Future Diagnosis']].apply(lambda x: '-'.join(x),axis=1)
plt.figure(3)
sns.countplot(x="Disease Progression (Baseline -> 24 Months)",data=df,hue='COLPROT')
plt.legend(loc='best')

df['patient_label_future'] = df[['COLPROT', 'Future Diagnosis']].apply(lambda x: '-'.join(x),axis=1)
fig = plt.figure(4)
ax = Axes3D(fig)
for index, row in df.iterrows():
    ax.scatter(row['PCA_1'], row['PCA_2'], row['PCA_3'], color=color_dict[row['patient_label_future']])
plt.title('Image Features')
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='orange')
p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
#plt.legend((p1, p2, p3, p4, p5, p6), ('Normal_CI-Dementia', 'Normal_CI-MCI', 'MCI-MCI', 'Dementia-Dementia', 'Normal_CI-Normal_CI', 'MCI-Dementia'), loc='best')
#plt.show()
plt.legend((p1, p2, p3, p4, p5, p6), ('PPMI Dementia', 'PPMI MCI', 'ADNI MCI', 'ADNI Dementia', 'ADNI Control', 'PPMI Control'), loc='best')
plt.show()
'''