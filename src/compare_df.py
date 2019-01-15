import pandas as pd

df = pd.read_csv('read_features.csv')
df_1 = pd.read_csv('read_features_1.csv')
df.drop(columns=['Unnamed: 0','Baseline Diagnosis Moca','Future Diagnosis Moca'], inplace=True)
df_1.drop(columns=['Unnamed: 0'], inplace=True)
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('COLPROT')))
df = df.reindex(columns=cols)
#df.to_csv('read_features_modified.csv')
count=0
for index,row in df.iterrows():
    #print (row)
    #print((df_1[df_1.PAT_ID == row['PAT_ID']] == row).all(axis=1))
    if (df.iloc[index,1] != df_1.iloc[index,1]):
        print("index ",index)
    count+=1
print("count is",count)