import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing

pd.options.mode.chained_assignment = None

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['name','body','home.dest'],1,inplace=True)
df['cabin'] = pd.Series(np.array(['A','B','C','D']))
df_sex_embarked = pd.get_dummies(df,columns=['embarked','sex','cabin'],drop_first=True)

df = pd.concat([df,df_sex_embarked],axis=1)

df.drop(['sex','embarked','cabin','boat','ticket'],1,inplace=True)
df.fillna(0,inplace=True)


X = np.array(df.drop(['survived'],1).astype(float))
y = np.array(df['survived'])
X = preprocessing.scale(X)

clf = MeanShift()
clf.fit(X) 

labels = clf.labels_
cluster_centers = clf.cluster_centers_
n_clusters = len(np.unique(labels))

original_df['cluster_group'] = np.nan
#print(original_df['survived']

survival_rates = {}

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
for i in range(n_clusters):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived']== 1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)



        
    




