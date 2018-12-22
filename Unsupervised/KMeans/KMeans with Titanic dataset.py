import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')

df.drop(['name','body','home.dest'],1,inplace=True)
df['cabin'] = pd.Series(np.array(['A','B','C','D']))
df_sex_embarked = pd.get_dummies(df,columns=['embarked','sex','cabin'],drop_first=True)

df = pd.concat([df,df_sex_embarked],axis=1)

df.drop(['sex','embarked','cabin','boat','ticket'],1,inplace=True)
df.fillna(0,inplace=True)


X = np.array(df.drop(['survived'],1).astype(float))
y = np.array(df['survived'])
X = preprocessing.scale(X)

clf = KMeans(n_clusters=2)
clf.fit(X) 

correct = 0

for i in range(len(X)):
    prediction = np.array(X[i].astype(float))
    prediction = prediction.reshape(-1,len(prediction))
    predict_me = clf.predict(prediction)
    if predict_me[0] == y[i].any():
        correct +=1
        
print(correct/len(X))
        
    




