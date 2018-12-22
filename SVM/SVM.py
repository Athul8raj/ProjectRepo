import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


df = pd.read_excel('breast-cancer-wisconsin.data.xlsx',delimiter=',',header=None)

df = df[0].str.split(',',expand=True)
breast_cancer_df = pd.DataFrame()
breast_cancer_df['id'] = df[0]
breast_cancer_df['clump_thickness'] = df[1]
breast_cancer_df['unif_cell_size'] = df[2]
breast_cancer_df['cell_shape'] = df[3]
breast_cancer_df['marg_adh'] = df[4]
breast_cancer_df['single_epi_size'] = df[5]
breast_cancer_df['bare_nuclei'] = df[6]
breast_cancer_df['chromatin'] = df[7]
breast_cancer_df['norm_nuclei'] = df[8]
breast_cancer_df['mitosis'] = df[9]
breast_cancer_df['Class'] = df[10]

#print(breast_cancer_df.isin(['?']).any())
breast_cancer_df.replace('?',-99999,inplace=True)
breast_cancer_df.drop(['id'],1,inplace=True)

X = np.array(breast_cancer_df.drop(['Class'],1))
y = np.array(breast_cancer_df['Class'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

clf = svm.SVC()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)