import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing


df_train = pd.read_csv('train.csv',na_filter=False,dtype=np.int64)
#print(df_train.info())
X = np.array(df_train.drop(['label'],1))
y = np.array(df_train['label'])
X = preprocessing.scale(X) 
'''SVM Kernel'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
clf = svm.SVC(kernel='rbf',max_iter=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
correct = np.sum(y_pred == y_test)
accuracy = correct/len(y_pred)
print("%d out of %d predictions correct" % (correct, len(y_pred)))
print("Accuracy: ",accuracy)

'''create plot with seaborn/matplotlib'''








































'''Neural Network'''
#hidd_lyr1= 500
#hidd_lyr2= 500
#hidd_lyr3= 500
#
#n_classes = 10
#
#def neural_network(data):
#    hiddern_layer1 = {'weights':tf}
    
