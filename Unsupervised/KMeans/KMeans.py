import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import style

style.use('ggplot')

X = np.array([[1,2],[2,3],[5,7],[5,8],[8,8],[1,3]])

#plt.scatter(X[:,0],X[:,1],s=150)
#plt.show

clf = KMeans(n_clusters=2)

clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

print(labels)
colors = ['g','r','b','k','c']

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
    
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=100,linewidths=5)    
plt.show()