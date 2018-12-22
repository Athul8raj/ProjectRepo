import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

X = np.array([[1,2],[2,3],[5,7],[5,8],[8,8],[1,3]])
colors = ['g','r','b','k','c']

class K_means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i] 
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)


            prev_centroid = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
             
            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroid[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized =False
                    
            if optimized:
                break   
            
            
    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    
clf = K_means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='x',s=150,linewidths=5,color='k')
    
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],color=color,s=150,linewidths=5)
 
unknowns = np.array([[4,5],[8,9],[5,4],[6,4]]) 

for unknown in unknowns:
    clf.predict(unknown)
    plt.scatter(unknown[0],unknown[1],marker='o',color=colors[classification],s=150,linewidths=5)

plt.show()

                
            
        