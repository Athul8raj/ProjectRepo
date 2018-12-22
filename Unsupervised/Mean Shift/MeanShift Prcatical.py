import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np


X = np.array([[1,2],[2,3],[5,7],[5,8],[8,8],[1,3],[2,4],[8,10]])
colors = ['g','r','b','k','c']

class Mean_shift:
    def __init__(self,radius=4):
        self.radius = radius

    def fit(self,data):
        centroids = {}
        
        for i in range(len(data)):
            centroids[i] = data[i]
            
        while True:
            new_centroids = []
            
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))
                
            uniques = sorted(list(set(new_centroids)))
            
            prev_centroids = dict(centroids)
            
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
                
            optimized = True
            for i in centroids:
                if not np.array_equal(prev_centroids[i],centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
        self.centroids = centroids
        self.classifications = {}
        
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:    
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            
            self.classifications[classification].append(featureset)
                
        
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
clf = Mean_shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0],X[:,1],s=150,marker='o')

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],s=150,marker='x')
    
plt.show