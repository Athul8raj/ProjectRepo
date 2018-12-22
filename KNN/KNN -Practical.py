import pandas as pd
import numpy as np
import random
from collections import Counter

dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data,predict,k=3):
    if len(data)>= k :
        raise Exception
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence  

#result = k_nearest_neighbors(dataset,new_features,3)
#print(result)

accuracies = []  
for i in range(5):    
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
    full_data = breast_cancer_df.astype(float).values.tolist()
    random.shuffle(full_data) 
    
    test_size = 0.2
    train_set = {2: [],4: []}
    test_set = {2: [],4: []}
    
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
        
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        
    correct =  0
    total = 0
    
    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set,data,k=5)
            if group == vote:
                correct +=1
            total +=1
            
#    print("Accuracy:",(correct/total))
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))