from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score


def preprocess(X,normalize=True):
    if not normalize:
        X = preprocessing.scale(X)
    else:
        X = preprocessing.Normalizer(norm='l2').fit_transform(X) 
    return X

def label_encoder(y,one_hot=True):
    if not one_hot:
        y = preprocessing.LabelEncoder().fit(y)
    else:
        y = preprocessing.OneHotEncoder().fit(y)
    return y
        
def cross_validation(X,y,normal=True,classifier=svm.svc):
    if normal:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        return X_train,X_test,y_train,y_test
    else:
        parameters = {'kernel':('linear','poly','rbf'), 'C':[1, 10]}
        clf = GridSearchCV(classifier,parameters,cv=5)
        return clf

params = {'objective' : 'binary : logistic','max_depth':2,'learning_rate': 0.1,'silent':0.01,'n_estimators':100}

KNN = KNeighborsClassifier()
svc = svm.SVC()
gbc = GradientBoostingClassifier()
xgb = XGBClassifier(**params)
    
class Classifier:
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def train(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self._classifiers.fit(self.X_train,self.y_train)
        clf = self._classifiers.score(X_test,y_test)
        vote = []
        vote.append([clf,self._classifiers])
        max_score = sorted(vote,reverse=True)[0][0]
        best_classifier = sorted(vote,reverse=True)[0][1]
        
        return max_score,best_classifier
    
    def predict(self,X_test,y_test):
        best_accuracy,best_classifier = self.train(self.X_train,self.y_train,X_test,y_test)
        prediction = best_classifier.predict(X_test)
        correct = 0
        for i in range(len(prediction)):
            if prediction == y_test:
                correct +=1
        error = 1 - (correct/len(prediction))
        return accuracy_score(y_test,prediction),error
    

classifier = Classifier(KNN,svc,gbc,xgb)
X = preprocess(X)
X_train,y_train,X_test,y_test = cross_validation(X,y,normal=True)
max_score,best_classifier = classifier.train(X_train,y_train,X_test,y_test)
classifier.predict(X_test,y_test)

        
         
        
        
        