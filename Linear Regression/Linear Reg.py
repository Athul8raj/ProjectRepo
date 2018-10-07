import pandas as pd
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')
df = pd.read_csv('EOD-DIS.csv')
df = df[['Date','Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]
df['HL_PCT'] = (df['Adj_High'] - df['Adj_Close'])/df['Adj_Close']* 100
df['PCT_change'] = (df['Adj_Close'] - df['Adj_Open'])/df['Adj_Open']* 100

df = df[['Date','Adj_Close','HL_PCT','PCT_change','Adj_Volume']]

forecast_col = 'Adj_Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.001*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label','Date'],1))
X = preprocessing.scale(X)
X_lately = X[:forecast_out]
X = X[forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.1)

clf = LinearRegression()
clf.fit(X_train,y_train)
with open('linear.pickle', 'wb') as f:
    pickle.dump(clf,f)

pickle_in = open('linear.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,y_test)

forcast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1,0]
last_unix = datetime.strptime(last_date,'%Y-%m-%d')
one_day = 86400
next_unix = last_unix + timedelta(seconds=one_day)

for i in forcast_set:
    next_date = next_unix
    next_unix += timedelta(seconds=one_day)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

   
df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
