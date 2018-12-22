import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

style.use('ggplot')
df = pd.read_csv('EOD-DIS.csv',index_col=[0])
df.sort_values('Date',inplace=True)
df = df[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]
df['HL_PCT'] = (df['Adj_High'] - df['Adj_Close'])/df['Adj_Close']* 100
df['PCT_change'] = (df['Adj_Close'] - df['Adj_Open'])/df['Adj_Open']* 100

df = df[['Adj_Close','HL_PCT','PCT_change','Adj_Volume']]

forecast_col = 'Adj_Close'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

clf = LinearRegression()
clf.fit(X_train,y_train)
with open('linear.pickle', 'wb') as f:
    pickle.dump(clf,f)
 
pickle_in = open('linear.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,y_test)

forcast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = pd.to_datetime(df.iloc[-1].name)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

   
df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
