from tensorflow.keras.layers import LSTM,Dropout,Activation,BatchNormalization,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random,time


main_df = pd.DataFrame()
forecast_time = 3
currency_to_predict = 'BCH-USD'
seq_len = 60
Epochs = 5
Batch_size = 64
name = f'{currency_to_predict}-seq-{forecast_time}-pred-{int(time.time())}'

def classify(current,future):
    if float(future) > float(current):
        return 1
    return 0


def preprocess(df):
    for col in df.columns:
        if col != 'Target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
      
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=seq_len)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days),i[-1]])  
            
    random.shuffle(sequential_data)
    
    buys,sells = [],[]
    
    for seq,target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        else:
            buys.append([seq,target])
    
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys),len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]   
    
    sequential_data = buys+sells
    
    random.shuffle(sequential_data)
    
    X,y = [],[]
    
    for seq,target in sequential_data:
        X.append(seq)
        y.append(target)
       
    return np.array(X),y

currencies = ['BCH-USD','LTC-USD','ETH-USD','BTC-USD']

for currency in currencies:
    dataset = f'crypto_data/{currency}.csv'
    
    df = pd.read_csv(dataset,names=['Time','Low','High','Open','Close','Volume'])
    df.rename(columns={'Close':f'{currency}_close','Volume':f'{currency}_volume'},inplace=True)
    
    df.set_index('Time',inplace=True)
    df = df[[f'{currency}_close',f'{currency}_volume']]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df['Forecast'] = main_df[f'{currency_to_predict}_close'].shift(-forecast_time)

main_df['Target'] = list(map(classify,main_df[f'{currency_to_predict}_close'],main_df['Forecast']))

main_df.drop(['Forecast'],1,inplace=True)
main_df.dropna(inplace=True)

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]


validation_df = main_df.loc[(main_df.index >= last_5pct)]
main_df = main_df.loc[(main_df.index < last_5pct)]
train_x,train_y = preprocess(main_df)
validation_x,validation_y = preprocess(validation_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True)) 
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
tensorboard =TensorBoard(log_dir=f'logs/{name}') 

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"

checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y,batch_size=Batch_size,epochs=Epochs,validation_data=(validation_x, validation_y),callbacks=[tensorboard, checkpoint])

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save(f"models/{name}")
