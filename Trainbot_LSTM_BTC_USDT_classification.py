from numpy import *
import numpy as np
import pandas as pd
import time
import datetime
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
#from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, MaxPool2D, BatchNormalization,Embedding
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import signal,sys
def def_handler(sig,frame):
    print('\n\n[!] Saliendo ...')
    model.save('/Users/felix/Desktop/Bot_cripto/lstm_model_BTC/BTC_USDT_1m_classification.h5')
    sys.exit(1)
signal.signal(signal.SIGINT,def_handler)
np.random.seed(2)
tf.random.set_seed(2)
df = pd.read_json('/Users/felix/Desktop/Bot_cripto/user_data/data/binance/futures/BTC_USDT-1m-futures.json')
df.columns=['data','open','high','low','close','volume']
def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
     X = []
     y = []
     start = start + window
     if end is None:
         end = len(dataset) - horizon
     for i in range(start, end):
         indices = list(range(i-window, i))
         X.append(dataset[indices])
         indicey = list(range(i, i+horizon))
         y.append(target[indicey])
     return np.array(X), np.array(y)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def serie_lstm(data,n_past,n_future):
    train_X=[]
    for i in range(n_past,len(data)-n_future+1):
        train_X.append(data[i-n_past:i,0:data.shape[1]])
    return np.array(train_X)
def Scaler_fit_lstm(data):
    scalers = {}
    for i in range(data.shape[1]):
        scalers[i] = StandardScaler()
        data[:, i, :] = scalers[i].fit_transform(data[:, i, :]) 
    #scalers=StandardScaler()
    #data=scalers.fit_transform(data)
    return data,scalers
def Scaler_fit_test_lstm(scalers,data):
    for i in range(data.shape[1]):
        data[:,i,:]=scalers[i].transform(data[:,i,:])
    #data=scalers.transform(data)
    return data
def Scaler_fit_lstm_y(data):
    scaler=StandardScaler()
    data=scaler.fit_transform(data)
    return data,scaler
def Scaler_fit_lstm_y_train(scaler,data):
    data=scaler.transform(data)
    return data

def CreateTargets(data, offset):
    
    y = []
    
    
    for i in range(0, len(data)-offset):
        current = float(data.loc[i,'close'])
        comparison = float(data.loc[i+offset,'close'])
        
        if current<comparison:
            y.append(1)

        elif current>=comparison:
            y.append(0)
            
    return y
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):  # Time Series setps (0-99,100-200,,,,) any steps
        a = dataset[i:(i+time_step), 0]   
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
print("Creando Variables...")
df=df.dropna().reset_index().drop(columns=['index','data'])
X,y=create_dataset(df,1)
print("Preparando data...")
X,y=custom_ts_multi_data_prep(df.values,y,0,None,10,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
#values=data.values
#train = values[:int(len(data)*0.8), :]
#test = values[int(len(data)*0.8):, :]
# split into input and outputs
#X_train, y_train = train[:, :-1], train[:, -1]
#X_test, y_test = test[:, :-1], test[:, -1]
X_train,scalers_x=Scaler_fit_lstm(X_train)
X_test = Scaler_fit_test_lstm(scalers_x,X_test)
#y_train,scalers_y = Scaler_fit_lstm_y(y_train)
#y_test = Scaler_fit_lstm_y_train(scalers_y,y_test)
#X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
print("Iniciando el modelo")
# # Build the LSTM Stack model
T=X_train.shape[1]
N=X_train.shape[2]
LAYERS=[128,64,32,1]
DP=0.6
RDP=0.4
model = Sequential()
model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(T, N)))
model.add(tf.keras.layers.MaxPooling1D())
model.add(LSTM(units=LAYERS[0],
               activation='relu', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[1],
               activation='relu', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[2],
               activation='relu', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=LAYERS[3], activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary() 
history=model.fit(X_train,y_train, batch_size=1024,epochs=50,verbose=1,validation_data=(X_test,y_test))
model.save('/Users/felix/Desktop/Bot_cripto/lstm_model_BTC/BTC_USDT_1m_classification.h5')
#plt.plot(model.predict(X_train))
#plt.plot(y_train)
#plt.show()
#plt.plot(model.predict(X_test))
#plt.plot(y_test)
#plt.show()
print(model.predict(X_test))