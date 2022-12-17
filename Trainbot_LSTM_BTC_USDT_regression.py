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
    model.save('/Users/felix/Desktop/Bot_cripto/lstm_model_BTC/BTC_USDT_1m_regression.h5')
    sys.exit(1)
signal.signal(signal.SIGINT,def_handler)
np.random.seed(2)
tf.random.set_seed(2)
df = pd.read_json('/Users/felix/Desktop/Bot_cripto/user_data/data/binance/BTC_USDT-1m.json')
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
def populate_indicators(dataframe):
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        #dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        ## Minus Directional Indicator / Movement
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        #dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        #dataframe['aroondown'] = aroon['aroondown']
        #dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        #keltner = qtpylib.keltner_channel(dataframe)
        #dataframe["kc_upperband"] = keltner["upper"]
        #dataframe["kc_lowerband"] = keltner["lower"]
        #dataframe["kc_middleband"] = keltner["mid"]
        #dataframe["kc_percent"] = (
        #    (dataframe["close"] - dataframe["kc_lowerband"]) /
        #    (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        #)
        #dataframe["kc_width"] = (
        #    (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        #)

        # # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        #rsi = 0.1 * (dataframe['rsi'] - 50)
        #dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        #dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        #dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        #stoch_fast = ta.STOCHF(dataframe)
        #dataframe['fastd'] = stoch_fast['fastd']
        #dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        #dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        #bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        #dataframe['bb_lowerband'] = bollinger['lower']
        #dataframe['bb_middleband'] = bollinger['mid']
        #dataframe['bb_upperband'] = bollinger['upper']
        #dataframe["bb_percent"] = (
        #    (dataframe["close"] - dataframe["bb_lowerband"]) /
        #    (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        #)
        #dataframe["bb_width"] = (
        #    (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        #)

        ##Bollinger Bands - Weighted (EMA based instead of SMA)
        #weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #    qtpylib.typical_price(dataframe), window=20, stds=2
        #)
        #dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        #dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        #dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        #dataframe["wbb_percent"] = (
        #    (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #    (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        #)
        #dataframe["wbb_width"] = (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        
        # # EMA - Exponential Moving Average
        #dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        #dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        #dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        #dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        #dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        #dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        #dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        #dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        #dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        #dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        #dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        #dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        #dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        #dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        #dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        ## # Inverted Hammer: values [0, 100]
        #dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        #dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        #dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        #dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        #dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        #dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        #dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        #dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        #dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        #dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        #dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        #dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        #dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        #dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        #dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        #dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        #dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        #heikinashi = qtpylib.heikinashi(dataframe)
        #dataframe['ha_open'] = heikinashi['open']
        #dataframe['ha_close'] = heikinashi['close']
        #dataframe['ha_high'] = heikinashi['high']
        #dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe
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
print("Creando Variables...")
data=populate_indicators(df)
print(data)
size=len(data)
data=data.dropna().reset_index().drop(columns=['index','high','low','open','data'])
y=data['close'].values
print("Preparando data...")
X,y=custom_ts_multi_data_prep(data.values,y,0,None,10,1)
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
LAYERS=[64,64,64,1]
DP=0.6
RDP=0.4
model = Sequential()
model.add(tf.keras.layers.Conv1D(32, 3, activation='linear', input_shape=(T, N)))
model.add(tf.keras.layers.MaxPooling1D())
model.add(LSTM(units=LAYERS[0],
               activation='linear', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[1],
               activation='linear', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[2],
               activation='linear', recurrent_activation='hard_sigmoid',
               kernel_regularizer='l2', recurrent_regularizer='l2',
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(Dense(units=LAYERS[3], activation='linear'))

model.compile(loss='mse',optimizer='adam')
model.summary() 
history=model.fit(X_train,y_train, batch_size=1024,epochs=200,verbose=1,validation_data=(X_test,y_test))
model.save('/Users/felix/Desktop/Bot_cripto/lstm_model_BTC/BTC_USDT_1m_regression.h5')
plt.plot(model.predict(X_train))
plt.plot(y_train)
plt.show()
plt.plot(model.predict(X_test))
plt.plot(y_test)
plt.show()
print(model.predict(X_test))