import logging
from time import time
from typing import Any
from typing import Any, Dict, List, Literal, Optional, Tuple
from pandas import DataFrame
import sys, os
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
import numpy.typing as npt
from numpy import *
import numpy as np
import pandas as pd
#import time
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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, MaxPool2D, BatchNormalization,Embedding, Conv1D
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
#import talib
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras import callbacks
logger = logging.getLogger(__name__)


class LSTMModelRegressor(IFreqaiModel):
    """
    Base class for TensorFlow type models.
    User *must* inherit from this class and set fit() and predict().
    """
    def custom_ts_multi_data_prep_x(self, dataset, start, end, window, horizon):
        X = []
        start = start + window
        if end is None:
            end = len(dataset) - horizon
        for i in range(start, end):
            indices = list(range(i-window, i))
            X.append(dataset[indices])
        return np.array(X)
    def custom_ts_multi_data_prep(self, dataset, target, start, end, window, horizon):
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
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Most regressors use the same function names and arguments e.g. user
        can drop in LGBMRegressor in place of CatBoostRegressor and all data
        management will be properly handled by Freqai.
        :param data_dictionary: Dict = the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 15, 
                                        restore_best_weights = True)
        X_train = data_dictionary['train_features'].values
        y_train = data_dictionary['train_labels']['&s-close_price'].values
        X_test = data_dictionary['test_features'].values
        y_test = data_dictionary['test_labels']['&s-close_price'].values
        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
        T=X_train.shape[1]
        N=X_train.shape[2]
        LAYERS=[128,128,64,32,1]
        DP=0.4
        RDP=0.4
        model = Sequential()
        model.add(Conv1D(filters=LAYERS[0], kernel_size=10,
                         padding="same",
                         activation = 'tanh', input_shape=(T,N)))
        model.add(LSTM(units=LAYERS[1], input_shape=(T,N),
                    activation='tanh', recurrent_activation='sigmoid',
                    kernel_regularizer='l2', recurrent_regularizer='l2',
                    dropout=DP, recurrent_dropout=RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(LSTM(units=LAYERS[2],
                    activation='tanh', recurrent_activation='sigmoid',
                    kernel_regularizer='l2', recurrent_regularizer='l2',
                    dropout=DP, recurrent_dropout=RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(LSTM(units=LAYERS[3],
                    activation='tanh', recurrent_activation='sigmoid',
                    kernel_regularizer='l2', recurrent_regularizer='l2',
                    dropout=DP, recurrent_dropout=RDP,
                    return_sequences=True, return_state=False,
                    stateful=False, unroll=False
                    ))
        model.add(Dropout(0.2))
        model.add(Flatten())
        #model.add(Dense(units=LAYERS[4], activation='linear'))
        #model.add(Dense(units=LAYERS[5], activation='linear'))
        model.add(Dense(units=LAYERS[4], activation='linear'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
        ),metrics=[tf.keras.metrics.RootMeanSquaredError()])
        model.fit(X_train,y_train, batch_size=64,epochs=60,verbose=1,validation_data=(X_test,y_test),callbacks =[earlystopping])
        plt.plot(model.predict(X_test))
        plt.plot(y_test)
        plt.show()
        return model
    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")
        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date} --------------------")
        # split data into train/test data.
        data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        # normalize all data based on train_dataset only
        data_dictionary = dk.normalize_data(data_dictionary)

        # optional additional data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(data_dictionary['train_features'])} data points")
        model = self.fit(data_dictionary, dk)
        end_time = time()
        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")
        return model
    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.normalize_data_from_metadata(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk)
        X_pred = dk.data_dictionary["prediction_features"].values
        predictions = self.model.predict(X_pred.reshape(X_pred.shape[0],1,X_pred.shape[1]))
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))

        pred_df = DataFrame(predictions, columns=dk.label_list)
        pred_df = dk.denormalize_labels_from_metadata(pred_df)
        print('Este es el print_df')
        print(pred_df)
        return (pred_df, dk.do_predict)
