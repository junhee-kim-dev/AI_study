import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) 

x = np.array([[1,2,3], 
              [2,3,4], 
              [3,4,5], 
              [4,5,6], 
              [5,6,7], 
              [6,7,8],  
              [7,8,9],  
             ])
y = np.array([ 4, 5, 6, 7, 8, 9,10]) 

#2. 모델
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))
model.add(Bidirectional(LSTM(10), input_shape=(3,1)))
# model.add(GRU(10, input_shape=(3,1)))
# model.add(Bidirectional(GRU(10), input_shape=(3,1)))
# model.add(SimpleRNN(10, input_shape=(3,1)))
# model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, ))
model.summary()

'''
#################### param 개수 ######################
RNN         -> 120 / 205
Bidirection -> 240 / 395

GRU         -> 390 / 475
Bidirection -> 780 / 935

LSTM        -> 480 / 565
Bidirection -> 960 / 1115

'''





















