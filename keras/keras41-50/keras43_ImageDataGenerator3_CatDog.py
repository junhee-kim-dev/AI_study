from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import datetime
import time

path = './_data/kaggle/CatDog/'
trn_path = ''.join([path, 'train2/'])

tst_path = ''.join([path, 'test2/'])

train = ImageDataGenerator(
    rescale=1/255.,
)

test = ImageDataGenerator(
    rescale=1/255.,
)

s = time.time()
xy_train = train.flow_from_directory(
    trn_path,
    target_size=(200,200),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

test = test.flow_from_directory(
    tst_path,
    target_size=(200,200),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=42
)
print(xy_train[0][0].shape)     # (100, 200, 200, 3)
print(xy_train[0][1].shape)     # (100,)
print(len(xy_train))            # 250
e1 = time.time()

###################모든 수치화된 batch 데이터를 하나로 합치기#######
all_x = []
all_y = []
for i in range(len(xy_train)) : 
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)

e2 = time.time()

###################리스트를 하나의 numpy 배열로 합친다.(concatenate)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

print('x.shape :',x.shape)      #x.shape : (25000, 200, 200, 3)
print('y.shape :',y.shape)      #y.shape : (25000,)

e3 = time.time()

# print(all_x)
# print(all_y)
print('time:', np.round(e1 - s,2), 'sec')   # time: 1.18 sec
print('time:', np.round(e2 - e1,2), 'sec')  # time: 23.55 sec
print('time:', np.round(e3 - e2,2), 'sec')  # time: 4.89 sec