from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/CatDog_aug/'
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
    target_size=(100,100),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

test = test.flow_from_directory(
    tst_path,
    target_size=(100,100),
    batch_size=64,
    class_mode='binary',
    color_mode='rgb',
)
# print(xy_train[0][0].shape)     # (100, 200, 200, 3)
# print(xy_train[0][1].shape)     # (100,)
# print(len(xy_train))            # 250
e1 = time.time()

# print(test[0])
# exit()
###################모든 수치화된 batch 데이터를 하나로 합치기#######
all_x = []
all_y = []
for i in range(len(xy_train)) : 
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    
all_test = []
for i in range(len(test)) :
    test_batch, y_test_batch = test[i]
    all_test.append(test_batch)

e2 = time.time()

###################리스트를 하나의 numpy 배열로 합친다.(concatenate)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
test = np.concatenate(all_test, axis=0)

# print('x.shape :',x_train.shape)      #x.shape : (25000, 200, 200, 3)
# print('y.shape :',y_train.shape)      #y.shape : (25000,)

e3 = time.time()



####################저장한다.#######################################
np_path = 'c:/Study25/_data/_save_npy/keras44/'
np.save(np_path + 'keras44_01_200_x.npy', arr=x)
np.save(np_path + 'keras44_01_200_y.npy', arr=y)
np.save(np_path + 'keras44_01_200_test.npy', arr=test)
e4 = time.time()
# print(all_x)
# print(all_y)
print('time:', np.round(e1 - s,2), 'sec')           #time: 0.83 sec
print('time:', np.round(e2 - e1,2), 'sec')          #time: 16.83 sec
print('time:', np.round(e3 - e2,2), 'sec')          #time: 2.51 sec
print('time:', np.round(e4 - e3,2), 'sec')          #time: 18.96 sec