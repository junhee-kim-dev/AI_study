from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

# path = './Study25/_data/kaggle/CatDog/'
# trn_path = ''.join([path, 'train2/'])
# tst_path = ''.join([path, 'test2/'])

# train = ImageDataGenerator(
#     rescale=1/255.,
# )

# test = ImageDataGenerator(
#     rescale=1/255.,
# )

# s = time.time()
# xy_train = train.flow_from_directory(
#     trn_path,
#     target_size=(200,200),
#     batch_size=100,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
#     seed=42
# )

# test = test.flow_from_directory(
#     tst_path,
#     target_size=(200,200),
#     batch_size=32,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
#     seed=42
# )
# print(xy_train[0][0].shape)     # (100, 200, 200, 3)
# print(xy_train[0][1].shape)     # (100,)
# print(len(xy_train))            # 250

# ###################모든 수치화된 batch 데이터를 하나로 합치기#######
# all_x = []
# all_y = []
# for i in range(len(xy_train)) : 
#     x_batch, y_batch = xy_train[i]
#     all_x.append(x_batch)
#     all_y.append(y_batch)

# ###################리스트를 하나의 numpy 배열로 합친다.(concatenate)
# x = np.concatenate(all_x, axis=0)
# y = np.concatenate(all_y, axis=0)
# path_NP = './Study25/_data/kaggle/CatDog/'

# np.save(path_NP + '04_x_train.npy', arr=x)
# np.save(path_NP + '04_y_train.npy', arr=y)

# exit()

path_NP = './Study25/_data/kaggle/CatDog/'

x = np.load(path_NP + '04_x_train.npy')
y = np.load(path_NP + '04_y_train.npy')

print('x.shape :',x.shape)      #x.shape : (25000, 200, 200, 3)
print('y.shape :',y.shape)      #y.shape : (25000,)

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.8,
                                           shuffle=True,
                                           random_state=50,
                                           stratify=y,
                                           )

x_trn = x_trn.reshape(-1,x_trn.shape[1]*x_trn.shape[2]*x_trn.shape[3])  
x_tst = x_tst.reshape(-1,x_tst.shape[1]*x_tst.shape[2]*x_tst.shape[3])
print(x_trn.shape, x_tst.shape)     # (20000, 120000) (5000, 120000)

##########################################################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
x_train_1 = lda.fit_transform(x_trn, y_trn)
x_test_1 = lda.transform(x_tst)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train_1, y_trn)
score = xgb.score(x_test_1, y_tst)

print(f'n_components_1 :',score)    

print(np.cumsum(lda.explained_variance_ratio_))
