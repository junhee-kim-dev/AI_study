from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

datagen = ImageDataGenerator(
    rescale=1/255., horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.1, fill_mode='nearest'
)
# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)
# print(y_test.shape)     #(10000, 1)
aug_x = x_train.copy()
aug_y = y_train.copy()

x_train = x_train/255.
x_test = x_test/255.

s_time = time.time()
print('#####저장 시작#####')

path = 'c:/STUDY25/_data/_save_img/04_cifar100/'
augment = datagen.flow(
    aug_x, aug_y, batch_size=100, shuffle=False, save_to_dir=path
)

print('#####저장 끝#####')
e_time = time.time()
print('time :', np.round(e_time - s_time,1), 'sec')