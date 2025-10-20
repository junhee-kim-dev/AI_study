from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True, width_shift_range=0.1,
    zoom_range=1.2, rotation_range=20,
    fill_mode='nearest'
)
augment_size=40000
rad_idx = np.random.randint(60000, size=40000)
aug_x = x_train[rad_idx].copy()
aug_y = y_train[rad_idx].copy()

aug_x = aug_x.reshape(-1,28,28,1)
path = 'c:/STUDY25/_data/_save_img/02_mnist/'

s_time = time.time()
print('#####저장 시작#####')
aug_x = datagen.flow(
    aug_x, aug_y, batch_size=augment_size, shuffle=False, save_to_dir=path
).next()[0]
print('#####저장 끝#####')
e_time = time.time()
print('time :', np.round(e_time - s_time,1), 'sec')

x_train = x_train/255.
x_test = x_test/255.
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
