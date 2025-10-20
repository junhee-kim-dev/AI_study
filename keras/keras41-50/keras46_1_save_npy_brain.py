from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import time
import datetime

train_datagen = ImageDataGenerator(
    rescale=1/255.,
)
test_datagen = ImageDataGenerator(
    rescale=1/255.,
)

path = './_data/image/brain/'
path_train = ''.join([path, 'train/'])
path_test = ''.join([path, 'test/'])

train = train_datagen.flow_from_directory(
    path_train, target_size=(200,200), batch_size=10,
    class_mode='binary', color_mode='grayscale',
    shuffle=True 
)

test = test_datagen.flow_from_directory(
    path_test, target_size=(200,200), batch_size=10,
    class_mode='binary', color_mode='grayscale'
)

all_x_train = []
all_y_train = []
for i in range(len(train)) :
    x_train_bat, y_train_bat = train[i]
    all_x_train.append(x_train_bat)
    all_y_train.append(y_train_bat)
    
all_x_test = []
all_y_test = []
for i in range(len(test)):
    x_test_bat, y_test_bat = test[i]
    all_x_test.append(x_test_bat)
    all_y_test.append(y_test_bat)
        
x_train = np.concatenate(all_x_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)
x_test = np.concatenate(all_x_test, axis=0)
y_test = np.concatenate(all_y_test, axis=0)

npy_path = 'c:/Study25/_data/_save_npy/keras46/'
np.save(npy_path + 'brain(200,200)_x_train', arr=x_train)
np.save(npy_path + 'brain(200,200)_y_train', arr=y_train)
np.save(npy_path + 'brain(200,200)_x_test', arr=x_test)
np.save(npy_path + 'brain(200,200)_y_test', arr=y_test)