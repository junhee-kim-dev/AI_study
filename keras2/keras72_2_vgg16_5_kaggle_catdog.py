import numpy as np

from keras.models import Sequential
from keras.layers import Dense, AveragePooling2D
import tensorflow as tf
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import datetime

SEED=333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from keras.applications import VGG16


np_path = 'c:/Study25/_data/_save_npy/keras44/'
x = np.load(np_path + 'keras44_01_x.npy')
y = np.load(np_path + 'keras44_01_y.npy')
test = np.load(np_path + 'keras44_01_test.npy')
print('로드 됨.')

print(x.shape)  #(25000, 100, 100, 3)
print(y.shape)  #(25000,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42,
)

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
)

##########
# False와 AveragePooling2D 비교
# vgg16.trainable=False   # 가중치 동결
vgg16.trainable=True   # 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(AveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

##########
# Flatten 과 AveragePooling2D 비교

model.copile(loss='sparse_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, verbose=2, )

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(f"[Final] Loss {loss:.4f} | Acc {acc:.4f}")