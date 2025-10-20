from sklearn.datasets import load_digits

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt

datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  #(1797, 64)
# print(y)
# print(y.shape)  #(1797,)
# print(x[0])
# aaa= x[1].reshape(8,8)
# print(aaa)
# print(y[1])
# exit()
# print(datasets.DESCR)
# print(datasets.feature_names)

print(np.max(x), np.min(x))
print(np.max(y), np.min(y))
# x_show = x.reshape(-1,8,8,1)
# import random
# n = random.randint(1,200)
# print(n)
# print(y[n])
# plt.imshow(x_show[n], 'gray')
# # plt.show()

# exit()

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
)

ms = MaxAbsScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# exit()
x_train = x_train.reshape(-1,8,8)
x_test = x_test.reshape(-1,8,8)
from keras.layers import Dropout, Flatten, Conv2D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(8,8,1), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(LSTM(128, input_shape=(8,8)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, restore_best_weights=True, verbose=1
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './_save/keras41/11digits/'
filename = '({epoch:04d}-{val_loss:.4f}).hdf5'
filepath = ''.join([path1, 'k41_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath,
    verbose=1
)

s_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=10000, batch_size=64,
    verbose=2, validation_split=0.2,
    callbacks=[es, mcp]
)
e_time = time.time()
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
results_arg = np.argmax(results, axis=1)
acc = accuracy_score(y_test_arg, results_arg)
f1 = f1_score(y_test_arg, results_arg, average='macro')
print('CNN 11')
print('Loss :', loss[0])
print('acc :', acc)
print('time :', np.round(e_time - s_time, 1), 'sec')


# print('Acc  :', acc)
# print('F1   :', f1)
# images = datasets.images
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 2))  # 가로 길게
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(images[i], cmap='gray')  # 흑백 이미지
#     plt.axis('off')                     # 축 제거
#     plt.title(str(y[i]), fontsize=10)

# plt.tight_layout()
# plt.show()

# LSTM 11
# Loss : 0.19004195928573608
# acc : 0.9444444444444444
# time : 15.9 sec