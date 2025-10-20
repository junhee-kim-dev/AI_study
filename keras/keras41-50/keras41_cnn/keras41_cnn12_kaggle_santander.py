# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import datetime
import time

path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  #(200000, 200)
print(y.shape)  #(200000,)

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y
)

ms = MaxAbsScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)
test_csv = ms.transform(test_csv)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# exit()
x_train = x_train.reshape(-1,10,10,2)
x_test = x_test.reshape(-1,10,10,2)
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(10,10,2), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, restore_best_weights=True, verbose=1
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './_save/keras41/12santander/'
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
results = np.round(results)
f1 = f1_score(y_test, results)

# y_submit = model.predict(test_csv)
# sub_csv['target'] = y_submit
# filename1 = ''.join(['submission_', date,'.csv'])
# sub_csv.to_csv(path+ filename1)
# print('File :', filename1)
# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# # print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# if gpus:
#     print('GPU 있다~')
# else:
#     print('GPU 없다~')
print('CNN 12')
print('F1  :', f1)
print('time:', np.round(e_time - s_time, 2), 'sec')


# File : submission_0608_2242.csv
# GPU 없다~
# F1  : 0.362078599366735
# time: 259.94 sec


# File : submission_0608_2248.csv
# GPU 있다~
# F1  : 0.4329590488771466
# time: 220.57 sec

# CNN 12
# F1  : 0.35077739855896856
# time: 498.55 sec