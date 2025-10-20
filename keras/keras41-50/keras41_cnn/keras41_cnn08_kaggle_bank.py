# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)
# exit()
from tensorflow.keras.layers import Dropout, Flatten, Conv2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(5,2,1), padding='same'))
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
path1 = './_save/keras41/08bank/'
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
acc = accuracy_score(y_test, results)
f1 = f1_score(y_test, results)

print('CNN 08')
print('Loss :', loss[0])
print('Acc  :', acc)
print('F1   :', f1)
print('time :', np.round(e_time - s_time, 1), 'sec')

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_1.csv', index=False)


# Loss : 0.3281714916229248
# Acc  : 0.862483715575484
# F1   : 0.6295601077287195


# CNN 08
# Loss : 0.33338749408721924
# Acc  : 0.86021147029418
# F1   : 0.6185515873015873
# time : 759.9 sec