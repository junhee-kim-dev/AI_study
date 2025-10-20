from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from tensorflow.keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50
)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# exit()
x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)
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
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, restore_best_weights=True, verbose=1
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './_save/keras41/03diabetes/'
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
rmse = np.sqrt(loss[0])
r2 = r2_score(y_test, results)

print('######03######')
print('CNN')
print('RMSE :', rmse)
print('R2 :', r2)
print('time :', np.round(e_time - s_time, 1), 'sec')


# RMSE : 50.469586873866426
# R2 : 0.542062781751149

# MinMaxScaler
# RMSE : 50.73969453556555
# R2 : 0.5371480334293155

# MaxAbsScaler
# RMSE : 49.63791010045271
# R2 : 0.5570309888342313

# StandardScaler
# RMSE : 53.714245920833235
# R2 : 0.4812891184686696

# RobustScaler
# RMSE : 52.68465016758913
# R2 : 0.500983793006675

# CNN
# RMSE : 53.5036460133209
# R2 : 0.48534858314642704
# time : 11.9 sec