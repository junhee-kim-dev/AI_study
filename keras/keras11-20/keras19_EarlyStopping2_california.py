import tensorflow as tf
# print(tf.__version__)
import numpy as np
# print(np.__version__)

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target
# print(x.shape)  #(20640, 8)
# print(y.shape)  #(20640, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=False,
)

hist = model.fit(
        x_train, y_train, epochs=100000, 
        batch_size=32, verbose=2, validation_split=0.2,
        callbacks=[es],
        )

print(hist.history)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('############')
# print('예측값 :', results)
print('RMSE :', rmse)
print('R2 :', r2)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('보스턴 레드삭스')
plt.xlabel('에포')
plt.ylabel('loss')
plt.grid()
plt.legend(loc='upper right')
plt.show()

#region

#최고 성능
# ############
# batch_size : 50
# epochs : 100
# loss : 0.6732680797576904
# RMSE : 0.8205291724362186
# R2 : 0.5018464741308137

#True
# ############
# RMSE : 0.8200559940156316
# R2 : 0.5024208535457415

#False
# ############
# RMSE : 0.8399125508040597
# R2 : 0.4780326979673245

#endregion