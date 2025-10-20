from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, 
    shuffle=True, random_state=444
)

model = Sequential()
model.add(Dense(50, input_dim = 13, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(
            x_train, y_train, 
            epochs=100, batch_size=1, 
            verbose=3, validation_split=0.3
            )
print('========hist========')
print(hist)
print('====hist.history====')
print(hist.history)
print('======= loss =======')
print(hist.history['loss'])
print('===== val_loss =====')
print(hist.history['val_loss'])

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.figure(figsize=(9,6))       #9*6(인치) 사이즈
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.plot(hist.history['loss'], c= 'red', label='loss')  #리스트[]는 x축 순서가 이미 있기 때문에 순서를 넣어주지 않아도 됨
plt.plot(hist.history['val_loss'], c= 'blue', label='val_loss')  
plt.title('보스턴 loss')
plt.xlabel('epoch')             #x축 라벨
plt.ylabel('loss')              #y축 라벨
plt.grid()                      #격자 표시
plt.legend(loc='upper right')   #우측 상단에 라벨 표시
plt.show()

"""
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

#전
# ###################
# RMSE : 4.827135674076012
# R2 : 0.756898103912625

#후
# ###################
# RMSE : 5.60012539314481
# R2 : 0.6256338850238707
"""