# 18-1 카피

#EarlyStopping : 최소값을 기준으로 몇번 참고 갱신되지 않으면 멈추겠다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
    shuffle=True, random_state=38
)

model = Sequential()
model.add(Dense(50, input_dim = 13, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',                 # 최소값 min, 최대값 max, 알아서 찾아줘: auto
        patience=20,                  # 파라미터
        restore_best_weights=False,    # default는 False 
        )

hist = model.fit(
        x_train, y_train, 
        epochs=1000000, batch_size=1, 
        verbose=2, validation_split=0.2,
        callbacks=[es],               # 다른 친구들도 있음
        )

#region
# print('========hist========')
# print(hist)
# print('====hist.history====')
# print(hist.history)
# print('======= loss =======')
# print(hist.history['loss'])
# print('===== val_loss =====')
# print(hist.history['val_loss'])
#endregion

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
plt.legend(loc='upper right')   #우측 상단에 라벨 표시
plt.grid()                      #격자 표시
plt.show()

#region(결과)

#이전 최고
# ###################
# RMSE : 5.60012539314481
# R2 : 0.6256338850238707

#true
# ###################
# RMSE : 6.044281632593038
# R2 : 0.5526805626496353

#False
# ###################
# RMSE : 6.200505008434324
# R2 : 0.5332187582289023

#endregion