from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]).T
y = np.array([1,2,3,4,5,6,7,8,9,10]).T

## <train_test_split>
x_train, x_test, y_train, y_test = train_test_split(x, y,                 #x를 앞에 두개로, y를 뒤에 두개로 나눌 수 있음
                                                    train_size=0.7,       #test_size로 생략가능, 디폴트 : 0.75
                                                    test_size=0.3,        #train_size로 생략가능, 디폴트 : 0.25
                                                    shuffle=True,         #True / False, 디폴트 = True
                                                    random_state=2632507999,     #랜덤 난수 : 사실 랜덤은 아님, 생략하면 그냥 막나옴
                                                    )
print(x_train)
print(x_test)
print(y_train)
print(y_test)

## <numpy data slicing>
# x_train = x[:7]
# y_train = y[:7]

# x_test = x[7:]
# y_test = y[7:]

# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('#####################')
print('loss :', loss)
print('예측값 :', result)

# #####################
# loss : 7.768600758344124e-13
# 예측값 : [[10.999999]]