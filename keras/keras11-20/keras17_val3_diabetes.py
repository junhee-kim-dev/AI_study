from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_diabetes
import numpy as np

dataset = load_diabetes()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)
x = dataset.data
y = dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape)     #x.shape (442,10), y.shape (442,)
x_tr, x_ts, y_tr, y_ts = train_test_split(
    x, y, train_size=0.7, test_size=0.3, 
    shuffle=True, random_state=9)

model = Sequential()
model.add(Dense(50, input_dim=10))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_tr, y_tr, epochs=100, batch_size=1, verbose=1, validation_split=0.2)

loss = model.evaluate(x_ts, y_ts)
results = model.predict(x_ts)

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_ts, results)
r2 = r2_score(y_ts, results)

print('############')
print('loss :', loss)
# print('예측값 :', results)
print('RMSE :', rmse)
print('R2 :', r2) #0.62 이상    

#전
# ############
# loss : 2206.406982421875
# RMSE : 46.97240351916079
# R2 : 0.6007423516230339

#후
# ############
# loss : 2236.36083984375
# RMSE : 47.29017709743048
# R2 : 0.5953220333563718