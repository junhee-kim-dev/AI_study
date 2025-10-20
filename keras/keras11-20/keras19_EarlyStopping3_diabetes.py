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
# print(x.shape)  (442, 10)
# print(y.shape)  (442,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50
)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=False,
)

st_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=32,
    verbose=2, validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

# print('loss :', loss)
print('RMSE :', rmse)
print('R2 :', r2)
# print('걸린 시간:', end_time - st_time, '초')

# plt.rcParams['font.family'] = 'Malgun Gothic' 
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.grid()
# plt.title('보스턴')
# plt.legend(loc='upper right')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

#region

#종전 최고 성능
# RMSE : 50.98870126067147
# R2 : 0.5873367235371451

#True
# RMSE : 49.69274982093087
# R2 : 0.5560515933810308

#False
# RMSE : 50.928447719416845
# R2 : 0.533697974863679

#endregion