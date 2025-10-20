from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from tensorflow.keras.callbacks import EarlyStopping

path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
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

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='linear'))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=False
)


model.compile(loss='mse', optimizer='adam')
srt_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=30000, batch_size=32,
          verbose=2, validation_split=0.2,
          callbacks=[es,],
          )
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)
# print('걸린 시간 :', end_time - srt_time, '초')

y_submit = model.predict(test_csv)
test_csv_copy = test_csv.copy()
test_csv_copy[['casual', 'registered']] = y_submit
test_csv_copy.to_csv(path + 'new_test_1.csv', index=False)


# RMSE : 95.68644369926442
# R2 : 0.3849638144401447


# MinMaxScaler
# RMSE : 94.75008245378999
# R2 : 0.4237545378319282

# MaxAbsScaler
# RMSE : 95.84237446029549
# R2 : 0.4068468826743233

# StandardScaler
# RMSE : 95.36379767107641
# R2 : 0.41974378420078196

# RobustScaler
# RMSE : 95.36379767107641
# R2 : 0.41974378420078196