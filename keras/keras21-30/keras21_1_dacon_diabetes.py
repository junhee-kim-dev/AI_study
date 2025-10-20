# dacon, 데이터 파일 별도
# https://dacon.io/competitions/official/236068/leaderboard

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

# print(train_csv.info)   #[652 rows x 9 columns]>
# print(test_csv.info)    #[116 rows x 8 columns]>
x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

r = 55 # random.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=r
)


model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=100, restore_best_weights=True
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=10000, batch_size=32,
    verbose=2, validation_split=0.1,
    callbacks=[es]                  # ''는 값 / 없으면 변수
)
end_time = time.time()

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

# print(y_predict)
print('RanNo. :', r)
print('Loss   :', round(results[0], 4))
print('Acc    :', round(accuracy_score, 4))

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['Outcome'] = y_submit
submission_csv.to_csv(path + 'submission_1.csv', index=False)

# exit()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('당뇨병 예측')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()


# _1
# RanNo. : 55
# Loss   : 0.357
# Acc    : 0.8788

# RanNo. : 55
# Loss   : 0.3962
# Acc    : 0.9091

