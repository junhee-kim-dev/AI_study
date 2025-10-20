import numpy as np
import pandas as pd

npy_path = './_data/kaggle/jena/dataset/'
x = np.load(npy_path + '(144steps)x.npy')
y = np.load(npy_path + '(144steps)y.npy')
test = np.load(npy_path + '(144steps)test.npy')
# lets_test = np.load(npy_path + '(144steps)lets_test.npy')
submit_csv = pd.read_csv('./_data/kaggle/jena/submission_file.csv')

# print(x)
# print(y)
# print(test)
# print(x.shape)              #(69996, 144, 13)
# print(y.shape)              #(69996, 144)
# print(test.shape)           #(144, 13)
# exit()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import time
import datetime

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, shuffle=True
)

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])
test = test.reshape(-1, test.shape[0]*test.shape[1])

rs = RobustScaler()
rs.fit(x_train)
x_train = rs.transform(x_train)
x_test = rs.transform(x_test)
test = rs.transform(test)

x_train = x_train.reshape(-1, 144,13)
x_test = x_test.reshape(-1, 144,13)
test = test.reshape(1, 144, 13)

model = Sequential()
# model.add(GRU(64, input_shape=(144,13), activation='relu'))
model.add(Bidirectional(GRU(64, activation='relu'), input_shape=(144,13)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(144))


es = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1,
    patience=20, restore_best_weights=True
)

mcp_path = './_data/kaggle/jena/mcp/'
date = datetime.datetime.now()
date = date.strftime('%H%M')
name = '({epoch:04d}_{val_loss:.4f}).hdf5'
f_path = ''.join([mcp_path, 'k56_', date, '_', name])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min', verbose=1,
    save_best_only=True, filepath=f_path
)

s_time = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, verbose=1, 
          epochs=1000, batch_size=365,
          validation_split=0.2, callbacks=[es, mcp])
e_time = time.time()


loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('loss:', loss)
print('#####Jena_CPU#####')
print('time:', np.round(e_time - s_time, 1), 'sec')
print('rmse:', np.round(rmse, 6))
print('r2_s:', np.round(r2, 6))


y_pred= model.predict(test)
y_pred = y_pred%360
y_pred = y_pred.reshape(144,1)
# print(y_pred)
submit_csv['wd (deg)'] = y_pred
submit_path = './_data/kaggle/jena/submission/'
submit_name = ''.join([submit_path, 'k57_', date, '_jena.csv'])
submit_csv.to_csv(submit_name, index=False)


import os

# 모델 체크포인트 저장 디렉토리
mcp_dir = './_data/kaggle/jena/mcp/'

# 디렉토리 내 파일 목록 중 .hdf5 파일만 필터링
files = [f for f in os.listdir(mcp_dir) if f.endswith('.hdf5')]

# val_loss 값 추출 및 (파일명, loss값) 튜플로 리스트 생성
file_losses = []
for f in files:
    try:
        # 파일명에서 loss 추출: (0034_0.1234).hdf5 → 0.1234
        loss_str = f.split('_')[-1].replace(').hdf5', '')
        val_loss = float(loss_str)
        file_losses.append((f, val_loss))
    except Exception as e:
        print(f"⚠️ 파일 해석 실패: {f}, 이유: {e}")

# 가장 val_loss가 낮은 파일 찾기
best_file = min(file_losses, key=lambda x: x[1])[0]

# 다른 파일들 삭제
for f, _ in file_losses:
    if f != best_file:
        os.remove(os.path.join(mcp_dir, f))
        print(f"🗑️ 삭제됨: {f}")

print(f"✅ 최종 남은 파일: {best_file}")


#####Jena_CPU#####
# 1차(Gru))
# time: 68.8 sec
# rmse: 80.947834
# r2_s: 0.1282

# loss: 4448.5400390625
# time: 3266.4 sec
# rmse: 66.697377
# r2_s: 0.40224

# loss: 6547.55615234375
# time: 246.0 sec
# rmse: 80.916971
# r2_s: 0.133507







