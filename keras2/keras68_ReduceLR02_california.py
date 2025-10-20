import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler
from keras.callbacks import Callback
import numpy as np

class TerminateOnNaN(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and np.isnan(loss):
            print("\n🛑 Loss became NaN. Stopping training.")
            self.model.stop_training = True
            
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))
#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from keras.optimizers import Adam, Adagrad, SGD
# optim = [Adam, Adagrad, SGD]
# lr = [0.1, 0.01, 0.001, 0.0001]

# result_log = []


#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss', mode='min',
    patience=6, verbose=1, factor=0.5
)

hist = model.fit(
        x_train, y_train, epochs=100000, 
        batch_size=32, verbose=2, validation_split=0.2,
        callbacks=[es, rlr],
        )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
# 결과 저장
print(f"R2 {r2:.4f} | RMSE {rmse:.4f}")
