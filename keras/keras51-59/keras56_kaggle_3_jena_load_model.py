import numpy as np
import pandas as pd

npy_path = './_data/kaggle/jena/dataset/'
x = np.load(npy_path + '(144steps)x.npy')
y = np.load(npy_path + '(144steps)y.npy')
test = np.load(npy_path + '(144steps)test.npy')
lets_test = np.load(npy_path + 'clean_feat_(144steps_36st)lets_test.npy')
submit_csv = pd.read_csv('./_data/kaggle/jena/submission_file.csv')

# print(x)
# print(y)
# print(test)
# print(x.shape)              #(69996, 144, 13)
# print(y.shape)              #(69996, 144)
# print(test.shape)           #(144, 13)
# exit()
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
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

model_path = './_data/kaggle/jena/'
model = load_model(model_path + '2차/k56_1846_(0309_4446.7871).hdf5')

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
lets_test = np.array(lets_test)
rmse = RMSE(lets_test, results)

print('#####Jena_load_model#####')
print('loss:', loss)
print('rmse:', np.round(rmse, 6))
print('r2_s:', np.round(r2, 6))



