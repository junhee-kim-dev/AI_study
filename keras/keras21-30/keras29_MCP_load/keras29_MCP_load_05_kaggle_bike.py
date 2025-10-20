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

from tensorflow.python.keras.models import load_model
path1 = './_save/keras28_mcp/05_kaggle_bike/'
model = load_model(path1 + 'k28_0604_1214_0073-9062.0605.hdf5')


loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)

y_submit = model.predict(test_csv)
test_csv_copy = test_csv.copy()
test_csv_copy[['casual', 'registered']] = y_submit
test_csv_copy.to_csv(path + 'new_test_1.csv', index=False)

# RMSE : 95.68644369926442
# R2 : 0.3849638144401447
# 걸린 시간 : 15.975741624832153 초

# ???????????? 왜 다르지..?
# RMSE : 95.80376468659779
# R2 : 0.3974316436084781