from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T
x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

x1_datasets = np.array([range(100), range(301,401)]).T
# print(x1_datasets.shape)    #(100, 2)

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
# print(x2_datasets.shape)    #(100, 3)

x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T
# print(x3_datasets.shape)    #(100, 3)
# exit()

y = np.array(range(2001,2101))
# print(y.shape)              #(100,)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size=0.7, random_state=42, shuffle=True
)
print(x1_train.shape, x1_test.shape)    #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    #(70, 3) (30, 3)
print(x3_train.shape, x3_test.shape)    #(70, 4) (30, 4)
print(y_train.shape, y_test.shape)      #(70,) (30,)

mms = MinMaxScaler()
mms.fit(x1_train)
x1_train = mms.transform(x1_train)
x1_test = mms.transform(x1_test)
mms.fit(x2_train)
x2_train = mms.transform(x2_train)
x2_test = mms.transform(x2_test)
mms.fit(x3_train)
x3_train = mms.transform(x3_train)
x3_test = mms.transform(x3_test)

x3_train = x3_train.reshape(-1,2,2)
x3_test = x3_test.reshape(-1,2,2)
x3_pred = x3_pred.reshape(-1,2,2)

#2 모델 구성
#2-1 모델
i_put1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(i_put1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
o_put1 = Dense(50, activation='relu', name='ibm5')(dense4)  # 앙상블 모델에서 모델 하나의 아웃풋레이어는 엄밀히 따지면 전체 모델의 히든임
# model1 = Model(inputs=i_put1, outputs=o_put1)             # 앙상블 모델에서는 하위 모델의 인풋 아웃풋을 지정할 필요가 없다

#2-2 모델
i_put2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(i_put2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
o_put2 = Dense(30, activation='relu', name='ibm23')(dense22)
# model2 = Model(inputs=i_put2, outputs=dense23)

#2-3 모델
i_put3 = Input(shape=(2,2))
lstm31 = LSTM(32, activation='relu')(i_put3)
dense31 = Dense(128, activation='relu')(lstm31)
dense32 = Dense(64, activation='relu')(dense31)
o_put3 = Dense(32, activation='relu')(dense32)

#2-4 모델
from keras.layers import concatenate, Concatenate                             #1) layers에도 있고
# from tensorflow.python.keras.layers.merge import concatenate, Concatenate   #2) python.keras.layers.merge에도 있음
merge1 = Concatenate(axis=1)([o_put1, o_put2, o_put3]) 

merge2 = Dense(40, name='mg2', activation='relu')(merge1)
merge3 = Dense(20, name='mg3', activation='relu')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[i_put1, i_put2, i_put3], outputs=last_output)


es = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=150,
    restore_best_weights=True, start_from_epoch=200,
)
filepath = './_save/keras63/ensemble.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss', mode='min', verbose=1,
    save_best_only=True, filepath=filepath
)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=10000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, mcp])

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
results = model.predict([x1_pred, x2_pred, x3_pred])
rmse = np.sqrt(loss)

print('Loss :', np.round(loss, 5))
print('rmse :', np.round(rmse, 5))
print('results :', results)


# Loss : 0.00379
# rmse : 0.0616
# results : [[250050.8 ]
#  [251013.36]
#  [251977.11]
#  [252942.05]
#  [253907.95]
#  [254874.77]]


# Loss : [0.00472 0.05028]
# rmse : [0.06869 0.22422]
# results : [[282884.44]
#  [283484.44]
#  [284006.72]
#  [284414.78]
#  [284643.56]
#  [284576.84]]