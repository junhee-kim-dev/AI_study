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
# x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T
# x3_pred = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

x1_datasets = np.array([range(100), range(301,401)]).T
# print(x1_datasets.shape)    #(100, 2)
# x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
# # print(x2_datasets.shape)    #(100, 3)
# x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T
# print(x3_datasets.shape)    #(100, 3)
# exit()

y1 = np.array(range(2001,2101))
# print(y1.shape)              #(100,)
y2 = np.array(range(13001,13101))
# print(y2.shape)              #(100,)


x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
                   x1_datasets, y1, y2, 
                   train_size=0.7, random_state=42, shuffle=True
                   )
# print(x1_train.shape, x1_test.shape)    #(70, 2) (30, 2)
# print(x2_train.shape, x2_test.shape)    #(70, 3) (30, 3)
# print(x3_train.shape, x3_test.shape)    #(70, 4) (30, 4)
# print(y1_train.shape, y1_test.shape)      #(70,) (30,)
# print(y2_train.shape, y2_test.shape)      #(70,) (30,)

mms = MinMaxScaler()
mms.fit(x1_train)
x1_train = mms.transform(x1_train)
x1_test = mms.transform(x1_test)
x1_pred = mms.transform(x1_pred)

# x3_train = x3_train.reshape(-1,2,2)
# x3_test = x3_test.reshape(-1,2,2)
# x3_pred = x3_pred.reshape(-1,2,2)

#2 모델 구성
#2-1 모델
i_put1 = Input(shape=(2,))
dense1 = Dense(64, activation='relu', name='ibm11')(i_put1)
dense2 = Dense(128, activation='relu', name='ibm12')(dense1)
dense3 = Dense(64, activation='relu', name='ibm13')(dense2)
o_put1 = Dense(32, activation='relu', name='ibm14')(dense3)  # 앙상블 모델에서 모델 하나의 아웃풋레이어는 엄밀히 따지면 전체 모델의 히든임
# model1 = Model(inputs=i_put1, outputs=o_put1)             # 앙상블 모델에서는 하위 모델의 인풋 아웃풋을 지정할 필요가 없다

#2-4 모델
from keras.layers import concatenate, Concatenate                             #1) layers에도 있고
# from tensorflow.python.keras.layers.merge import concatenate, Concatenate   #2) python.keras.layers.merge에도 있음

merge1_2 = Dense(40, name='mg1_2', activation='relu')(o_put1)
merge1_3 = Dense(20, name='mg1_3', activation='relu')(merge1_2)
last_output_1 = Dense(1, name='last1')(merge1_3)

#2-5 모델
merge2_2 = Dense(40, name='mg2_2', activation='relu')(o_put1)
merge2_3 = Dense(20, name='mg2_3', activation='relu')(merge2_2)
last_output_2 = Dense(1, name='last2')(merge2_3)

model = Model(inputs=[i_put1], outputs=[last_output_1, last_output_2])



es = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=150,
    restore_best_weights=True, start_from_epoch=200,
)
filepath = './_save/keras63/ensemble.hdf5'
mcp = ModelCheckpoint(
    monitor='val_', mode='min', verbose=1,
    save_best_only=True, filepath=filepath
)

model.compile(loss='mse', optimizer='adam')
model.fit([x1_train], [y1_train, y2_train], epochs=1000, batch_size=64, verbose=1, 
          validation_split=0.2, callbacks=[es])

#4. 평가 예측
loss = model.evaluate([x1_test], [y1_test, y2_test])
results = model.predict([x1_pred])
rmse = np.sqrt(loss)

print('Loss :', np.round(loss, 5))
print('rmse :', np.round(rmse, 5))
print('results :', results)

# Loss : [571.12909 558.17609  12.95298]
# rmse : [23.89831 23.62575  3.59903]
# results : 
# [array([[2061.5698],
#         [2063.3096],
#         [2065.0496],
#         [2066.789 ],
#         [2068.529 ],
#         [2070.269 ]], dtype=float32), 
#  array([[13120.788],
#         [13131.861],
#         [13142.936],
#         [13154.008],
#         [13165.084],
#         [13176.156]], dtype=float32)]