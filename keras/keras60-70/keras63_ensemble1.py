from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516), range(249,255)]).T

x1_datasets = np.array([range(100), range(301,401)]).T
# print(x1_datasets.shape)    #(100, 2)
# 삼성전자 종가, 하이닉스 종가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
# print(x2_datasets.shape)    #(100, 3)
# 원유, 환율, 금시세

y = np.array(range(2001,2101))
# print(y.shape)              #(100,)
# 화성의 화씨 온도

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=42, shuffle=True
)
# print(x1_train.shape, x1_test.shape)    #(70, 2) (30, 2)
# print(x2_train.shape, x2_test.shape)    #(70, 3) (30, 3)
# print(y_train.shape, y_test.shape)      #(70,) (30,)

ss = StandardScaler()
ss.fit(x1_train)
x1_train = ss.transform(x1_train)
x1_test = ss.transform(x1_test)

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
from keras.layers import concatenate, Concatenate                             #1) layers에도 있고
# from tensorflow.python.keras.layers.merge import concatenate, Concatenate   #2) python.keras.layers.merge에도 있음
merge1 = concatenate([o_put1, o_put2], name='mg1')

merge2 = Dense(40, name='mg2', activation='relu')(merge1)
merge3 = Dense(20, name='mg3', activation='relu')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[i_put1, i_put2], outputs=last_output)
# model.summary()
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to                  
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 2)]                  0         []                                                                                                                              
#  ibm1 (Dense)                (None, 10)                   30        ['input_1[0][0]']             
#  ibm2 (Dense)                (None, 20)                   220       ['ibm1[0][0]']                
#  input_2 (InputLayer)        [(None, 3)]                  0         []                            
#  ibm3 (Dense)                (None, 30)                   630       ['ibm2[0][0]']                
#  ibm21 (Dense)               (None, 100)                  400       ['input_2[0][0]']             
#  ibm4 (Dense)                (None, 40)                   1240      ['ibm3[0][0]']                
#  ibm22 (Dense)               (None, 50)                   5050      ['ibm21[0][0]']               
#  ibm5 (Dense)                (None, 50)                   2050      ['ibm4[0][0]']                
#  ibm23 (Dense)               (None, 30)                   1530      ['ibm22[0][0]']               
#  mg1 (Concatenate)           (None, 80)                   0         ['ibm5[0][0]','ibm23[0][0]']           
#  mg2 (Dense)                 (None, 40)                   3240      ['mg1[0][0]']                 
#  mg3 (Dense)                 (None, 20)                   820       ['mg2[0][0]']                 
#  last (Dense)                (None, 1)                    21        ['mg3[0][0]']                 
# ==================================================================================================
# Total params: 15231 (59.50 KB)
# Trainable params: 15231 (59.50 KB)
# Non-trainable params: 0 (0.00 Byte)
# __________________________________________________________________________________________________

#3. 컴파일 훈련
es = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=150,
    restore_best_weights=True, start_from_epoch=20,
)
filepath = './_save/keras63/ensemble.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss', mode='min', verbose=1,
    save_best_only=True, filepath=filepath
)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit([x1_train, x2_train], y_train, epochs=100000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, mcp])

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
results = model.predict([x1_pred, x2_pred])
rmse = np.sqrt(loss)

print('Loss :', np.round(loss, 5))
print('rmse :', np.round(rmse, 5))
print('results :', results)

# Loss : 1.03705
# results : [[2111.345 ]
#  [2116.054 ]
#  [2120.7642]
#  [2125.4758]
#  [2130.1868]
#  [2134.8984]]

# Loss : 0.58324
# rmse : 0.7637
# results : [[2098.3953]
#  [2100.8242]
#  [2103.2732]
#  [2105.7407]
#  [2108.235 ]
#  [2110.7295]]

# Loss : 0.03842
# rmse : 0.196
# results : [[2098.1184]
#  [2101.354 ]
#  [2104.6233]
#  [2107.9172]
#  [2111.2979]
#  [2114.7124]]

# Loss : [0.02447 0.10717]
# rmse : [0.15641 0.32736]
# results : [[-3095.5217]
#  [-3179.597 ]
#  [-3263.672 ]
#  [-3347.7498]
#  [-3431.8232]
#  [-3515.8987]]




