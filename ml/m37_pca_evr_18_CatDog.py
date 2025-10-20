# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition
# 45.copy
##########################################################################
#0. 준비
##########################################################################
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 518
##########################################################################
#1. 데이터
##########################################################################
path_NP = 'C:/Study25/_data/kaggle/cat_dog/'

x = np.load(path_NP + '04_x_train.npy')
y = np.load(path_NP + '04_y_train.npy')

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=50,
                                           stratify=y,
                                           )

#####################################
### pred 데이터 설정
path_PD = 'C:/Study25/_data/kaggle/cat_dog/'

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

a = len(np.where(evr_cumsum >= 1.)[0]) + 1
b = len(np.where(evr_cumsum >= 0.999)[0]) + 1
c = len(np.where(evr_cumsum >= 0.99)[0]) + 1
d = len(np.where(evr_cumsum >= 0.95)[0]) + 1

num = [a,b,c,d]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostClassifier(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f}")

'''
[LSTM]
save : 0623_0_5
loss : 0.6918511986732483
acc  : 0.5257750153541565

[Conv1D]
save : 0623_0_5
loss : 0.5684945583343506
acc  : 0.7196000218391418
'''

#####################################
### 저장 설정

D = datetime.datetime.now()
D = D.strftime('%m%d')
saveNum = f'{D}_0_5'

#####################################
### callbacks

ES = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=P,
    restore_best_weights=True
)

MCP = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    filepath="".join([path_MCP,'MCP_',saveNum,'.h5']),
    save_best_only=True
)

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)

#####################################
### 가중치 불러오기
path_W = 'C:/Study25/_data/kaggle/cat_dog/weights/'
# model.load_weights(path_W + 'weights_0615_3_0.h5')


S = time.time()
H = model.fit(
    x_trn, y_trn,
    epochs = E,
    batch_size = B,
    verbose = 1,
    validation_split=0.2,
    callbacks=[ES, MCP]
)
E = time.time()
T = E-S

#####################################
### 모델 및 가중치 저장
model.save(path_S + f'save_{saveNum}.h5')
# model.save_weights(path_W + f'weights_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = np.round(model.predict(x_tst))

ACC = accuracy_score(y_tst, y_pred)

#####################################
### 그래프
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

print('save :', saveNum)
print('loss :', LSAC[0])
print('acc  :', LSAC[1])
# print('Vlss :', np.round(H.history['val_loss'][-1], 6))
# print('Vacc :', np.round(H.history['val_acc'][-1], 6))
# print('time :', T)
# plt.show()

#####################################
### 파일송출
# x_pred = np.load(path_NP + '04_x_predict.npy')

# y_pred = model.predict(x_pred)

# path = './_data/kaggle/cat_dog/'
# sub_csv = pd.read_csv(path + 'sample_submission.csv')
# sub_csv['label'] = y_pred
# sub_csv.to_csv(path + f'sub/sample_submission_{saveNum}.csv', index=False)