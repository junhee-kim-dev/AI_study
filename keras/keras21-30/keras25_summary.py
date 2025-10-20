from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary()     # 총 연산량을 보여줌
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6
#  dense_1 (Dense)             (None, 2)                 8
#  dense_2 (Dense)             (None, 4)                 12
#  dense_3 (Dense)             (None, 1)                 5
# =================================================================
# Total params: 31
# Trainable params: 31
# Non-trainable params: 0