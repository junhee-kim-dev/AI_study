import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import random

SEED=333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# print(tf.__version__) 2.15.0

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
#  dense_1 (Dense)             (None, 2)                 8         
#  dense_2 (Dense)             (None, 1)                 3         
# =================================================================
# Total params: 17 (68.00 Byte)
# Trainable params: 17 (68.00 Byte)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.8516406 , -0.52920127, -0.9112464 ]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 

# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy= array([[ 0.6380346 , -1.0862008 ], [-0.38601977, -0.21482044], [ 0.20393836, -0.87937653]], dtype=float32)>, 
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 

# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy= array([[-0.5044266 ], [ 0.90831745]], dtype=float32)>, 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# print("=======================")
# print(model.trainable_weights)

print(len(model.weights))               #6
print(len(model.trainable_weights))     #6

print("=======================")
################# 동결 ##################
model.trainable = False         # 가중치를 갱신하지 않겠다.
################# 동결 ##################

print(len(model.weights))               #6
print(len(model.trainable_weights))     #0

print("=======================")
print(model.weights)

print("=======================")
print(model.trainable_weights)  # []


model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
#  dense_1 (Dense)             (None, 2)                 8         
#  dense_2 (Dense)             (None, 1)                 3         
# =================================================================
# Total params: 17 (68.00 Byte)
# Trainable params: 0 (0.00 Byte)
# Non-trainable params: 17 (68.00 Byte)
# _________________________________________________________________


