# [실습]
# 100,100,3 짜리 이미지를 10,10,11으로 만들어보자

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(5, (2,2), input_shape=(100,100,3), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(7, (2,2), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(9, (2,2)))
model.add(MaxPooling2D())
model.add(Conv2D(11, (3,3)))
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 100, 100, 5)       65
#  max_pooling2d (MaxPooling2D  (None, 50, 50, 5)        0
#  )
#  conv2d_1 (Conv2D)           (None, 50, 50, 7)         147
#  max_pooling2d_1 (MaxPooling  (None, 25, 25, 7)        0
#  2D)
#  conv2d_2 (Conv2D)           (None, 24, 24, 9)         261
#  max_pooling2d_2 (MaxPooling  (None, 12, 12, 9)        0
#  2D)
#  conv2d_3 (Conv2D)           (None, 10, 10, 11)        902
# =================================================================
# Total params: 1,375
# Trainable params: 1,375
# Non-trainable params: 0
# _________________________________________________________________