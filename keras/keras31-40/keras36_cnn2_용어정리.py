from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
                                                      #(세로, 가로, 색깔)
model = Sequential()       # 가중치 크기           #(height, width, channels)
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))
model.add(Conv2D(5, (2,2)))
model.add(Flatten())
model.add(Dense(units=10))  # input = (batch, input_dim)
model.add(Dense(3))

model.summary()