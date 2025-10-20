from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(10,10,1),
                 strides=1, padding='same', # default ='valid' 
                 ))
model.add(Conv2D(9, (3,3),
                 strides=1,
                 padding='valid'
                 ))
model.add(Conv2D(8, 4))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 10)        50        
#  conv2d_1 (Conv2D)           (None, 8, 8, 9)           819       
#  conv2d_2 (Conv2D)           (None, 5, 5, 8)           1160      
# =================================================================
# Total params: 2,029
# Trainable params: 2,029
# Non-trainable params: 0
# _________________________________________________________________