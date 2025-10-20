from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1/255.
)

path = './_data/tensor_cert/horse_or_human/'

train = train_datagen.flow_from_directory(
    path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=True
)

all_x = []
all_y = []
for i in range(len(train)):
    x_bat, y_bat = train[i]
    all_x.append(x_bat)
    all_y.append(y_bat)
    
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

npy_path = './_data/_save_npy/keras46/horse/'
np.save(npy_path + '(100,100)x.npy', arr=x)
np.save(npy_path + '(100,100)y.npy', arr=y)