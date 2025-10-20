from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

origin_train_datagen = ImageDataGenerator(
    rescale=1/255.,
)
aug_train_datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True, vertical_flip=True,
    height_shift_range=0.1, width_shift_range=0.1,
    shear_range=0.8, zoom_range=1.2,
    rotation_range=20
)

path = './_data/kaggle/men_women/'

origin_train = origin_train_datagen.flow_from_directory(
    path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=False
)
aug_train = aug_train_datagen.flow_from_directory(
    path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=True
)

all_x = []
all_y = []
for i in range(len(origin_train)):
    x_bat, y_bat = origin_train[i]
    all_x.append(x_bat)
    all_y.append(y_bat)
all_aug_x = []
all_aug_y = []
for i in range(len(aug_train)):
    x_bat, y_bat = aug_train[i]
    all_x.append(x_bat)
    all_y.append(y_bat)
    
x = np.concatenate(all_x + all_aug_x, axis=0)
y = np.concatenate(all_y + all_aug_y, axis=0)

npy_path = './_data/_save_npy/keras46/gender/'
np.save(npy_path + '(100,100)_flip_x.npy', arr=x)
np.save(npy_path + '(100,100)_flip_y.npy', arr=y)