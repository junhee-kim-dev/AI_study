from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# print(x.shape)  #(25000, 100, 100, 3)
# print(y.shape)  #(25000,)

### 원본 ###
datagen = ImageDataGenerator(
    rescale=1/255.
)

path = './_data/kaggle/CatDog/'
train_path = ''.join([path, 'train2/'])
test_path = ''.join([path, 'test2/'])

train = datagen.flow_from_directory(
    train_path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=True
)
test = datagen.flow_from_directory(
    train_path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=True
)

all_x = []
all_y =[]
test_x = []
test_y = []

for i in range(len(train)):
    x_bat, y_bat = train[i]
    all_x.append(x_bat)
    all_y.append(y_bat)

for i in range(len(test)):
    test_bat, t = test[i]
    test_x.append(test_bat)
    test_y.append(t)
    
origin_x = np.concatenate(all_x, axis=0)
origin_y = np.concatenate(all_y, axis=0)
test = np.concatenate(test_x, axis=0)

aug_datagen = ImageDataGenerator(
    rescale=1/255., horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.1, fill_mode='nearest'
)

augment_size = 15000
ran_idx = np.random.randint(25000, size=15000)
aug_x = origin_x[ran_idx].copy()
aug_y = origin_y[ran_idx].copy()

path_save = 'c:/STUDY25/_data/_save_img/05_catdog/'
import time
s_time = time.time()
print('#####저장 시작#####')

augment = aug_datagen.flow(
    aug_x, aug_y, batch_size=100, shuffle=False, save_to_dir=path_save
)

all_aug_x =[] 
for i in range(len(aug_x)//100):
    aug_x_bat, aug_y_bat = augment[i]
    all_aug_x.append(aug_x_bat)

print('#####저장 끝#####')
e_time = time.time()
print('time :', np.round(e_time - s_time,1), 'sec')
x_augment = np.concatenate(all_aug_x)

x = np.concatenate((origin_x, x_augment))
y = np.concatenate((origin_y, aug_y))

npy_path = './_data/_save_npy/keras50/catdog/'
np.save(npy_path + 'x.npy', arr=x)
np.save(npy_path + 'y.npy', arr=y)
np.save(npy_path + 'test.npy', arr=test)






