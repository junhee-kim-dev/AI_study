from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

orginal_datagen = ImageDataGenerator(
    rescale=1/255.
)

path = './_data/tensor_cert/rps/'
train = orginal_datagen.flow_from_directory(
    path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=False
)

all_x = []
all_y = []
for i in range(len(train)):
    xb, yb = train[i]
    all_x.append(xb)
    all_y.append(yb)    
original_x = np.concatenate(all_x, axis=0)
original_y = np.concatenate(all_y, axis=0)

print(original_x.shape) #(2048, 100, 100, 3)
print(original_y.shape) #(2048,)

# exit()

augment_datagen = ImageDataGenerator(
    rescale=1/255., horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.1, fill_mode='nearest'
)

augment_size = 952
ran_idx = np.random.randint(original_x.shape[0], size=augment_size)
aug_x = original_x[ran_idx].copy()
aug_y = original_y[ran_idx].copy()


path_save = 'c:/STUDY25/_data/_save_img/07_rps/'
import time
s_time = time.time()
print('#####저장 시작#####')

augment = augment_datagen.flow(
    aug_x, aug_y, batch_size=100, shuffle=False, save_to_dir=path_save
)

arr_aug_x = []
arr_aug_y = []
for i in range(10):
    ax, ay = augment[i]
    arr_aug_x.append(ax)
    arr_aug_y.append(ay)
    
        
print('#####저장 끝#####')
e_time = time.time()
print('time :', np.round(e_time - s_time,1), 'sec')
    
augment_x = np.concatenate(arr_aug_x, axis=0)
augment_y = np.concatenate(arr_aug_y, axis=0)

x = np.concatenate((original_x, augment_x))
y = np.concatenate((original_y, augment_y))

npy_path = './_data/_save_npy/keras50/rps/'
np.save(npy_path + 'x.npy', arr=x)
np.save(npy_path + 'y.npy', arr=y)






