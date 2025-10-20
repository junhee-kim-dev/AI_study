import tensorflow as tf
print(tf.__version__)   # 2.7.3

gpus = tf.config.list_physical_devices('GPU')
print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
