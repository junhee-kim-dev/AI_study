import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
tf.random.set_random_seed(777)

x_data, y_data = fetch_california_housing(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=777, test_size=0.3
)
# print(x_train.shape, y_train.shape)
# (14448, 8) (14448,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([8,1]), dtype=tf.float32)
b = tf.compat.v1.Variable(0, dtype=tf.float32)
lr = tf.compat.v1.Variable(0.1, dtype=tf.float32)

hypo = x @ w + b

loss = tf.reduce_mean(tf.square(hypo - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)

epochs = 100000
threshold = np.inf
counts = 0
reduce_counts = 0
decay = tf.compat.v1.assign(lr, lr*0.5)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs):
        _, val_w, val_b= sess.run([train, w, b],
                     feed_dict ={
                         x:x_train, y:y_train
                     }
                     )
        
        val_hypo = x_test @ val_w + val_b
        
        val_rmse = np.sqrt(mean_squared_error(val_hypo, y_test))
        
        if step % 100 == 0 :
            print(f"{step+1}\tepochs | Val RMSE {val_rmse:.6f}")
            
        if val_rmse < threshold :
            threshold = val_rmse
            counts = 0
            
            best_rmse = val_rmse
            best_step = step +1
        else :
            counts +=1
            reduce_counts +=1
            
            if reduce_counts == 10 :
                sess.run(decay)
                print(f"Reduce Learning Rate {sess.run(lr)}")
                reduce_counts = 0
            
            if counts == 100:
                print(f"\nEarly Stopping Triggered - {step+1} epochs")
                print(f"Best Step {best_step} | RMSE {best_rmse}\t")
                break
            else:
                continue


