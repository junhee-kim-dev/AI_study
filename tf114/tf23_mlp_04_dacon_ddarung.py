import tensorflow as tf
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
tf.random.set_random_seed(777)

x_data, y_data = load_diabetes(return_X_y=True)

path ='./backups/Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)

import random
r = 7275 #random.randint(1,10000)     # 7275, 208, 6544, 1850, 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=r
)

y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([9, 32]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([32]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([32, 16]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([16]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([16, 1]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.matmul(x, w1) + b1
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.matmul(drop1, w2) + b2
hypo = tf.matmul(layer2, w3) + b3

loss = tf.reduce_mean(tf.square(hypo - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

epochs=1000000
threshold = np.inf
counts = 0
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs):
        _, trnloss= sess.run([train, loss],
                    feed_dict = {
                        x:x_train, y:y_train, dr:0.3
                        }
                    )
        val_loss, y_pred = sess.run([loss, hypo], feed_dict={
                        x:x_test, y: y_test, dr:0.3
                        })
        val_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        val_r2 = r2_score(y_test, y_pred)
        
        if step % 100 == 0 :
            print(step+1, f"\tepochs | Loss {trnloss:.6f}\t| R2 {val_r2:.6f}\t| RMSE {val_rmse:.6f}")
        
        if val_loss < threshold :
            threshold = val_loss
            best_steps = step
            best_rmse = val_rmse
            best_r2 = val_r2
            
            counts = 0
        else :
            counts += 1
            
            if counts == 2000 :
                print(f"\nEarly Stopping Triggered - {step+1} epoch")
                print("Best Step", best_steps+1, f"\tepochs | RMSE {best_rmse:.6f}\t| R2 {best_r2:.6f}")
                break
            else :
                continue


# Early Stopping Triggered - 8847 epoch
# Best Step 6847  epochs | RMSE 53.077434 | R2 0.610319