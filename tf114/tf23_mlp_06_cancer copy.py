import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import expit

import numpy as np
import pandas as pd
# print(np.__version__)   1.16.5
# print(pd.__version__)   1.2.0

x_data, y_data = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=7777, train_size=0.8, shuffle= True
)
print(x_train.shape)   # (455, 30)
print(y_train.shape)   # (455,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([30, 64]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([64]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([64, 32]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([32]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([32,1]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([1]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.matmul(x, w1) + b1
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.matmul(drop1, w2) + b2
layer3 = tf.matmul(layer2, w3) + b3
hypo   = tf.sigmoid(layer3)

# 2) 수치안정한 BCE 로스 (log(0) 방지)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer3))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

epochs=100000
counts=0
threshold=np.inf
best_step=1
best_acc = 0

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs) :
        sess.run([train], feed_dict={x: x_train, y: y_train, dr:0.4})

        y_prob, val_loss = sess.run([hypo, loss], feed_dict = {x:x_test, y:y_test, dr:0.})
        y_pred = np.round(y_prob)
        val_acc = accuracy_score(y_test, y_pred)

        if step % 100 == 0 :
            print(f"{step+1}\tepochs | Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")
        
        if val_loss < threshold :
            threshold = val_loss
            counts=0
            best_loss = val_loss
            best_acc = val_acc
            best_step = step+1
        else:
            counts +=1
            if counts==1000:
                print(f"Best Step {best_step} epochs | Best Loss {best_loss:.6f} | Best ACC {best_acc:.6f}")
                break
            else:
                continue
            
# Best Step 5541 epochs | Best Loss 0.134978 | Best ACC 0.947368
# Best Step 7174 epochs | Best Loss 0.622915 | Best ACC 0.947368