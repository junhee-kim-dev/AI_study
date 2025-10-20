import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.special import expit

tf.compat.v1.random.set_random_seed(777)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

#2. 모델 구성
hypo = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

#3. 컴파일
loss = - tf.reduce_mean(y * tf.log(hypo) + (1 - y) * tf.log(1 - hypo))  # binary cross entropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

pred = tf.cast(hypo >= 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

epochs = 1000
counts = 0
threshold = np.inf

with tf.compat.v1.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(epochs) :
        _, val_acc, val_loss, val_w, val_b = sess.run([train, acc, loss, w, b],
                                    feed_dict ={
                                       x:x_data, y:y_data
                                        }
                                   )
        
        if step%100==0 :
            print(f"{step+1}\tepoch | Loss {val_loss:.6f}\t| Acc {val_acc:.6f}")
            print(f"y_pred = \n{np.round(expit(x_data@val_w+val_b), 5)}")

        if val_loss < threshold :
            threshold = val_loss
            counts = 0
        else :
            counts += 1
            if counts==50:
                best_w = val_w
                best_b = val_b
                best_loss = val_loss
                best_acc = val_acc
                best_step = step+1
                print(f"\nEarly Stopping Triggered - {step +1}")
                print(f"{best_step}\tepoch | Loss {val_loss:.6f}\t| Acc {acc:.6f}")
                break
            else :
                continue
