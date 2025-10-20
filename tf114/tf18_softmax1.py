import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.random.set_random_seed(777)

x_data = [
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,6,7],
]
y_data = [
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [1,0,0],
    [1,0,0],
]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4,3]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

hypo = tf.nn.softmax(x @ w + b)
de = 2.00e-10
loss = -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(hypo+de), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

epochs = 10000
patience = 100
best_acc = 0
best_loss = 0
best_step = 0
threshold = np.inf

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs) :
        _, train_loss, y_prob = sess.run([train, loss, hypo], feed_dict={x:x_data, y:y_data})
        y_pred = np.round(y_prob)

        val_acc = accuracy_score(y_data, y_pred)
        
        if step % 1000 ==0 :
            print(f"{step+1}\tEpochs | Loss {train_loss:.6f}\t| Val ACC {val_acc:.6f}")
            
        if train_loss < threshold :
            threshold = train_loss
            best_loss = train_loss
            best_acc = val_acc
            best_step = step+1
            patience=0
        else :
            patience+=1
            if patience==1000:
                print(f"\nEarly Stopping Triggered - {step+1} epochs")
                print(f"Best Epochs {best_step}\t| Val ACC {val_acc:.6f}")
                break

print(y_pred)
# 9001    Epochs | Loss 0.028374  | Val ACC 1.000000
