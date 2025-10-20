import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
tf.random.set_random_seed(7777)

x_data, y_data = load_iris(return_X_y=True)

y_data = y_data.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)
y_data = oh.fit_transform(y_data)
print(x_data.shape)
print(y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=777, train_size=0.8
)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

w1 = tf.Variable(tf.random_normal([4,16]), dtype=tf.float32)
w2 = tf.Variable(tf.random_normal([16,8]), dtype=tf.float32)
w3 = tf.Variable(tf.random_normal([8,3]), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([16]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([8]), dtype=tf.float32)
b3 = tf.Variable(tf.zeros([3]), dtype=tf.float32)

dr = tf.placeholder(tf.float32)
layer1 = tf.nn.relu(x@w1+b1)
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.nn.relu(drop1@w2+b2)
hypo = tf.nn.softmax(layer2@w3+b3)
de = 1e-8
loss = -tf.reduce_mean(tf.reduce_sum(y*tf.math.log(hypo + de), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

epochs=10000
patience = 0
best_step = 0
best_acc = 0
best_loss = 0
threshold = np.inf

from sklearn.metrics import accuracy_score

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(epochs) :
        _, train_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train, dr:0.3})
        val_loss, prob = sess.run([loss, hypo], feed_dict={x:x_test, y:y_test, dr:0.0})
        y_pred = np.round(prob)
        val_acc = accuracy_score(y_test, y_pred)
        
        if step % 100 == 0:
            print(f"Epochs {step+1}\t| Loss {train_loss:.6f}\t| Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")

        if val_loss < threshold :
            threshold = val_loss
            best_acc = val_acc
            best_loss = val_loss
            best_step = step+1
            patience=0
        else :
            patience += 1
            if patience==100 :
                print(f"\nEarly Stopping Triggered - {step+1} epochs")
                print(f"Best Epochs {best_step}\t| Val Loss {best_loss:.6f}\t| Val ACC {best_acc:.6f}")
                break

# Early Stopping Triggered - 266 epochs
# Best Epochs 166 | Val Loss 0.030026     | Val ACC 1.000000