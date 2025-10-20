import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_covtype
tf.random.set_random_seed(7777)

x_data, y_data = fetch_covtype(return_X_y=True)

y_data = y_data.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)
y_data = oh.fit_transform(y_data)

print(x_data.shape)
print(y_data.shape)
# exit()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=7777, train_size=0.8
)

x = tf.placeholder(tf.float32, shape=[None, 54])
y = tf.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.random.normal([54, 128]), name='weights', dtype=tf.float32)
b1 = tf.compat.v1.Variable(tf.zeros([128]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random.normal([128, 64]), name='weights', dtype=tf.float32)
b2 = tf.compat.v1.Variable(tf.zeros([64]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random.normal([64,7]), name='weights', dtype=tf.float32)
b3 = tf.compat.v1.Variable(tf.zeros([7]), dtype=tf.float32)
dr = tf.placeholder(tf.float32)

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
drop1 = tf.nn.dropout(layer1, rate=dr)
layer2 = tf.nn.relu(tf.matmul(drop1, w2) + b2)
layer3 = tf.matmul(layer2, w3) + b3
hypo   = tf.nn.softmax(layer3)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer3)
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1) 
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
        _, tr_loss = sess.run([train, loss], feed_dict={x: x_train, y: y_train, dr:0.2})
        val_loss, val_logits = sess.run([loss, layer3], feed_dict={x: x_test, y: y_test, dr:0.0})

        y_pred = np.argmax(val_logits, axis=1)
        y_true = np.argmax(y_test, axis=1)    
        val_acc = accuracy_score(y_true, y_pred)
        
        if step % 100 == 0:
            print(f"Epochs {step+1}\t| Loss {tr_loss:.6f}\t| Val Loss {val_loss:.6f}\t| Val ACC {val_acc:.6f}")

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
        

# Early Stopping Triggered - 1795 epochs
# Best Epochs 1695        | Val Loss 9.430551     | Val ACC 0.657427

# Early Stopping Triggered - 277 epochs
# Best Epochs 177 | Val Loss 1.204503     | Val ACC 0.486829