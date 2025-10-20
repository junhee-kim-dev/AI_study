import tensorflow as tf
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
tf.random.set_random_seed(7777)

#1. 데이터
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
y_data = [
    [0],
    [1],
    [1],
    [0]
]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,3]), name='weights')
b = tf.Variable(tf.zeros([3]), name='bias')
w2 = tf.Variable(tf.random_normal([3,4]), name='weights2')
b2 = tf.Variable(tf.zeros([4]), name='bias')
w3 = tf.Variable(tf.random_normal([4,1]), name='weights2')
b3 = tf.Variable(tf.zeros([1]), name='bias')

rate = tf.placeholder(tf.float32)

logits1 = tf.matmul(x,w)+b
hypo1 = tf.nn.relu(logits1)
logits2 = tf.matmul(hypo1,w2)+b2
hypo2 = tf.nn.relu(logits2)
################## DROP OUT #####################
drop1 = tf.nn.dropout(hypo2, rate=rate)
################## DROP OUT #####################
logits3 = tf.matmul(drop1,w3)+b3
hypo3 = tf.sigmoid(logits3)


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits3))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 900
for step in range(epochs) :
    _, loss_val, w_val, b_val = sess.run([train ,loss, w, b], feed_dict={x:x_data, y:y_data, rate:0.1})
    if step%100 ==0 :
        prob = sess.run(hypo3, feed_dict={x:x_data, rate:0.0})
        y_pred = np.round(prob)
        acc = accuracy_score(y_data, y_pred)
        print(f"Epochs {step+1}\t| Loss {loss_val:.6f}\t| ACC {acc:.6f}")
        
print(f"Epochs {step+1}\t| Loss {loss_val:.6f}\t| ACC {acc:.6f}")
print("y_pred =\n", y_pred)
