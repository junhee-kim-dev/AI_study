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

w = tf.Variable(tf.random_normal([2,1]), name='weights')
b = tf.Variable(tf.zeros([1]), name='bias')

logits = tf.matmul(x,w)+b
hypo = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 1000
for step in range(epochs) :
    _, loss_val, w_val, b_val = sess.run([train ,loss, w, b], feed_dict={x:x_data, y:y_data})
    if step%100 ==0 :
        prob = sess.run(hypo, feed_dict={x:x_data})
        y_pred = np.round(prob)
        # print(y_pred)
        acc = accuracy_score(y_data, y_pred)
        # print(acc)
        print(f"Epochs {step+1}\t| Loss {loss_val:.6f}\t| ACC {acc:.6f}")

