import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_random_seed(777)

#1. 데이터
# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]

x_data = [
    [77, 51, 65],
    [92, 98, 11],
    [89, 31, 33],
    [99, 33, 100],
    [73, 66, 70]
]

y_data = [[152.], [185.], [180.], [196.], [142.]]

# x1 = tf.placeholder(tf.float32, shape=[None])
# x2 = tf.placeholder(tf.float32, shape=[None])
# x3 = tf.placeholder(tf.float32, shape=[None])

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

# w1 = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
# w2 = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
# w3 = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

w = tf.Variable(tf.random.normal([3,1]), name='weights')
b = tf.Variable(0, dtype=tf.float32)

# hypo = x * w + b
hypo = tf.compat.v1.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypo - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(loss)

loss_val_list = []
w_val_list = []
epochs=10000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs) :
        _, val_loss, val_w, val_b = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data, y:y_data})
        if step % 100 == 0 :
            print(f"{step+1} | Val Loss {val_loss} | Val Bias {val_b}")
        
        loss_val_list.append(val_loss)
        w_val_list.append(val_w)
    
    y_pred = np.matmul(x_data, val_w) + val_b
    print(y_pred)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_pred)
mae = mean_absolute_error(y_data, y_pred)
print(r2, mae)  
# [[151.88835442]
#  [184.77517273]
#  [180.32112669]
#  [195.6966927 ]
#  [142.42089471]]
# 0.9997931168907065 0.27636031024158003