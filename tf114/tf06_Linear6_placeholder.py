import numpy as np
import tensorflow as tf


#1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델구성
# y = x*w +b
hypothesis = x * w + b

#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())

    # model.fit(x, y, epochs=30) 
    epochs=100
    for step in range(epochs) :
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        # sess.run(train, feed_dict={})
        if step%10 ==0 :
            print(step, loss_val, w_val, b_val)


# sess.close()
