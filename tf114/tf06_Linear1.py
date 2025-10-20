import numpy as np
import tensorflow as tf


#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)

#2. 모델구성
# y = x*w +b
hypothesis = x * w + b

#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.097)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# model.fit(x, y, epochs=30) 
epochs=10000
for step in range(epochs) :
    sess.run(train, feed_dict={})
    if step%100 ==0 :
        print(step+1, sess.run(loss), sess.run(w), sess.run(b), sess.run(hypothesis))

sess.close()
