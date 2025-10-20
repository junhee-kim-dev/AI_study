import numpy as np
import tensorflow as tf

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

test_data = [6,7,8]
test = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델구성
hypothesis = x * w + b

#3. 컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs=2000
    for step in range(epochs) :
        sess.run(train)
        if step%100 ==0 :
            print(step+1, loss.eval(session=sess), w.eval(session=sess), 
                  b.eval(session=sess), hypothesis.eval(session=sess))
            
    #4. 평가 예측
    #1) 불편한 방식
    y_pred = sess.run(test * w + b, feed_dict={test:test_data})
    print(y_pred)
    #2) 편한 방식
    y_pred_comfy = test_data * w + b
    print(y_pred_comfy)


