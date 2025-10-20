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
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        if step%100 ==0 :
            print(step, loss_val, w_val, b_val)
    #4. 평가 예측
    #1) 불편한 방식
    y_pred = sess.run(test * w_val + b_val, feed_dict={test:test_data})
    print(y_pred)
    #2) 편한 방식
    y_pred_comfy = test_data * w_val + b_val
    print(y_pred_comfy)

# 0 228.52231 [-0.8366468] [-0.317366]
# 100 0.18836911 [2.2808228] [0.986141]
# 200 0.09568541 [2.2001474] [1.2774042]
# 300 0.048604943 [2.1426485] [1.4849931]
# 400 0.02468965 [2.1016681] [1.6329457]
# 500 0.012541523 [2.0724607] [1.738394]
# 600 0.0063706413 [2.051644] [1.8135488]
# 700 0.0032361161 [2.0368078] [1.8671124]
# 800 0.0016438371 [2.0262334] [1.9052886]
# 900 0.00083501154 [2.0186968] [1.9324979]
# 1000 0.00042415806 [2.0133257] [1.9518899]
# 1100 0.00021545282 [2.0094974] [1.9657114]
# 1200 0.000109439636 [2.006769] [1.9755622]
# 1300 5.5592187e-05 [2.0048244] [1.9825824]
# 1400 2.824128e-05 [2.0034387] [1.9875855]
# 1500 1.4348926e-05 [2.0024512] [1.9911512]
# 1600 7.289309e-06 [2.0017471] [1.9936929]
# 1700 3.7043465e-06 [2.0012453] [1.995504]
# 1800 1.8821831e-06 [2.0008874] [1.9967955]
# 1900 9.561563e-07 [2.000633] [1.9977156]
# [14.001081 16.001534 18.001986]