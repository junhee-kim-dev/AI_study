import tensorflow as tf
tf.random.set_random_seed(555)

import matplotlib.pyplot as plt

x_data = [
    1, 2, 3, 4, 5
]
y_data = [
    3, 5, 7, 9, 11
]
test_data = [
    6, 7, 8
]
test = tf.compat.v1.placeholder(tf.float32, shape=[None])

x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

hypo = x * w + b

loss = tf.reduce_mean(tf.square(hypo - y))
#################### optimizer 정의 ####################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

# loss = (hypo - y)**2
# 미분 -> 2(x*w -y) x
gradient =  tf.reduce_mean(2 * (x * w-y) * x)
lr = 0.02
descent = w - lr * gradient

train = w.assign(descent)

epochs=1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs) :
        _, val_loss, val_w, val_b = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(f"{step+1} | Val Loss {val_loss} | Val Weight {val_w} | Val Bias {val_b}")
        
    y_pred = test * val_w + val_b
    print("pred : ",sess.run(y_pred, feed_dict={test:test_data}))




