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
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

loss_val_list = []
w_val_list = []
epochs=1000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs) :
        _, val_loss, val_w, val_b = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(f"{step+1} | Val Loss {val_loss} | Val Weight {val_w} | Val Bias {val_b}")
        
        loss_val_list.append(val_loss)
        w_val_list.append(val_w)
        
    y_pred = test * val_w + val_b
    print("pred : ",sess.run(y_pred, feed_dict={test:test_data}))

print("="*10, " 그림그리기 ", "="*10)
# loss와 epoch와의 관계

# plt.plot(loss_val_list)
# plt.show()

# w와 epoch와의 관계
# plt.plot(w_val_list)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('w')
# plt.show()

# w와 loss와의 관계
# plt.plot(w_val_list, loss_val_list)
# plt.xlabel('w')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(loss_val_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.subplot(2,2,2)
plt.plot(w_val_list)
plt.xlabel('epoch')
plt.ylabel('w')
plt.grid()

plt.subplot(2,1,2)
plt.plot(w_val_list, loss_val_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.grid()

plt.tight_layout()
plt.show()


