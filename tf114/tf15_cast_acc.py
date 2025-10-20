import tensorflow as tf
aaa = tf.constant([0.3, 0.4, 0.8, 0.9])
bbb = tf.constant([0., 1., 1., 1.])

sess = tf.compat.v1.Session()

pred = tf.cast(aaa >= 0.5, dtype=tf.float32)
predict = sess.run(pred)

print(predict)

acc = tf.reduce_mean(tf.cast(tf.equal(pred, bbb), dtype=tf.float32))
print(sess.run(acc))

sess.close()

