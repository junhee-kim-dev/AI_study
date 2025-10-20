import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈
node_add = tf.add(node1, node2)
sess_1 = tf.Session()
print(sess_1.run(node_add))

# 뺄셈
node_sub = tf.subtract(node1, node2)
sess_2 = tf.Session()
print(sess_2.run(node_sub))

# 곱셈
node_mul = tf.multiply(node1, node2)
sess_3 = tf.Session()
print(sess_3.run(node_mul))

# 나눗셈
node_div = tf.divide(node1, node2)
sess_4 = tf.Session()
print(sess_4.run(node_div))
