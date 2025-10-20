import numpy as np
import tensorflow as tf
tf.random.set_random_seed(777)

var = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')

print(var)  # <tf.Variable 'weights:0' shape=(2,) dtype=float32_ref>

# 초기화 1
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(var)
print("aaa:", aaa)
sess.close()

# 초기화 2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = var.eval(session=sess)    # 텐서플로 데이터형인 'var'을 파이썬에서 쓸 수 있게 바꿔준다.
print("bbb:", bbb)
sess.close()

# 초기화 3
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = var.eval()
print("ccc:", ccc)
sess.close()



