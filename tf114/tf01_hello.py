import tensorflow as tf
print(tf.__version__)   #1.14

# python : 3.7.16

# pip install protobuf==3.20
# pip install numpy==1/16

print("hello world")
hello = tf.constant("hello world")
print(hello)

sess = tf.Session()
print(sess.run(hello))  #b'hello world'
