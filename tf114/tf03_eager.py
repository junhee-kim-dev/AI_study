import tensorflow as tf

print(tf.__version__)
print('즉시실행모드 :', tf.executing_eagerly())
# 1.14.0
# 즉시실행모드 : False
# 2.15.0
# 즉시실행모드 : True

tf.compat.v1.disable_eager_execution()
print('즉시실행모드 :', tf.executing_eagerly())
# 즉시실행모드 : False

# tf.compat.v1.enable_eager_execution()
print('즉시실행모드 :', tf.executing_eagerly())
# 즉시실행모드 : True

hello = tf.constant("Hello world")

sess = tf.compat.v1.Session()
print(sess.run(hello))


##################################
# 즉시 실행모드 -> 텐서1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행
# tf.compat.v1.disable_eager_execution() 끄기 텐서플로우 1.0
# tf.compat.v1.enable_eager_execution() 켜기 텐서플로우 2.0 이후

# sess.run() 실행시!
# Tensor1은 '그래프 연산' 모드
# Tensor2는 '즉시 실행' 모드

# tf.compat.v1.enable_eager_execution()
# -> Tensor2 의 default
# tf.compat.v1.disable_eager_execution()
# -> 그래프 연산모드로 돌아감
# -> Tensor 1코드를 쓸 수 있음

# tf.executing_eagerly()
# -> True : 즉시 실행 모드, Tensor 2 코드만 써야함
# -> False : 그래프 연산 모드, Tensor1 코드를 쓸 수 있음